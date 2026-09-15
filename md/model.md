# KIẾN TRÚC LUỒNG MA TRẬN UAVReIDNet

Tài liệu này phân tích chi tiết biến đổi ma trận ở từng module của mạng nhận diện UAV (UAVReIDNet), bao gồm 3 trục chính: Hình ảnh (GASNet), Thời gian (Mamba), và Định danh (ReID Head).

**Khởi đầu:** Dữ liệu đầu vào là một batch video clip. 
- Giả sử Batch Size $B = 1$, Số lượng khung hình $N = 16$.
- Kích thước ảnh chuẩn: $224 \times 224$.
- Kích thước Tensor gốc: `[1, 16, 3, 224, 224]`.
- **Thao tác Fold:** Trước khi vào GASNet, hệ thống ép 2 trục $B$ và $N$ làm một để xử lý tĩnh. Tensor chuyển thành: **`[16, 3, 224, 224]`**.

---

## 1. TRỤC XƯƠNG SỐNG HÌNH ẢNH: GASNET (2D CNN)
Nhiệm vụ: Trích xuất đặc trưng không gian của UAV trên từng frame ảnh rời rạc.

### 1.1. Stem (Trạm tiếp đón)
*   **Phép toán:** Tích chập 7x7 (stride 2) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool 3x3 (stride 2).
*   **Biến đổi Ma trận:** `[16, 3, 224, 224]` $\rightarrow$ **`[16, 64, 56, 56]`**.
*   **Chi tiết:** Đóng vai trò như một "cái phễu" thu nhỏ khung hình siêu tốc. Giúp giảm tải đột ngột lượng tính toán (FLOPs) trước khi đưa vào các lớp mạng sâu, đồng thời trích xuất các đặc trưng viền/góc cơ bản.

### 1.2. ResNet50-IBN Backbone (Các lớp sâu)
*   **Layer 1 (Có IBN):** Dữ liệu chẻ làm đôi. 32 kênh đi qua Instance Norm (lọc bỏ nhiễu môi trường như độ sáng, màu mây), 32 kênh đi qua Batch Norm (giữ lại cấu trúc hình học của UAV).
    *   Ma trận: `[16, 64, 56, 56]` $\rightarrow$ **`[16, 256, 56, 56]`**.
*   **Layer 2 (Có IBN):** Bước nhảy (Stride = 2) làm giảm nửa kích thước không gian.
    *   Ma trận: `[16, 256, 56, 56]` $\rightarrow$ **`[16, 512, 28, 28]`**.
*   **Layer 3 (Có IBN):** Tiếp tục giảm nửa kích thước không gian.
    *   Ma trận: `[16, 512, 28, 28]` $\rightarrow$ **`[16, 1024, 14, 14]`**.
*   **Layer 4 (KHÔNG IBN):** 
    *   Ma trận: `[16, 1024, 14, 14]` $\rightarrow$ **`[16, 2048, 7, 7]`**.
    *   *Chi tiết:* Ở độ sâu này, ảnh chỉ còn $7 \times 7$. Mạng cố tình tắt IBN (chỉ dùng BN thuần túy) vì lúc này dữ liệu mang 100% ngữ nghĩa (Semantic) định danh ID. Nếu dùng Instance Norm ở đây sẽ vô tình "rửa trôi" mất đặc điểm ID cốt lõi của UAV.

### 1.3. Khối RGA (Relation-Aware Global Attention)
Được chèn vào sau mỗi Layer để tự động nhóm các điểm ảnh có liên kết cấu trúc lại với nhau. Dưới đây là kiến trúc của RGA Spatial (Áp dụng tại Layer 4, Ma trận vào `[16, 2048, 7, 7]`):

```mermaid
flowchart TD
    classDef tensor fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef op fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef split fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef block fill:#c8e6c9,stroke:#388e3c,stroke-width:1px,color:#000

    X("Input X<br>[16, 2048, 7, 7]"):::tensor
    N["N = 49 (Số node)"]:::op
    
    X --> Theta["Theta (1x1)"]:::op & Phi["Phi (1x1)"]:::op
    Theta --> T_Out("[16, 49, inter_C]"):::tensor
    Phi --> P_Out("[16, inter_C, 49]"):::tensor
    
    T_Out & P_Out --> BMM["Nhân Ma Trận (BMM)"]:::op
    BMM --> R_Mat("Relation Matrix<br>[16, 49, 49]"):::tensor
    
    R_Mat --> Rel_Info["Gộp Hàng & Cột"]:::op --> Rel_Vec("[16, 98, 49]"):::tensor
    Rel_Vec --> Conv1["Conv1d"]:::op --> Rel_Comp("[16, rel_feat, 49]"):::tensor
    
    X --> G_Branch["G (Global Pool)"]:::op --> G_Out("[16, 1, 49]"):::tensor
    
    Rel_Comp & G_Out --> Concat["Concat"]:::split --> Attn["Conv1d + Sigmoid"]:::op
    Attn --> Mask("Spatial Mask<br>[16, 1, 7, 7]"):::tensor
    
    X & Mask --> Mul["Multiply"]:::op --> Out("Output<br>[16, 2048, 7, 7]"):::tensor
```
*(Cơ chế RGAChannel lặp lại tương tự nhưng dùng ma trận quan hệ `[2048, 2048]`).*

**Giải phẫu chi tiết cơ chế RGA:**
*   **Vấn đề:** Các cơ chế Attention cũ thường chỉ đoán vùng quan trọng dựa trên màu sắc cục bộ. Đối với UAV bay trên bầu trời, nhiễu (mây) rất dễ đánh lừa mô hình.
*   **Cách hoạt động:** RGA sinh ra Ma trận quan hệ $N \times N$ (ở đây là $49 \times 49$). Nó tính toán mức độ giống nhau của từng điểm ảnh với 48 điểm ảnh còn lại.
*   **Hiệu quả:** Mô hình học được "Sự liên kết bầy đàn". Dù UAV có mờ, nhưng mô hình phát hiện ra cấu trúc 4 cánh quạt có mối quan hệ hình học mật thiết với nhau, nó sẽ cấp hệ số Mask sáng rực cho toàn bộ 4 cánh quạt đó, đồng thời dập tắt hoàn toàn các điểm ảnh của bầu trời do chúng không có liên kết cấu trúc nào.

### 1.4. Nhánh Đa tỷ lệ Full Scale (OSBlockFS)
Rẽ nhánh từ Layer 3 để dò tìm UAV ở nhiều mức độ bao quát khác nhau (từ UAV bay sát camera đến UAV lẩn khuất ở xa).

```mermaid
flowchart TD
    classDef tensor fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef op fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef split fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef block fill:#c8e6c9,stroke:#388e3c,stroke-width:1px,color:#000

    X("Layer 3 Output<br>[16, 1024, 14, 14]"):::tensor --> Split["Chia 4 Streams"]:::split
    
    subgraph Omni_Scale_Streams
        S1["Luồng 1 (1x Lite3x3) - Tầm gần"]:::block
        S2["Luồng 2 (2x Lite3x3) - Tầm vừa"]:::block
        S3["Luồng 3 (3x Lite3x3) - Tầm xa"]:::block
        S4["Luồng 4 (4x Lite3x3) - Siêu xa"]:::block
    end
    
    Split --> S1 & S2 & S3 & S4
    
    Gate["Channel Gate<br>Sinh ra 4 Masks [16, 512, 1, 1] riêng biệt"]:::op
    S1 & S2 & S3 & S4 --> Gate
    
    Gate -.-> Mul1["* L1"]:::op & Mul2["* L2"]:::op & Mul3["* L3"]:::op & Mul4["* L4"]:::op
    S1 --> Mul1 
    S2 --> Mul2 
    S3 --> Mul3 
    S4 --> Mul4
    
    Mul1 & Mul2 & Mul3 & Mul4 --> Add["Cộng gộp 4 luồng"]:::split
    X --> Shortcut["Shortcut (Conv 1x1)"]:::op
    
    Add & Shortcut --> Final["Add + ReLU"]:::op --> Out("FS Output<br>[16, 512, 14, 14]"):::tensor
```

**Giải phẫu chi tiết cơ chế Full Scale:**
*   **Khối Lite3x3 (Công nghệ Depthwise):** Để mô phỏng tầm nhìn xa $5 \times 5, 7 \times 7$, mạng xếp chồng nhiều khối $3 \times 3$ lên nhau. Để tránh nặng máy, Lite3x3 trượt bộ lọc $3 \times 3$ **độc lập trên từng kênh riêng biệt** (không trộn kênh), giúp giảm tới 90% số lượng phép tính (FLOPs) so với tích chập truyền thống.
*   **Cổng kiểm soát (Channel Gate):** Hoạt động như một bàn trộn âm thanh (Mixer). Nó được "Share" (dùng chung 1 bộ trọng số) cho cả 4 luồng. Khi nhận ảnh UAV nhỏ xíu, Gate sẽ tự động sinh ra một tấm Mask (Mặt nạ) toàn số 0 cho Luồng 1 (tắt luồng cận cảnh), và Mask toàn số 1 cho Luồng 4 (bật tối đa luồng siêu xa). Do đó, nhánh này đóng vai trò như một bộ **Auto-focus tự động bắt nét mọi kích cỡ**.

### 1.5. GeM Pooling (Generalized Mean Pooling)
*   **Cơ chế:** Kỹ thuật ép phẳng ảnh từ 2D về 1D. Nó áp dụng công thức Lũy thừa $p=3.0$ trước khi lấy trung bình, sau đó khai căn bậc $3$.
*   **Tác dụng:** Việc mũ $3$ giúp khuếch đại cực mạnh khoảng cách giữa điểm sáng (UAV) và điểm mờ (Nền trời). Nó là dạng lai hoàn hảo giữa Average Pool (giữ bối cảnh) và Max Pool (giữ đặc trưng nổi bật).
*   **Kết quả Ma trận:**
    *   Nhánh Global: `[16, 2048, 7, 7]` $\rightarrow$ GeM $\rightarrow$ Vector `[16, 2048]`.
    *   Nhánh FS: `[16, 512, 14, 14]` $\rightarrow$ GeM $\rightarrow$ Vector `[16, 512]`.
    *   Ghép nối (Concat): **`[16, 2560]`**.
    *   **Thao tác Unfold:** Mở bung ma trận trở lại định dạng Video $\rightarrow$ **`[1, 16, 2560]`**.

---

## 2. TRỤC NÃO BỘ CHUYỂN ĐỘNG: TEMPORAL MAMBA
Nhiệm vụ: Tìm kiếm quỹ đạo bay và mô hình chuyển động theo trục thời gian bằng công nghệ State Space Model (SSM).

### 2.1. Cấu trúc Bi-Mamba (Tầm nhìn 2 chiều)

```mermaid
flowchart TD
    classDef tensor fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef op fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef split fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef block fill:#c8e6c9,stroke:#388e3c,stroke-width:1px,color:#000

    In("Input Mamba Layer<br>[1, 16, 512]"):::tensor
    Split["Tách 2 chiều"]:::split
    Fwd["Chiều Xuôi<br>Lõi S6"]:::block
    Flip1["Lật Video"]:::op
    Bwd["Chiều Ngược<br>Lõi S6"]:::block
    Flip2["Lật Lại"]:::op
    Add["Element-wise Add"]:::split
    Norm["LayerNorm"]:::op
    Out("Output Mamba Layer<br>[1, 16, 512]"):::tensor

    In --> Split
    Split --> Fwd
    Split --> Flip1
    Flip1 --> Bwd
    Bwd --> Flip2
    
    Fwd --> Add
    Flip2 --> Add
    In --> Add
    
    Add --> Norm
    Norm --> Out
```

**Giải phẫu chi tiết cơ chế Bi-Mamba:**
*   **Linear Projection & Positional Embedding:** Nén vector khổng lồ `2560` xuống `512` chiều để giảm tải RAM, đồng thời cộng thêm ma trận số thứ tự (tọa độ thời gian) để Mamba phân biệt được trật tự trước/sau của các frame ảnh.
*   **Phân tích Kép (Bidirectional):** Khi UAV bị che khuất ngang chừng (Occlusion), việc chỉ nhìn từ Quá khứ (Chiều xuôi) sẽ khiến mạng mất dấu. Bằng cách lấy thêm dữ liệu tua ngược từ Tương lai (Chiều ngược) và cộng gộp lại, mạng có khả năng **nội suy và vá lỗi quỹ đạo** cực kỳ mạnh mẽ.

### 2.2. Giải phẫu chi tiết Cấp độ Ma trận của khối SimpleS6Block (Selective Scan)
Đây là lõi toán học thay thế hoàn hảo cho Attention, giúp Mamba xử lý tuyến tính mà vẫn "chọn lọc" được ngữ cảnh. Giả định đầu vào `x` có kích thước **`[B=1, L=16, d_model=512]`** (1 video, 16 frames, 512 chiều), hệ số mở rộng `expand=2`, bộ nhớ `d_state=16`.
=> `d_inner = expand * d_model = 1024`.

```mermaid
flowchart TD
    classDef tensor fill:#e1bee7,stroke:#8e24aa,stroke-width:1px,color:#000
    classDef op fill:#bbdefb,stroke:#1976d2,stroke-width:1px,color:#000
    classDef split fill:#ffcc80,stroke:#f57c00,stroke-width:1px,color:#000
    classDef param fill:#fff9c4,stroke:#fbc02d,stroke-width:1px,color:#000

    X("Input x<br>[1, 16, 512]"):::tensor
    InProj["Linear in_proj"]:::op
    XZ("XZ<br>[1, 16, 2048]"):::tensor
    Chunk["Chunk (Chia đôi)"]:::split
    
    X_Proj("x_proj<br>[1, 16, 1024]"):::tensor
    Z_Gate("z_gate<br>[1, 16, 1024]"):::tensor
    
    Conv1D["Conv1d (Depthwise)<br>SiLU"]:::op
    X_Conv("x_conv (Feature)<br>[1, 16, 1024]"):::tensor
    
    X_Proj_Param["Linear x_proj"]:::op
    XDbl("x_dbl<br>[1, 16, 33]"):::tensor
    SplitParam["Tách tham số SSM"]:::split
    
    DT_raw("dt_raw<br>[1, 16, 1]"):::tensor
    B_mat("B_mat<br>[1, 16, 16]"):::tensor
    C_mat("C_mat<br>[1, 16, 16]"):::tensor
    
    DTProj["Linear dt_proj<br>Softplus"]:::op
    DT("dt (Thời gian bù)<br>[1, 16, 1024]"):::tensor
    
    A_Log("A_log Parameter<br>[1024, 16]"):::param
    A_Exp["-exp(A)"]:::op
    A_mat("A_mat<br>[1024, 16]"):::tensor

    X --> InProj --> XZ --> Chunk
    Chunk --> X_Proj
    Chunk --> Z_Gate
    
    X_Proj --> Conv1D --> X_Conv
    X_Conv --> X_Proj_Param --> XDbl --> SplitParam
    
    SplitParam --> DT_raw
    SplitParam --> B_mat
    SplitParam --> C_mat
    
    DT_raw --> DTProj --> DT
    A_Log --> A_Exp --> A_mat
    
    subgraph Parallel_Selective_Scan ["Thuật toán Parallel Scan"]
        direction TB
        MulW["W = dt * A"]:::op
        W("W<br>[1, 16, 1024, 16]"):::tensor
        MulV["V = (dt * B) * x_conv"]:::op
        V("V<br>[1, 16, 1024, 16]"):::tensor
        
        Cumsum["Cumsum theo L<br>Tránh NaN bằng P_i - P_j"]:::op
        P("Ma trận P<br>[1, 1024, 16, 16]"):::tensor
        AttnForm["Nhân Trọng số (Attention-like)"]:::op
        H("Trạng thái h<br>[1, 16, 1024, 16]"):::tensor

        DT --> MulW
        A_mat --> MulW
        MulW --> W
        
        DT --> MulV
        B_mat --> MulV
        X_Conv --> MulV
        MulV --> V
        
        W --> Cumsum --> P
        P --> AttnForm
        V --> AttnForm
        AttnForm --> H
    end
    
    Y_sum["Output: sum(h * C)"]:::op
    Y_out("y<br>[1, 16, 1024]"):::tensor
    SkipD["Cộng Skip Connection (D)"]:::op
    Y_final("y_final<br>[1, 16, 1024]"):::tensor
    Gate["Gating: y * SiLU(z_gate)"]:::op
    Y_gated("y_gated<br>[1, 16, 1024]"):::tensor
    OutProj["Linear out_proj"]:::op
    FinalOut("Output S6 Block<br>[1, 16, 512]"):::tensor

    H --> Y_sum
    C_mat --> Y_sum
    Y_sum --> Y_out
    
    Y_out --> SkipD
    X_Conv --> SkipD
    SkipD --> Y_final
    
    Y_final --> Gate
    Z_Gate --> Gate
    Gate --> Y_gated
    Y_gated --> OutProj --> FinalOut
```

**Biến đổi Ma trận chi tiết trong SimpleS6Block:**

*   **Bước 1: Mở rộng và Khởi tạo cục bộ (Linear & Local Conv)**
    *   Ma trận đầu vào `[1, 16, 512]` được nhân ma trận (Linear) bung ra thành `[1, 16, 2048]`, sau đó cắt làm 2 nửa độc lập: Nhánh đặc trưng `x_proj` `[1, 16, 1024]` và Nhánh cổng kiểm soát `z_gate` `[1, 16, 1024]`.
    *   `x_proj` đi qua tích chập 1D (Depthwise) để thu thập thông tin cục bộ giữa các frames lân cận, tạo ra `x_conv` **`[1, 16, 1024]`**.
*   **Bước 2: Sinh tham số động "Selective" (Tính chọn lọc)**
    *   Sự khác biệt lớn nhất giữa Mamba và SSM cũ là tham số không cố định. `x_conv` đi qua Linear để đẻ ra một ma trận siêu nhỏ `x_dbl` `[1, 16, 33]`.
    *   `33` chiều này được cắt nhỏ thành: `dt` (1 chiều), `B` (16 chiều, tức `d_state`), `C` (16 chiều). Do đó, $B$ và $C$ biến đổi linh hoạt theo từng Frame.
    *   Hệ số `dt` (bước nhảy thời gian) sau đó được giãn nở trở lại bằng Linear để phủ kín 1024 kênh $\rightarrow$ `dt` **`[1, 16, 1024]`**.
*   **Bước 3: Parallel Selective Scan (Attention-like Formulation)**
    *   Đây là phép thuật toán học của Mamba. Thay vì vòng lặp tuần tự $h_t = A * h_{t-1} + B * x_t$ rất chậm, đoạn code tính toán song song:
    *   Biến rời rạc hóa: `W = dt * A` và trọng số `V = (dt * B) * x_conv`. Chúng tạo ra các ma trận 4D khổng lồ lưu trữ toàn bộ trạng thái hệ thống: **`[1, 16, 1024, 16]`** (Batch, L, d_inner, d_state).
    *   Thuật toán tính toán ma trận tổng tích lũy (Cumsum) $P$ dọc theo chiều thời gian (L), sau đó lấy độ chênh lệch $(P_i - P_j)$. Trick toán học này để mô phỏng sự tích lũy của chuỗi thời gian mà không cần chạy vòng lặp, đồng thời tránh lỗi toán học NaN cực tốt.
    *   Áp dụng Causal Mask (Mặt nạ tam giác dưới) để đảm bảo Tương lai không ảnh hưởng đến Quá khứ.
    *   Hội tụ lại thu được bộ nhớ ẩn $h$ **`[1, 16, 1024, 16]`**.
*   **Bước 4: Hội tụ Đầu ra & Gating**
    *   Trạng thái $h$ nhân với hệ số quyết định $C$ rồi tính tổng (sum) dọc theo chiều `d_state` để nén lại thành `y` **`[1, 16, 1024]`**.
    *   Cơ chế Gating: `y` được nhân phần tử (element-wise) với `SiLU(z_gate)`. Nhánh Gating đóng vai trò như một bộ lọc nhiễu, dập tắt các tín hiệu nhiễu từ bối cảnh và giữ lại luồng thông tin chứa quỹ đạo UAV.
    *   Cuối cùng, chiếu tuyến tính (Linear out_proj) đưa ma trận về hình dáng ban đầu: **`[1, 16, 512]`**.

*   **Đầu ra Cấp cao (Về lại TemporalMambaEncoder):** Ma trận `[1, 16, 512]` của Mamba Layer sau đó đi qua Temporal Mean Pooling (Lấy trung bình theo 16 frames) và mạng MLP Head để hội tụ về dạng token định danh thuần túy **`[1, 512]`**.

---

## 3. TRẠM QUYẾT ĐỊNH: REID HEAD
Nhiệm vụ: Cân bằng hai thái cực Hình học (Hình cầu vs Mặt phẳng) và đưa ra phán quyết cuối cùng.

*   **1. Concatenation:** Nối Đặc trưng tĩnh `[1, 2560]` và Đặc trưng chuyển động `[1, 512]`. Trở thành Vector Đặc Trưng Hoàn Hảo (feat): **`[1, 3072]`**.

*   **2. Nút thắt cổ chai BNNeck:**
    *   *Mâu thuẫn ReID:* Trong huấn luyện, Triplet Loss ép các vector có chung ID cuộn lại thành dạng **Hình cầu (Sphere)**. Trong khi đó, Cross-Entropy (ID Loss) lại cố đẩy các vector văng ra xa để dễ dàng chặt bằng **Mặt phẳng tuyến tính (Hyperplane)**. Hai hàm Loss này "đánh nhau" nếu dùng chung 1 vector.
    *   *Giải pháp:* Chèn một lớp `BatchNorm1d` ở giữa. Vector gốc đưa cho Triplet Loss. Vector đi qua BNNeck (`bn_feat`) đưa cho ID Loss.
    *   *Bí mật toán học:* Tham số Bias $\beta$ bị ép khóa cứng ở mức $0$ (`requires_grad=False`). Điều này ép mọi vector phải có trung bình = 0 (luôn chụm vào Gốc tọa độ $(0,0,0)$). Đây là điều kiện tối quyết để thuật toán Cosine Similarity chạy chính xác 100% lúc Inference.
    *   *Đầu ra:* **`bn_feat` `[1, 3072]`**.

*   **3. Routing (Chế độ Phân luồng):**
    *   **Train Mode:** Đưa `bn_feat` qua lớp Linear. Đầu ra: Ma trận Logits **`[1, 1000]`** (Xác suất của 1000 ID).
    *   **Eval Mode:** Vứt bỏ bộ Linear. Trả về đúng vector đặc trưng gốc `bn_feat` **`[1, 3072]`** để so sánh khoảng cách thực tế giữa các Drone ngoài tự nhiên.
