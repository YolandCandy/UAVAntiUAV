# Cập nhật — 9/9/2026

## Bổ sung chỉ số TAR@FAR 0.1% vào script đánh giá

### Vấn đề
Mặc dù sử dụng backbone `convnext` cho kết quả Rank-1 và mAP cao hơn so với `resnet50`, nhưng có khả năng điểm ReID thực tế lại thấp khi triển khai. Để đánh giá mức độ tin cậy của tính năng nhận diện, ta cần một chỉ số khắt khe hơn: **TAR@FAR=0.1%** (True Accept Rate tại False Accept Rate 0.1%).

### Giải pháp (trong `evaluate_reid.py`)

1. **Thêm hàm `compute_tar_at_far()`**:
   - Tính toán ma trận cosine similarity giữa toàn bộ features của query và gallery.
   - Trích xuất **genuine scores** (cùng ID) và **impostor scores** (khác ID), bỏ qua self-match.
   - Tìm ngưỡng (threshold) từ phân phối impostor scores sao cho tỷ lệ False Accept Rate (FAR) ≤ 0.1%.
   - Áp dụng ngưỡng này lên genuine scores để tính True Accept Rate (TAR).

2. **Tích hợp vào quá trình in kết quả**:
   - In thêm dòng `TAR@FAR=0.1%` cùng với `actual FAR` ra màn hình.
   - Lưu 2 trường `TAR@FAR=0.1%` và `actual_FAR` vào file `evaluation_report.json` để dễ dàng log và so sánh tự động giữa các thí nghiệm.

### Nhận xét
Chỉ số này giúp đánh giá khả năng mô hình từ chối đúng các ID sai (impostors) mà không hy sinh quá nhiều độ chính xác nhận diện ID đúng (genuine) - điều rất quan trọng trong bài toán UAV tracking ngoài thực tế, nơi thường xuyên xuất hiện nhiễu hoặc đối tượng lạ.

---

## Threshold Calibration — Fixed FAR

### Vấn đề với in-sample threshold

Hàm `compute_tar_at_far()` cũ tính threshold **trực tiếp từ test set** (in-sample) → thiên lệch cao vì tìm và đánh giá trên cùng 1 tập. Trong triển khai thực tế, threshold phải được chọn trước khi thấy test data.

### Giải pháp: Tách cal-split từ test set (sequence-level)

Không cần train lại. Tách test set theo **sequence-level** (không trộn sequence giữa cal/eval → không leak identity):

| Split | Tỉ lệ | Mục đích |
|---|---|---|
| **cal-split** | 40% sequences | Tìm threshold t* sao cho FAR ≤ 0.1% |
| **eval-split** | 60% sequences | Áp t* → báo cáo TAR/FAR không thiên lệch |

### Script mới: `calibrate_threshold.py`

Chạy lần đầu để tìm threshold:
```bash
python calibrate_threshold.py --config configs/config_local.yaml \
    --cal-ratio 0.4 --far-target 0.001 --output-dir calib_results
```
Output: `calib_results/calibrated_threshold.json` + 3 biểu đồ (score distribution, DET curve, ROC comparison).

Sau đó dùng threshold đã calibrate khi evaluate:
```bash
python evaluate_reid.py --config configs/config_local.yaml \
    --threshold-file calib_results/calibrated_threshold.json
```

### Thay đổi trong `evaluate_reid.py`

- Thêm flag `--threshold-file`
- Nếu có file: dùng calibrated threshold (unbiased), in `[source=calibrated]`
- Nếu không có: fallback in-sample + **cảnh báo rõ ràng**
- `evaluation_report.json` bổ sung `threshold` và `threshold_source`

### Tối ưu thời gian chạy và dung lượng đánh giá
- Bổ sung tuỳ chọn tắt hoàn toàn việc sinh ảnh trực quan (visualization) trong quá trình đánh giá. 
- **Cách thực hiện:** Trong `evaluate_reid.py`, nếu tham số `max_correct_vis` được gán bằng `0`, chương trình sẽ bỏ qua toàn bộ khối lệnh vẽ ảnh và lưu ảnh (`Skipping visualization...`).
- **Config:** Cập nhật `configs/config_colab.yaml` thêm `max_correct_vis: 0` vào block `eval`. Điều này giúp tiết kiệm đáng kể thời gian I/O và dung lượng lưu trữ trên Google Drive khi chạy đánh giá trên Colab.

### Phân tích kết quả Calibration
Dựa trên kết quả chạy thử nghiệm đầu tiên với DINOv3 ConvNeXt-Small:
- **Ngưỡng tìm được (t*):** `0.852396` (tại `FAR ≤ 0.1%` trên tập calibration).
- **Kết quả trên tập holdout eval:**
  - **Actual FAR:** `0.1096%` (rất sát với target `0.1%`, chứng tỏ calibration hoạt động chính xác và không bị thiên lệch).
  - **TAR:** `8.45%`.
- **Nhận xét:** Model có khả năng xếp hạng tốt (Rank-1 `74.56%`, Rank-5 `85.36%`) nhưng các điểm số (scores) chưa đủ tách biệt tuyệt đối (absolute margin) để dùng làm ngưỡng cứng khắt khe. Phần lớn các cặp đúng có similarity `< 0.85`. Do đó, trong thực tế triển khai (inference), việc sử dụng ngưỡng thấp hơn (như `0.75`) để tránh bỏ sót (tăng TAR) và chấp nhận một lượng FAR nhất định là phương án thỏa hiệp hợp lý.
