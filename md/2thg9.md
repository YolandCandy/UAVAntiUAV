# Báo cáo sửa lỗi — 2/9/2026

## Tổng quan

Dự án: UAVAntiUAV — GASNet backbone **resnet50 → dinov3_convnext**  
Mục tiêu: fix lỗi logic khiến kết quả reid tệ, hỗ trợ huấn luyện + inference với backbone mới.  
Trạng thái: code đã sửa xong, cần đồng bộ lên Colab và chạy lại.

---

## 1. Fix load checkpoint (model.py) — Lỗi 1+2

### Vấn đề
- VRU checkpoint được train với `torch.compile` → state_dict có prefix `_orig_mod.` (ví dụ `_orig_mod.convnext_backbone.stages.0...`)
- `model.load_state_dict(state_dict, strict=False)` không strip prefix → **tất cả key đều không khớp** → backbone bị skip → giữ nguyên random init → kết quả tệ
- Khi backbone là convnext (dim 960) vs resnet (dim 2560), classifier shape khác nhau giữa VRU (N_VRU_classes) và UAV (N_UAV_identities) → `strict=False` vẫn báo lỗi RuntimeError

### Fix (model.py dòng 260-290)
```python
# Strip _orig_mod. prefix
new_k = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
# Filter shape mismatch
if new_k in model_state and v.shape != model_state[new_k].shape:
    continue
```
- Strip `_orig_mod.` prefix
- Filter shape-mismatched keys (classifier VRU vs UAV)
- In cảnh báo đỏ nếu convnext_backbone không load được
- `return_raw_features_eval = True` (giữ nguyên)

---

## 2. Fix ConvNeXt backbone dimension (model.py)

### Vấn đề
ResNet: c1-4 = [256, 512, 1024, 2048], visual_dim = 2560, head in_dim = 3072  
ConvNeXt: c1-4 = [96, 192, 384, 768], visual_dim = 960, head in_dim = 1472

Các hằng số phải match architecture.

### Fix
- `feat_dim = 1472 if backbone == "dinov3_convnext" else 3072` (train_reid.py)
- `visual_dim = 960`, `ReIDHead in_dim = 1472` (model.py đã xử lý tự động)

---

## 3. Fix config gasnet_weights path (config_colab.yaml) — Lỗi 3

### Vấn đề
Path cũ: `/content/gasnet.best.pth`  
Thực tế: `/content/drive/MyDrive/output/gasnet_convnext_v0.best.pth`

### Fix
```yaml
paths:
  gasnet_weights: /content/drive/MyDrive/output/gasnet_convnext_v0.best.pth
```

---

## 4. Kết nối TemporalConsistencyLoss (lam2) — Lỗi 4

### Vấn đề
`criterion_temporal` được tạo nhưng không dùng trong loss. Loss chỉ gồm:
```python
loss = loss_id + lam1 * loss_tri  # + lam3 * loss_center
```
Không có `lam2 * loss_temp`.

### Fix
- Model.py: `TemporalMambaEncoder.forward` trả về `(x, temporal_seq)`
- `extract_features` trả về `(visual_feat, temporal_token, temporal_seq)`
- `forward` train mode trả về `(feat_b, bn_feat_b, logits_b), (feat_a, ...), (t_seq_before, t_seq_after)`
- Infer/eval/robustness/realworld: unpack `temporal_token, _ = model.temporal_encoder(...)` (2 giá trị)
- train_reid.py: `loss_temp = criterion_temporal(t_seq_b) + criterion_temporal(t_seq_a)`
- `loss = loss_id + lam1*loss_tri + lam2*loss_temp + lam3*loss_center`

---

## 5. Fix stage2 param grouping (train_reid.py) — Lỗi 5

### Vấn đề
User đã train VRU → GASNet (convnext + ga1-4 + fs1-2 + bnneck) có weights. Cần để trong nhóm pretrained (lr 1e-5), không phải random (lr 1e-4).

### Fix
```python
if "convnext_backbone" in name or "swin_backbone" in name or "base" in name or "ga" in name or "fs" in name:
    pretrained_params.append(param)  # lr 1e-5
else:
    random_params.append(param)      # lr 1e-4
```

---

## 6. Fix validate fallback (train_reid.py + evaluate_reid.py) — Lỗi 6+8

### Vấn đề
Khi không có test dir, code fallback về train dir → metric ảo cao.

### Fix
```python
if not os.path.exists(test_dir):
    raise FileNotFoundError(
        "Không tìm thấy test dir. Chạy data_pipeline trước."
    )
```

---

## 7. Fix Center Loss (train_reid.py)

### Vấn đề
- `self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))` → init random → loss lớn, hội tụ chậm
- `loss_center = criterion_center(bn_b, pids)` dùng bn_b **đã normalize** (sai theo paper — center loss nên dùng feature thô)
- `lam3=0.05` quá lớn → center loss dominate gradient

### Fix
- `self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))`
- `loss_center = criterion_center(bn_b, pids)` — bn_b **thô** (chưa normalize), đúng paper
- `lam3: 0.001`

---

## 8. Fix evaluate_reid online eval (evaluate_reid.py) — Lỗi 7

### Vấn đề
Block "Online Sequential Evaluation" dùng `np.random.randint(1, 5)` để tạo latency giả + threshold cố định 0.7 → không có giá trị, gây nhiễu.

### Fix
Xóa toàn bộ block (dòng 392-395 cũ), thay bằng comment.

---

## 9. Fix infer.py absent.txt (infer.py) — Lỗi 9

### Vấn đề
```python
is_absent = (absent[frame_idx] == 1) if frame_idx < len(absent) else True
```
Khi absent.txt ngắn hơn số frame, frame ngoài mảng được coi là **absent = True** → UAV bị coi là mất vĩnh viễn → không bao giờ re-acquire.

### Fix
```python
is_absent = (absent[frame_idx] == 1) if frame_idx < len(absent) else False
```
Kèm warning nếu absent.txt bị cắt ngắn.

---

### Fix
Giữ bản đúng: `temporal_token, _ = model.temporal_encoder(seq_feats)`

---

## 11. Fix NaN Stage 2 (train_reid.py) — phát hiện mới từ log

### Vấn đề
Training log: Stage 2 epoch 1-5 OK (Rank-1 71.71%), epoch 6→30 **loss = NaN** → validation 0.57%.

Nguyên nhân:
1. `GradScaler` scale factor tích lũy qua 30 epoch Stage 1 (lớn gấp nhiều lần) → khi backbone unfreeze, gradient scaled overflow → inf → NaN weights vĩnh viễn
2. `torch.compile` graph cũ chứa backbone frozen; khi unfreeze, graph recompile + scaler interaction → `found_inf` detection sai → inf gradient lọt qua optimizer step

### Fix
```python
# Tạo scaler MỚI ở Stage 2 (reset scale factor)
scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
# Reset dynamo để compile graph mới
if args.use_compile and hasattr(torch, '_dynamo'):
    torch._dynamo.reset()
# Skip batch nếu loss không finite
if not torch.isfinite(loss):
    continue
```

---

## 12. Fix infer.py weighted mean vs plain mean (infer.py) — Lỗi A

### Vấn đề
Training dùng **plain mean** (`feats.mean(dim=1)`), inference dùng **weighted mean** theo sharpness (`get_weighted_visual_mean()`) → head nhận input khác training → fused feature lệch → fine score thấp.

```
Training: visual_feat = feats.mean(dim=1)                    # plain mean
Infer:    visual_mean = sliding_window.get_weighted_visual_mean()  # weighted!
```

### Fix
```python
visual_mean = sliding_window.get_weighted_visual_mean()   # cho coarse score (không qua head)
visual_plain = seq_feats.mean(dim=1)                     # giống training → cho head
fused_feat = compute_reid_embedding(model, seq_feats, visual_plain)
```

---

## 13. Debug: tách visual, temporal, fused (infer.py) — Giải pháp 3

### Thêm
- `compute_fused_vector` trả về `(visual_mean, temporal_token, fused_feat)`
- Memory bank lưu thêm `temporal` entry
- Thêm hàm `temporal_score()` để so sánh riêng temporal token
- Debug print trong T2_SEARCH:
```
[108] DEBUG sim: visual_mean=0.983 | temporal=0.581 | fused(fine)=0.150
```

### Kết quả debug
- `visual_mean` = 0.98-0.99 ✅ rất ổn định
- `temporal` = 0.47-0.74 🟡 dao động
- `fused(fine)` = 0.04-0.51 ❌ thấp hơn cả 2 thành phần

**Nguyên nhân gốc rễ**: BN1d trong head khuếch đại khác biệt nhỏ ở channel variance thấp. Convnext visual chỉ chiếm 65% fused (vs 83% resnet) → temporal ảnh hưởng lớn hơn.

---

## 14. Fix config model_path (config_colab.yaml) — Lỗi C

### Vấn đề
```yaml
infer.model_path: /content/best_model.pth          # checkpoint cũ
eval.model_path: /content/best_model.pth
```
Checkpoint thực tế nằm ở `.../checkpoints/v3.8/best_model.pth`.

### Fix
```yaml
infer.model_path: /content/drive/MyDrive/UAV_Anti_UAV/checkpoints/v3.8/best_model.pth
```

---

## Files đã sửa

| File | Số dòng | Các fix |
|---|---|---|
| `model.py` | 394 | 1, 2, 4, 7 |
| `train_reid.py` | 657 | 4, 5, 6, 7, 11 |
| `infer.py` | 615 | 4, 9, 10, 12, 13 |
| `evaluate_reid.py` | 397 | 6, 8 |
| `evaluate_reid_robustness.py` | 584 | 4 |
| `phan_rang/infer_realworld.py` | 574 | 4 |
| `configs/config_colab.yaml` | 110 | 3, 14 |

## Các vấn đề chưa giải quyết

- **Mamba fallback**: đang dùng `SimpleS6Block` (fallback) vì không có `mamba_ssm` → temporal token có thể kém ổn định hơn
- **requirements.txt**: thiếu `transformers>=4.56`, `huggingface_hub`, `packaging`
- **reid_threshold 0.75 quá cao**: fine score cao nhất chỉ ~0.5 (cùng identity), cần giảm xuống 0.45-0.5 hoặc dùng coarse re-acquire
- **num_frames**: training dùng 8, infer dùng 8 — nếu lệch sẽ ảnh hưởng temporal token
- **Temporal token re-acquire**: temporal token của cùng identity qua temporal window khác nhau không ổn định → nên dùng coarse (visual-only) để re-acquire, temporal/fused chỉ cho anti-hijack