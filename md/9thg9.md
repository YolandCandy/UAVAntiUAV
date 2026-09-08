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
