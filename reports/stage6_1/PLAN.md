# Stage 6.1 — kế hoạch kiểm chứng

Mục tiêu: kiểm chứng tính tái lập, tính thụ động của output và hai sự kiện an
toàn; không thay thuật toán, timing, data, model hoặc kết quả Stage 1–6.

Thực hiện trong workspace được giao, không commit/push, không train.
Python dùng `.venv/Scripts/python.exe`; baseline: 142 passed (12.80 s).
Python hệ thống: 139 passed, 3 lỗi import Matplotlib, không phải lỗi thuật toán.

## Giả thuyết và phép thử

- [x] Xác nhận Stage 6 từ raw XML/CSV, SHA code/input/binary và thông số.
- [x] Thêm `stage6_1_probe.py`: tái dùng `fixed_time` và `SumoEnv`, bọc
  `simulationStep` chỉ để đọc sau bước. Hash vị trí/tốc độ theo từng bước;
  chỉ giữ chi tiết quanh vùng nghi vấn. Lưu argv thực (cả remote port).
- [x] Chạy A (không tripinfo), B (tripinfo), C (tripinfo + CSV metrics),
  ít nhất ba lượt 700 s cùng seed 42. So sánh timestamp, đèn, departures,
  active IDs, trạng thái xe, queue, teleport/collision theo thứ tự.
- [x] Đối chứng observer: cùng output nhưng không đọc thêm; đảo thứ tự đọc
  nếu cần. Không coi tương quan output/sai khác là nhân quả.
- [x] Nếu có sai khác: xuất chi tiết bước đầu và điều tra upstream bằng
  tài liệu/source SUMO; không đổi cấu hình để che sai khác.
- [x] Replay FT Ingolstadt1 seed2026 đến sau 58134; Cologne1 seed2028 đến
  sau 26823. Đọc route, connection, priority/request và xe lân cận.
- [x] Bổ sung test còn thiếu về metric/probe; chỉ sửa production nếu có
  test tái hiện lỗi. Đối chiếu kết quả mới với raw XML/CSV.
- [x] Chạy lại suite, xác nhận SHA Stage 1–6/data/model không đổi; viết
  README và verification.json, công bố rõ mọi giới hạn chưa kết luận.

## Phạm vi file

Tạo `stage6_1_probe.py`, `test_stage6_1.py`; chỉ bổ sung `test_evaluation.py`
nếu thiếu trường hợp metric. Kết quả mới đều trong `reports/stage6_1/`.
Không dự kiến sửa `sumo_env.py`, `evaluate.py`, FT/MP/QL.

## Nghiệm thu

Có ít nhất 3 lượt ngắn, lệnh và SHA đủ tái chạy; chỉ rõ bước khác đầu
hoặc giới hạn không tìm được; kiểm tra A/B/C và observer; báo cáo hai sự
kiện; metric được đối chiếu raw; test cũ giữ nguyên; dữ liệu/kết quả cũ
bất biến. Không tuyên bố đã sửa nguyên nhân chưa chứng minh.
