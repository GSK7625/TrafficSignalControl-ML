# Giai đoạn 6 — kế hoạch thực hiện

Mục tiêu: đánh giá FT/MP/QL cố định, cùng phép đo và đủ 30 lượt 3600 s.
Đặc tả là yêu cầu giai đoạn 6 trong cuộc hội thoại và AGENTS.md.
Thực hiện trực tiếp trong workspace hiện tại; không commit/push/train thêm.

- [x] Thêm test tính tay cho `trip_metrics`, `signal_metrics`, kiểm tra demand,
  cấu hình chung, model bất biến và báo cáo seed thiếu.
- [x] Thêm `evaluation_metrics.py` (XML/CSV → metrics), `evaluate.py`
  (`run_evaluation`, CLI một lượt hoặc toàn bộ seed). Tái dùng FT/MP generator
  và hàm load/encode/choose greedy QL; mọi lệnh qua `SumoEnv.step`.
  Chỉ thêm `tripinfo=False` vào SumoEnv, bật ba tùy chọn output khi yêu cầu.
- [x] Kiểm tra 99 test cũ và test mới; pilot Cologne1 QL seed 42,
  đối chiếu tín hiệu/queue/teleport với stage5: khớp cả 3600 dòng.
  Tripinfo ghi cả unfinished/undeparted; phân loại bằng depart/arrival/teleport ID.
- [x] Chạy 30 lượt: mỗi map × FT/MP/QL × 42,12345,2026,2027,2028;
  lưu nguyên output, không chọn lọc. SHA input/model/code trước và sau.
- [x] `plot_results.py`: bảng từng lượt, mean/std mẫu/min/max/n,
  chênh lệch từng cặp seed, 8 biểu đồ/map; README tiếng Việt và kiểm tra
  độc lập một số giá trị từ XML/CSV, rà soát cuối.
- [ ] Chứng minh output thụ động và replay khớp tuyệt đối trên mọi lượt:
  CHƯA ĐẠT. Probe không tripinfo cũng khác nhau ở cùng seed; chưa xác định
  nguyên nhân. Giữ nguyên 30 lượt đầu, công bố giới hạn trong README mục 9.

Kết quả kiểm tra: 142 passed; 30/30 partition xe và phép đo được đối chiếu.
Review đã sửa lỗi cho phép model khác giữa seed, có test RED → GREEN.
Không sửa FT/MP/QL, data hoặc hai model; không commit/push.

Định nghĩa chính: ba trung bình waiting/travel/timeLoss chỉ trên chuyến có
arrival hợp lệ, chưa từng teleport, không vaporized. Travel=arrival-depart.
Throughput chính=chuyến hợp lệ×3600/horizon; luôn báo thêm arrived SUMO và
arrived từng teleport. Queue=Σqueue mỗi giây/3600; không phải toàn vùng.
Mẫu số completion/insertion là số trip/vehicle yêu cầu trong [begin,end)
ở route XML; từ chối flow hoặc depart phi số chưa được hỗ trợ.

Điểm cần kiểm tra: teleport lặp cùng ID; arrival=-1/depart=-1;
horizon cắt chuyển tiếp; departure đúng end; tập seed thiếu; NaN/không có
chuyến hợp lệ phải để trống, không đặt 0; thống kê loaded/running khác phân
loại tripinfo phải được đối chiếu thay vì suy arrived=inserted-running.
