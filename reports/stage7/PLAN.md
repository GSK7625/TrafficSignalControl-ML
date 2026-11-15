# Stage 7 — controlled training plan

Spec: yêu cầu Stage 7 trong chat ngày 15/11/2026; AGENTS.md áp dụng.
Thực thi trực tiếp trong checkout được giao, không commit/push, không DQN.

- [x] Đọc pipeline, các báo cáo Stage 5–6.2; Git ban đầu sạch tại c42e396.
- [x] Chụp SHA toàn bộ file tracked, data, model cũ; phiên bản Python/SUMO và binary.
- [x] Chạy toàn bộ pytest hiện có, gồm integration SUMO ngắn.
- [x] Pilot: 5 episode/map, SUMO seed 23423, exploration seed 7, từ bảng rỗng.
- [x] Replay từng update/RNG/checkpoint pilot; kiểm tra đủ 3600 s, action/epsilon/terminal.
- [x] Train chính: 2 map × exploration seeds 7/17 × 15 episode = 60 episode;
  mỗi lượt từ đầu, seed SUMO cố định 23423 mỗi episode, không resume/gộp.
- [x] Phân tích từng episode từ log có sẵn; giữ nguyên arrivals thô, không gọi là completion.
- [x] Validation: 42/12345 × (QL-3ep, QL-15ep seed7, QL-15ep seed17) × 2 map = 12 lượt.
- [x] Chọn tối đa một model cuối 15ep/map; ghi selection.json và SHA trước khi chạy test.
- [x] Test đã chốt: 31001/31002/31003; 2 lần/seed cho QL được chọn, QL-3ep, FT, MP
  = 48 lượt. Dùng evaluate.run_evaluation và evaluation_metrics nguyên bản; giữ mọi lượt.
- [x] Tổng hợp raw, mẫu số, biến thiên trong cùng seed, so Stage 6, báo cáo tiếng Việt.
- [x] Test bổ sung tối thiểu, pytest toàn bộ, SHA trước/sau; ghi mọi phần chưa kiểm chứng.

## Cấu hình cố định

Alpha=0.1, gamma=0.95 theo decision, bins 0/1–4/>=5, reward=-queue_seconds/elapsed,
Q khởi tạo 0, action IDs và tie-break first giữ nguyên. Epsilon tuyến tính 1→0.05
trong 5 episode pilot hoặc 15 episode chính. Green 30s, yellow/red/clearance nguyên bản.
Không sửa q_learning.py, SumoEnv, FT/MP, evaluate.py, evaluation_metrics.py hoặc data.
Chỉ cho analyze_ql.replay_training nhận thêm đường dẫn train tùy chọn nếu cần tái sử dụng;
default Stage 5 và output lịch sử giữ nguyên. stage7_summary.py xử lý offline,
test_stage7.py kiểm tra replay, coverage, chọn model và bất biến.

## Quy tắc chọn đã chốt trước training/test

Chỉ hai model CUỐI 15 episode, không tìm best reward/checkpoint. Xếp hạng theo
mean completion giảm dần (cùng requested/horizon nên tương đương throughput),
min movement coverage giảm dần, mean teleport tăng dần, max no-green gap tăng dần;
nếu vẫn hòa chọn exploration seed 7. QL-3ep chỉ là mốc, không phải ứng viên mới.
Validation là seed đã từng xem, không gọi là test mới. Không dùng test chọn/điều chỉnh.
Không loại kết quả có collision/teleport hoặc reward xấu, không chạy lại để chọn lượt đẹp.

## Mẫu số và kiểm tra

Coverage = cặp cập nhật / (state thực xuất hiện ở quyết định train × số action).
Q=0 không chứng minh chưa thử: dùng bộ đếm update replay, kể cả update reward=0.
Unseen/tie/untried báo cả numerator và số decision; distinct unseen cũng ghi riêng.
Movement = controlled links có G thực / tổng links; gap gồm hai biên horizon.
Waiting/travel/timeLoss chỉ trên valid_completed (arrived, không vaporized, chưa teleport).
Completion=valid/requested; throughput=valid×3600/horizon; queue=queue-seconds/3600.
So cặp lặp cùng seed và phạm vi biến thiên; không kết luận hội tụ từ reward/table tăng.

## Giới hạn suy luận

B thay cả ngân sách và lịch khám phá: không tách tác dụng riêng từng yếu tố.
SUMO cùng seed có thể khác quỹ đạo; hai exploration seeds không đủ suy luận nhân quả
hay thống kê chắc chắn. Training không bật tripinfo nên chỉ có inserted/arrived thô.
Pilot không tính vào 60 episode chính. Không thêm log lớn lặp lại, không xóa raw.
