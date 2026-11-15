# Kế hoạch giai đoạn 5

Phạm vi: yêu cầu GIAI ĐOẠN 5 trong chat và AGENTS.md. Triển khai trực tiếp
trong workspace hiện tại; không commit/push, cài gói, sửa data hay môi trường.

Thiết kế: dictionary tuple-state → list Q theo đúng env.actions. Nhóm lane
biên theo edge khai báo trong net.xml, sắp edge ID, cộng từng lane duy nhất;
encode mức 0 / 1–4 / ≥5, nối ID pha. JSON lưu mapping/encoding, actions,
hyperparameter/lịch epsilon, số episode, seed, phiên bản SUMO và hash nguồn.
Greedy hòa lấy action đầu; evaluation dùng zero-row tạm cho state lạ, không
chèn state. Discount theo quyết định; reward/chuyển pha giữ nguyên SumoEnv.

- [x] Viết test_q_learning.py trước: nhóm đúng edge/thứ tự/ngưỡng, lane trùng,
  action ID không liên tiếp, epsilon 0/1/tie, init, Q âm, update tay/terminal,
  validation alpha/gamma, JSON round-trip/sai map/cấu hình/ghi đè.
- [x] Chạy test thấy thiếu triển khai; tạo các hàm build_encoding, encode_state,
  get_q_values, choose_action, update_q_value, save_q_table, load_q_table.
- [x] Test rồi triển khai train/evaluate và runner dùng env.step: online update,
  terminal không encode, finally đóng SUMO cả khi lỗi. CSV actions/signals và
  XML/log theo FT/MP; summary theo episode. Output/checkpoint không ghi đè.
- [x] Chạy toàn bộ pytest; chạy 1 episode trước, sau đó thêm 2 episode mỗi map
  trong cùng lượt train 3 episode, SUMO seed 23423, exploration seed 7.
  Evaluate model cuối mỗi map bằng seed 42 và 12345. Không train dài.
- [x] Kiểm tra trace thật: timing/clearance/reward/action, update từ log,
  evaluate không thay model, tổng queue/collision/teleport đối chiếu XML.
  Phân tích bin queue thực và coverage; viết README tiếng Việt, lệnh và vị trí
  thuyết trình. Kiểm tra lại toàn suite và Git/data hash.

Điểm rà soát: terminal có phase=None; all-red có thể kéo tới hết episode;
state chưa thấy khi greedy; Q âm so với cột chưa học=0; model sai map/action/
lane order hoặc JSON hỏng; lỗi controller phải đóng process. Test phải kiểm
tra các trường hợp này. Không thêm resume/checkpoint overwrite/heuristic.

Kết quả: 6 train + 4 evaluate đủ 3600 s, 99 test passed (49 test cũ).
Rà soát độc lập: không Critical/Important; lưu ý unseen_decisions khi train
không phải coverage vì bootstrap tạo successor row, đã ghi rõ trong README.
Giữ nguyên ngưỡng và mọi kết quả xấu; chưa hội tụ, Cologne1 evaluate bỏ phục
vụ 10 movement và có 67/68 teleport. Chưa có bằng chứng thắng baseline.
