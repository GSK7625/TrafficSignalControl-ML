# Giai đoạn 4 — MP, 15/11/2026

Phạm vi theo yêu cầu giai đoạn 4 và AGENTS.md; làm trực tiếp trong checkout
hiện tại, không commit/push, không sửa data/FT/môi trường, không cài gói.
Git ban đầu sạch, nhánh main. Baseline: 26 test passed.

Thiết kế đã nêu trước code: MP theo làn có phân bổ đều **xấp xỉ**,
`P(a) = sum((q_i-q_j)/d_i for (i,j) có G trong a)`.
`d_i` đếm cặp lane vào/ra duy nhất trong toàn mapping, không tính lại theo
pha. Trọng số có tổng bằng 1 trên mỗi làn vào; không phải tỷ lệ rẽ đo được
hay saturation flow hiệu chuẩn. Chỉ dùng state chung của SumoEnv, không đọc
route tương lai. Không nhận bảo đảm ổn định/thông lượng của MP lý tưởng.

- [x] Tạo test_mp.py: ví dụ tay, downstream, làn nhiều hướng, link trùng,
  thứ tự state, hòa/0/âm, action hợp lệ, reward độc lập, horizon/chuyển pha.
  Chạy test trước khi có mp.py để xác nhận thiếu implementation.
- [x] Tạo mp.py: movement_weights(env), calculate_pressures(env,state),
  choose_action(env,state), max_pressure(env), run_mp(map_name,output,seed).
  Tie: action đầu tiên trong env.actions; không clip âm/bonus/chống đói.
  run_mp đóng trong finally, từ chối ghi đè, lưu metadata/actions/signals và
  log/XML SUMO giống FT; kiểm tra kết thúc, process, số collision/teleport.
- [x] Chạy toàn bộ pytest; chạy 2 map × seeds 23423,42,12345 đủ 3600 s
  bằng .venv/SUMO hiện có. Không đổi time-to-teleport hay collision checking.
- [x] Kiểm tra lại CSV thực: queue/pressure/action tại quyết định,
  xanh 30 s trừ horizon cuối, vàng 8/6 s, đỏ >=2 s và clearance; cùng action
  không chuyển tiếp; reward từ queue-seconds; đủ 3600 mẫu; process đóng.
- [x] Báo cáo mọi lượt, kể cả xấu: đói pha/chọn liên tiếp, cảnh báo,
  teleport/collision, biến thiên seed; đối chiếu hash data với giai đoạn 3.
  Rà soát độc lập theo skill requesting-code-review trước bàn giao.

Kiểm tra trọng tâm: mẫu số lấy toàn mapping (không chuẩn hóa lại theo pha),
lọc G runtime thay vì xanh gốc, gộp cặp trùng trước khi chia trọng số,
không nhầm q_in theo movement với q_i nguyên làn, không cộng trùng số xe
downstream khi báo tổng queue; ghi nhận rủi ro head-of-line và đói pha.

Kết quả: 49 passed, 6 lượt cuối đủ 3600 s, reviewer không có finding.
Quyết định sửa phát sinh: dùng fractions.Fraction cho phép cộng trọng số để
không sai tie-break do làm tròn. Test hồi quy RED→GREEN; giữ cả sáu lượt
float đầu trong initial_float_runs và chạy lại sáu lượt, không ghi đè/xóa.
