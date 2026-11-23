# Thu nhỏ bảng Q bằng cách ghi hướng có hàng đợi dài nhất

**Quyết định trên Cologne1: giữ QL-Overdue.** QL-Busiest giảm 28,4–31,2% số dòng sau khi huấn luyện lại, nhưng trượt tiêu chí giao thông trên validation. Trên cả ba seed test mới, tỷ lệ hoàn thành của biến thể đều thấp hơn đối chứng QL-Overdue. Mọi checkpoint và lượt chạy được giữ để đối chiếu.

## Thay đổi và cách kiểm tra

QL-Overdue chia queue mỗi edge vào thành ba mức (0, 1–4, ≥5). QL-Busiest thay bốn bin đó bằng bốn dấu hiệu có xe dừng và mã edge có **tổng queue thực** lớn nhất. Mã bằng 0 khi mọi edge trống. Khi hòa, lấy edge đầu tiên theo thứ tự trong metadata. Cả hai giữ bin đầu ra, ID pha xanh hiện tại, mã edge chờ xanh ít nhất 300 giây và đủ tám action. Chỉ cần thêm `--busiest-state` khi train. Khi evaluate, model cung cấp phiên bản `overdue-busiest-v1`. [Cách chạy](../../README.md) · [Mã hóa state](../../src/q_learning.py#L89)

Cả hai dùng `r = −queue_seconds / elapsed` và `Q(s,a) ← Q(s,a) + α[target − Q(s,a)]`. Bước thường có `target = r + γ max Q(s′,·)`. Bước terminal có `target = r`. Train từ đầu 15 episode/model với α = 0,1, γ = 0,95 theo quyết định, ε giảm từ 1 xuống 0,05, SUMO seed 23423 và seed khám phá 7, 17. Dùng lại các checkpoint đối chứng đã lưu, đánh giá mới mọi controller qua cùng runner. [Công thức cập nhật](../../src/q_learning.py#L147) · [Kế hoạch đã khóa](../../../TrafficSignalControl-ML-busiest-82mrs6qf/plan.json)

| Sau 15 episode, Overdue → Busiest | Seed khám phá 7 | Seed khám phá 17 |
|---|---:|---:|
| Số dòng thực lưu | 176 → 126 | 157 → 108 |
| Số ô Q (8 action/dòng) | 1.408 → 1.008 | 1.256 → 864 |
| Cặp đã cập nhật / tổng số ô | 41,69% → 44,15% | 42,68% → 48,73% |
| State chung đổi pha greedy giữa checkpoint 14 và 15 | 29/172 → 29/125 | 34/154 → 31/107 |

Số dòng trên lấy từ model sau khi train với mã hóa mới. Đếm lại trace cũ trước thí nghiệm từng cho 122 và 112 dòng. Policy mới đi qua tập state khác khi tương tác với SUMO. [Checkpoint và SHA](../../../TrafficSignalControl-ML-busiest-82mrs6qf/models.json) · [Kết quả từng episode](../../../TrafficSignalControl-ML-busiest-82mrs6qf/training.csv)

## Validation quyết định có giữ biến thể

Chốt trước khi train: cả hai bảng phải nhỏ hơn, completion trung bình phải tăng, queue trung bình phải giảm. Với **từng seed validation**, số teleport trung bình và khoảng thiếu xanh cực đại của một movement đều phải giữ hoặc giảm. Mỗi seed có hai model × hai lượt lặp cho mỗi biến thể. Kết quả dưới đây ghi Overdue → Busiest.

| SUMO seed | Completion | Queue trung bình (xe) | Teleport/lượt | Thiếu xanh cực đại (s) |
|---|---:|---:|---:|---:|
| 42 | 74,43% → 84,04% | 65,31 → 81,83 | 38,75 → 22,50 | 3.280 → 2.200 |
| 12345 | 85,61% → 67,27% | 60,76 → 66,98 | 41,50 → 57,00 | 3.130 → 3.120 |

Trên 16 lượt validation, completion trung bình giảm từ **80,02% xuống 75,66%**, queue tăng từ **63,03 lên 74,41 xe**. Biến thể đạt điều kiện kích thước và khoảng thiếu xanh, trượt điều kiện completion, queue và teleport. Kết luận được ghi trước khi bắt đầu bất kỳ lượt test nào. [Quyết định đã khóa](../../../TrafficSignalControl-ML-busiest-82mrs6qf/validation_lock.json)

## Test trên seed 95301, 95302 và 95303

Mỗi lượt dùng cùng demand 2.015 chuyến, cùng thời gian 3.600 giây, chuyển pha và định nghĩa metric. Mỗi controller có hai repeat/seed SUMO. Ba dòng QL mỗi dòng gồm 12 lượt vì có hai model đã khóa ở episode 15. FT và MP mỗi dòng gồm 6 lượt. QL dùng greedy, Guard tắt.

| Controller | Completion | Throughput (xe/h) | Chờ (s/xe) | Đi lại (s/xe) | Time loss (s/xe) | Queue (xe) | Teleport/lượt | Thiếu xanh cực đại (s) | Chuỗi cùng pha dài nhất |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FT | 93,39% | 1.881,83 | 146,77 | 190,15 | 167,20 | 74,12 | 0,00 | 250 | 1 |
| MP | 97,61% | 1.966,83 | 64,78 | 101,22 | 78,44 | 28,91 | 10,50 | 970 | 2 |
| QL-Original | 56,95% | 1.147,50 | 86,36 | 118,11 | 97,44 | 107,37 | 65,67 | 3.600 | 107 |
| QL-Overdue | 78,85% | 1.588,83 | 114,12 | 157,47 | 134,02 | 67,05 | 40,50 | 3.360 | 9 |
| QL-Busiest | 66,43% | 1.338,58 | 103,69 | 141,62 | 117,72 | 71,77 | 49,58 | 3.440 | 51 |

Hai cột cuối là cực đại qua các lượt. Các cột còn lại là trung bình từng lượt. Completion và throughput đếm xe đến đích hợp lệ trong thời hạn, loại xe bị xóa hoặc từng teleport. Ba chỉ số thời gian chỉ tính trên các chuyến hợp lệ ấy. Số xe chưa vào được mạng tăng từ 274,25 lên 493,58 xe/lượt ở Busiest, dù nhóm xe về đích có thời gian chờ trung bình thấp hơn. Queue đo xe dừng trên các làn vào theo giây, kể cả vàng/đỏ. Teleport đếm số sự kiện bắt đầu. [Từng lượt](runs.csv) · [Trung bình, min–max và độ lệch chuẩn mẫu](summary.csv)

## Vì sao bảng nhỏ hơn vẫn có thể chọn pha kém?

- **Cùng mã, khác mức ùn tắc.** Trong train seed 7, hai vector queue vào `(46, 0, 13, 18)` và `(107, 0, 13, 1)` có cùng dấu hiệu hiện diện, cùng hướng đông nhất, pha và mã quá hạn. Trên trace train mới, 56/126 khóa ở seed 7 và 58/108 ở seed 17 gộp nhiều tổ hợp bin đầu vào cũ. Đây là bằng chứng mất thông tin. Mức đóng góp riêng của nó vào thay đổi completion chưa được đo. [Trace train](../../../TrafficSignalControl-ML-busiest-82mrs6qf/training/cologne1/)
- **Đã gặp state vẫn có thể chọn pha chưa học.** Tỷ lệ quyết định test ở state chưa thấy giảm từ 21,19% xuống 12,00%, còn tỷ lệ chọn cặp state–action chưa cập nhật tăng từ 53,31% lên 62,79%. Ở seed khám phá 17, SUMO seed 95301, repeat 1, agent giữ pha 2 liên tục 51 quyết định (1.530 giây xanh). Trong 50 quyết định cùng khóa của chuỗi, năm pha hòa Q lớn nhất và pha 2 được chọn theo thứ tự, dù ô đó có 0 cập nhật. [Trace của chuỗi](../../../TrafficSignalControl-ML-busiest-82mrs6qf/test/cologne1/QL-Busiest/exploration_17/seed_95301/repeat_1/actions.csv) · [Đối chiếu pha và số lần cập nhật](../../../TrafficSignalControl-ML-busiest-82mrs6qf/edge_service.csv)
- **Kết quả phụ thuộc seed và repeat.** Completion trung bình test của Overdue → Busiest giảm ở cả seed khám phá 7 (83,71% → 79,56%) và 17 (74,00% → 53,30%). Trên 12 lượt mỗi biến thể, khoảng min–max là 66,00–84,27% và 37,57–92,75%, độ lệch chuẩn tương ứng 6,99 và 19,77 điểm phần trăm. Riêng Busiest seed khám phá 7, SUMO seed 95302, hai repeat hoàn thành 1.856 và 1.359 chuyến. Dữ liệu gồm một demand của Cologne1 và hai seed khám phá. Checkpoint cuối vẫn đổi action greedy. [Biến thiên giữa repeat](../../../TrafficSignalControl-ML-busiest-82mrs6qf/repeat_spread.csv)

## Kiểm chứng

`python -B -m pytest -q -p no:cacheprovider` đạt **186/186 test** trước khi huấn luyện, gồm một lượt train và một lượt evaluate SUMO ngắn cho mỗi kiểu state Overdue và Busiest. Đã phát lại 90 episode của sáu model để đối chiếu state, action, Bellman và terminal. Sau mô phỏng, đã tính lại metric từ XML/CSV, kiểm tra tín hiệu từng giây, thời gian pha FT, và SHA model bất biến cho đủ **16 lượt validation và 48 lượt test**. [Test mã hóa và tích hợp](../../tests/test_overdue_state.py) · [Biên bản kiểm chứng](../../../TrafficSignalControl-ML-busiest-82mrs6qf/verified.json)

Khi thuyết trình, bắt đầu ở `get_state()`, theo `encode_state()` → `choose_action()` → `SumoEnv.step()`. Trong `step()`, `_run_signal()` gửi lệnh đến SUMO, reward bằng âm số xe chờ trung bình trong bước. Khi train, `update_q_value()` cập nhật một ô và `save_q_table()` lưu checkpoint. Khi evaluate, `run_evaluation()` gọi `greedy_steps()` với model cố định. [Môi trường](../../src/sumo_env.py#L182) · [Q-Learning](../../src/q_learning.py#L89) · [Đánh giá](../../src/evaluate.py#L60)
