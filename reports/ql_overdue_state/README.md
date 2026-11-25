# Thí nghiệm 2 — Ghi hướng chờ xanh lâu nhất vào state trên Cologne1

**Giữ QL-Overdue làm biến thể cho Cologne1.** Biến thể đạt cả hai điều kiện validation đã chốt trước khi chạy. Trên ba seed test mới, tỷ lệ hoàn thành tăng từ 56,89% lên 82,08% và chuỗi giữ cùng pha dài nhất giảm từ 107 xuống 7 quyết định. Một movement vẫn thiếu xanh ưu tiên tới 3.130 giây. Cả FT và MP đều hoàn thành nhiều chuyến hơn.

## Thay đổi được thử

QL-Original dùng các bin queue theo edge vào và ra rồi nối ID pha xanh. QL-Overdue nối thêm một số `d` vào cuối khóa. Giá trị 0 nghĩa là chưa có edge vào quá hạn. Các giá trị 1–4 chỉ edge có queue quan sát được và thiếu xanh ưu tiên lâu nhất, từ 300 giây trở lên. Khi bằng tuổi, chọn edge đứng trước trong metadata. Đồng hồ bắt đầu tại quyết định quan sát thấy queue dương, đặt lại khi thấy queue bằng 0 hoặc khi log tín hiệu ghi nhận bất kỳ movement nào của edge nhận `G`. Model ghi phiên bản `overdue-edge-v1`. Biến thể vẫn tự chọn mọi pha xanh hợp lệ. [Mã hóa và tuổi nhu cầu](../../src/q_learning.py)

Cả hai dùng cùng reward `r = −queue_seconds / elapsed` và cập nhật `Q(s,a) ← Q(s,a) + α [target − Q(s,a)]`. Bước thường có `target = r + γ max Q(s′,·)`. Bước terminal có `target = r`. Cả hai khám phá đều trong các pha ở nhánh ε-greedy, chọn pha đầu tiên khi hòa Q ở nhánh greedy và chiết khấu theo quyết định. [Chọn pha và cập nhật Q](../../src/q_learning.py) · [Tính reward](../../src/sumo_env.py)

| Tập | Cách chạy |
|---|---|
| Train | Mỗi biến thể có hai model học từ đầu, seed khám phá 7 và 17, SUMO seed 23423, 15 episode/model. Dùng lại hai checkpoint Original đã lưu, train mới hai model Overdue. Bin 0 và 4, α = 0,1, γ = 0,95, ε giảm tuyến tính từ 1 xuống 0,05. Phát lại đủ 60 episode, khóa SHA của cả bốn checkpoint cuối trước validation. |
| Validation | SUMO seed 42 và 12345, hai lần lặp/seed cho mỗi model, tổng 16 lượt. Giữ biến thể nếu gap cực đại giảm **riêng ở từng seed** và tỷ lệ hoàn thành trung bình trên cả tập ít nhất bằng Original. |
| Test | Khóa quyết định validation trước test. Chạy mới FT, MP và cả bốn model QL trên seed 84201, 84202, 84203, hai lần lặp/seed, tổng 36 lượt. Mỗi lượt dùng cùng demand 2.015 chuyến, 3.600 giây, cơ chế chuyển pha và bộ tính metric. Các lượt QL đều dùng greedy, Guard tắt. |

Thiết kế chỉ xét Cologne1. Trong các trace train/validation đối chứng đã khảo sát trên Ingolstadt1, biến quá hạn chưa từng kích hoạt. [Kế hoạch và phiên bản](../../experiments/overdue-eval-ukeh31e3/plan.json) · [Model đã khóa](../../experiments/overdue-eval-ukeh31e3/models.json) · [Kết quả validation đã khóa](../../experiments/overdue-eval-ukeh31e3/validation_lock.json)

## Validation

Gap là khoảng dài nhất một movement liên tục thiếu xanh ưu tiên `G`, tính cả hai đầu khung mô phỏng. Với mỗi seed, lấy gap lớn nhất trong hai model × hai lần lặp. Completion tính trung bình tám lượt cho mỗi biến thể.

| Điều kiện | QL-Original | QL-Overdue |
|---|---:|---:|
| Gap cực đại ở seed 42 (s) | 3.440 | 3.280 |
| Gap cực đại ở seed 12345 (s) | 3.600 | 3.130 |
| Completion trung bình | 55,65% | 79,98% |

## Test trên ba seed giữ riêng

| Controller | Completion | Throughput (xe/h) | Chờ (s/xe) | Đi lại (s/xe) | Time loss (s/xe) | Queue (xe) | Teleport/lượt | Gap cực đại (s) | Chuỗi cùng pha dài nhất (quyết định) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FT | 93,73% | 1.888,67 | 146,96 | 190,46 | 167,48 | 74,43 | 0,00 | 250 | 1 |
| MP | 97,58% | 1.966,17 | 64,28 | 100,62 | 77,80 | 28,19 | 10,67 | 770 | 2 |
| QL-Original | 56,89% | 1.146,33 | 91,81 | 124,07 | 103,50 | 108,65 | 65,50 | 3.600 | 107 |
| QL-Overdue | 82,08% | 1.654,00 | 112,81 | 155,25 | 132,17 | 66,20 | 47,58 | 3.130 | 7 |

Hai cột cuối lấy cực đại qua mọi lượt. Các cột khác lấy trung bình từng lượt với trọng số bằng nhau, 6 lượt/controller cho FT và MP, 12 cho mỗi biến thể QL. Completion và throughput chỉ đếm xe đến đích hợp lệ trong giờ mô phỏng, loại xe bị xóa hoặc từng teleport. Chờ và time loss lấy từ SUMO, thời gian đi lại tính từ lúc vào mạng đến khi tới đích. Ba thời gian chỉ tính trên chuyến hoàn thành hợp lệ. Queue tính trên mọi giây của các làn vào, teleport đếm sự kiện bắt đầu. [Định nghĩa metric](../../src/evaluation_metrics.py) · [52 lượt riêng lẻ](runs.csv) · [Trung bình, min–max và độ lệch chuẩn mẫu](summary.csv)

QL-Overdue tăng completion ở cả hai seed khám phá. Seed 7 tăng từ 69,30% lên 83,31%, seed 17 từ 44,48% lên 80,86%. Số xe chưa vào được mạng giảm từ 596,25 xuống 187,33 xe/lượt. Trên nhóm chuyến hoàn thành, chờ tăng 21,00 s/xe và cả thời gian đi lại lẫn time loss đều tăng. Hai policy đưa những tập xe khác nhau tới đích nên số liệu này chưa cho biết thay đổi thời gian chờ của từng xe. [Theo seed khám phá](../../experiments/overdue-eval-ukeh31e3/summary_by_training_seed.csv)

### Vì sao vẫn còn hướng thiếu xanh?

Số quyết định chọn lại pha ngay trước đó giảm từ trung bình 48,83 xuống 9,00 lần/lượt. Chuỗi xanh thực dài nhất giảm từ 3.190 xuống 210 giây. Nhưng dù agent đổi pha, nhiều pha khác nhau vẫn có thể cùng bỏ qua một movement. Ở cả ba seed test, gap cực đại của QL-Overdue đều là 3.130 giây.

Phát lại trace test ghi nhận 359 quyết định của QL-Overdue có edge quá hạn. Chỉ 2 lần edge lâu nhất nhận `G` ngay trong quyết định đó. Trong 359 lần này, 128 lần gặp state chưa từng thấy trong train, và chỉ 155 lần có ít nhất một pha cấp xanh cho edge từng được cập nhật ở đúng state. Original có 610 quyết định quá hạn và cấp xanh ngay 10 lần. Đây là số đếm trên quỹ đạo riêng của từng policy. Một giá trị `d` đại diện cả edge, trong khi metric gap đo từng movement và tính cả lúc thiếu nhu cầu. [Đối chiếu queue, state và tín hiệu từng lượt](../../experiments/overdue-eval-ukeh31e3/edge_service.csv)

### Coverage, checkpoint và độ biến thiên

| Seed khám phá | Cặp đã cập nhật / số cặp trên state đã gặp, Original → Overdue | State chung đổi action greedy từ checkpoint 14 đến 15 | Max \|ΔQ\| |
|---|---|---|---|
| 7 | 467/920 (50,76%) → 587/1.408 (41,69%) | 35/113 → 29/172 | 17,19 → 17,08 |
| 17 | 472/976 (48,36%) → 536/1.256 (42,68%) | 31/119 → 34/154 | 18,87 → 18,51 |

Thêm `d` làm tăng số state quan sát được. Dù cập nhật được nhiều cặp hơn ở cả hai seed, tỷ lệ bao phủ giảm. Trên test, trung bình tỷ lệ quyết định ở state mới tăng từ 2,58% lên 13,91%, còn tỷ lệ chọn cặp chưa cập nhật tăng từ 16,24% lên 42,12%. Với checkpoint cuối vẫn đổi action greedy trên nhiều state chung, 15 episode chưa đủ để kết luận hội tụ. [Replay từng checkpoint](../../experiments/overdue-eval-ukeh31e3/training.csv)

Completion trên 12 lượt test Original trải từ 37,32% đến 71,91% (độ lệch chuẩn 13,77 điểm phần trăm), Overdue từ 80,25% đến 83,87% (1,37 điểm). Các số này gộp biến thiên theo seed khám phá, seed SUMO và lần lặp. Cùng model Original seed khám phá 7 và cùng SUMO seed 84201, hai lượt hoàn thành 1.341 và 1.449 chuyến. Chênh lệch lớn nhất giữa hai repeat của Overdue là 16 chuyến, FT là 22 và MP là 4. Khi đổi seed FT từ 84201 sang 84202, 1.902/1.959 xe chung đổi `speedFactor`. Phạm vi kết luận là một bản đồ, một demand và hai seed khám phá. [Chênh lệch từng cặp](../../experiments/overdue-eval-ukeh31e3/repeat_spread.csv) · [Ảnh hưởng của SUMO seed](../../experiments/overdue-eval-ukeh31e3/seed_effect.csv)

## Kiểm thử và hàm để thuyết trình

`python -B -m pytest -q -p no:cacheprovider` đạt **184/184 test**, gồm kiểm tra ngưỡng 300 giây, thứ tự khi hòa, reset theo tín hiệu thực, metadata và train/evaluate SUMO ngắn. Bộ chạy ma trận dùng `src/q_learning.py train` và `src/evaluate.py`. Sau mô phỏng, `verify_and_summarize.py` kiểm đủ 52 lượt, tính lại metric từ CSV/XML, đối chiếu tín hiệu từng giây, pha vàng/tất cả đỏ, thứ tự và 30 giây xanh của FT ở các quyết định trọn vẹn, SHA của từng model trước và sau evaluate. `check_edge_service.py` phát lại 40 lượt QL để đếm việc cấp xanh cho edge quá hạn. [Kiểm thử state](../../tests/test_overdue_state.py) · [Lệnh chạy ma trận](../../experiments/overdue-eval-ukeh31e3/run_evaluation_matrix.py) · [Biên bản kiểm chứng](../../experiments/overdue-eval-ukeh31e3/verified.json)

| Nội dung | Hàm và cách giải thích |
|---|---|
| Lấy state | `SumoEnv.get_state()` đọc queue và pha hiện tại. [Môi trường](../../src/sumo_env.py) |
| Mã hóa state | `EdgeDemand.observe()` tính mã edge quá hạn, `record_green()` ghi nhận xanh thực và `encode_state()` ghép bin queue, pha và mã edge thành khóa Q. [Q-Learning](../../src/q_learning.py) |
| Chọn action | `choose_action()` chọn pha xanh hợp lệ theo ε-greedy khi train và theo Q lớn nhất khi evaluate. [Q-Learning](../../src/q_learning.py) |
| Cập nhật và lưu | `update_q_value()` tính target và cập nhật một ô Q. `save_q_table()` ghi bảng cùng metadata để tải đúng encoding. [Q-Learning](../../src/q_learning.py) |
| Gửi action và tính reward | `SumoEnv.step()` nhận pha đã chọn, thực hiện chuyển tiếp và trả reward. `_run_signal()` gửi chuỗi đèn đến SUMO qua TraCI, lấy mẫu từng giây. [Môi trường](../../src/sumo_env.py) |
| Đánh giá | `run_evaluation()` chạy controller. `greedy_steps()` tra model đã khóa. `trip_metrics()` và `signal_metrics()` tổng hợp chuyến xe, queue và khoảng thiếu xanh. [Runner](../../src/evaluate.py) · [Metric](../../src/evaluation_metrics.py) |
