# Thí nghiệm 1 — Ưu tiên pha chưa cập nhật khi khám phá

**Giữ QL-Original làm mặc định trên cả hai bản đồ.** Biến thể `untried-first` thực hiện đúng quy tắc khám phá, nhưng trượt tiêu chí validation ở cả hai bản đồ. Trên test Cologne1, biến thể giảm tỷ lệ chọn action chưa học từ 14,28% xuống 6,39%. Tuy vậy, tỷ lệ hoàn thành chuyến giảm 7,33 điểm phần trăm và chuỗi giữ xanh dài nhất vẫn đạt 3.190 giây. Tên `QL-Improved` trong các bảng chỉ biến thể được thử.

## Phép thử

| Thành phần | Thiết kế |
|---|---|
| Thay đổi | Ở nhánh khám phá, Original lấy mẫu đều trong mọi pha hợp lệ. Biến thể lấy mẫu đều trong những pha có `N(s,a)=0` nếu còn pha như vậy, sau đó trở về lấy mẫu đều trong mọi pha. Bộ đếm tăng sau mỗi cập nhật Q thành công, kể cả khi Q vẫn bằng 0. |
| Công thức trước và sau | Cả hai dùng `r = −queue_seconds / elapsed` và `Q(s,a) ← Q(s,a) + α[target − Q(s,a)]`. Bước thường dùng `target = r + γ max Q(s′,·)`, bước terminal dùng `target = r`. |
| Training | Mỗi bản đồ dùng hai model cho mỗi chiến lược, một model cho mỗi seed khám phá 7 và 17. Mỗi model học 15 episode với SUMO seed 23423, bin 0 và 4, α = 0,1, γ = 0,95 theo quyết định, ε giảm tuyến tính từ 1 xuống 0,05. Dùng lại bốn model Original đã lưu, train mới bốn model biến thể. Replay 120 episode và khóa SHA của cả tám checkpoint cuối. |
| Validation | Seed SUMO 42 và 12345, hai repeat mỗi seed, tổng 32 lượt. Xét riêng từng bản đồ. Chấp nhận khi đồng thời giảm tỷ lệ trung bình chọn action chưa cập nhật, giữ hoặc tăng completion trung bình và giữ hoặc giảm gap cực đại. |
| Test | Seed SUMO mới 73101, 73102 và 73103, hai repeat mỗi seed. Chạy mới FT, MP và cả hai model mỗi chiến lược QL bằng cùng runner. Tổng 72 lượt. Mỗi lượt đủ 3.600 giây, cùng demand và chuyển pha an toàn. QL dùng greedy, Guard tắt. |

Tất cả model ở episode 15 đều vào test. Với từng controller và bản đồ, số liệu trung bình có trọng số bằng nhau giữa các lượt. FT và MP có 6 lượt mỗi dòng, QL có 12 lượt vì gồm hai seed khám phá. Trong lần chạy đầu, bốn lượt validation Cologne1 seed 12345 repeat 1 dừng giữa mô phỏng. Đã giữ trace dở và chạy lại đúng bốn cấu hình vào thư mục riêng. Kết quả mỗi cấu hình chỉ dùng một lượt đầy đủ. [Kế hoạch](../../../TrafficSignalControl-ML-exploration-204c7770/plan.json) · [Model đã khóa](../../../TrafficSignalControl-ML-exploration-204c7770/models.json) · [Quyết định trước test](../../../TrafficSignalControl-ML-exploration-204c7770/validation_lock.json)

## Validation quyết định giữ hay thay

Tỷ lệ chưa học là phần trăm quyết định chọn cặp `(state, action)` có **0 lần cập nhật trong train**, tính từng lượt rồi lấy trung bình. Gap là khoảng dài nhất một movement thiếu xanh ưu tiên `G`, tính cả hai biên thời gian mô phỏng.

| Bản đồ | Chọn action chưa học, Original → biến thể | Completion, Original → biến thể | Gap cực đại (s), Original → biến thể | Kết quả |
|---|---:|---:|---:|---|
| Cologne1 | 15,40% → 9,19% | 55,83% → 42,36% | 3.600 → 3.480 | Trượt completion |
| Ingolstadt1 | 0,51% → 0,76% | 92,63% → 88,51% | 1.114 → 1.634 | Trượt cả ba điều kiện |

## Test trên các seed giữ riêng

| Bản đồ | Controller | Completion | Throughput (xe/h) | Chờ (s/xe) | Đi lại (s/xe) | Time loss (s/xe) | Queue (xe) | Teleport/lượt | Gap cực đại (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cologne1 | FT | 93,72% | 1.888,50 | 146,54 | 190,03 | 167,00 | 74,49 | 0,00 | 250 |
| Cologne1 | MP | 97,50% | 1.964,67 | 70,84 | 107,91 | 85,01 | 29,40 | 12,33 | 770 |
| Cologne1 | QL-Original | 49,69% | 1.001,25 | 112,81 | 147,01 | 125,61 | 115,22 | 70,00 | 3.600 |
| Cologne1 | QL-Improved | 42,36% | 853,50 | 224,93 | 268,73 | 246,38 | 117,29 | 77,50 | 3.600 |
| Ingolstadt1 | FT | 98,27% | 1.686,33 | 31,12 | 63,29 | 42,50 | 9,21 | 0,00 | 84 |
| Ingolstadt1 | MP | 97,65% | 1.675,67 | 28,92 | 60,39 | 39,63 | 5,61 | 0,00 | 312 |
| Ingolstadt1 | QL-Original | 94,12% | 1.615,17 | 53,40 | 84,56 | 64,44 | 8,35 | 5,00 | 1.220 |
| Ingolstadt1 | QL-Improved | 89,53% | 1.536,33 | 41,74 | 71,21 | 51,37 | 6,70 | 9,67 | 1.634 |

Completion chia số chuyến hợp lệ cho nhu cầu 2.015 chuyến ở Cologne1 hoặc 1.716 ở Ingolstadt1. Throughput quy đổi số chuyến đó ra xe/giờ. Chuyến hợp lệ phải đến đích trong horizon, chưa bị loại và chưa từng teleport. Thời gian chờ và time loss lấy từ SUMO, thời gian đi lại tính từ lúc xe thực sự vào mạng tới lúc đến đích. Ba trung bình này chỉ dùng xe hoàn thành hợp lệ. Queue là số xe dừng trung bình theo giây trên các làn vào, tính cả vàng và đỏ. Teleport đếm sự kiện bắt đầu teleport. Chỉ cột gap lấy cực đại qua mọi lượt. [Định nghĩa metric](../../src/evaluation_metrics.py) · [Từng lượt](runs.csv) · [Trung bình, min–max và độ lệch chuẩn mẫu](summary.csv)

**Ingolstadt1 có sự đánh đổi rõ nhất.** Biến thể giảm queue trung bình từ 8,35 xuống 6,70 xe và giảm chờ trên nhóm hoàn thành từ 53,40 xuống 41,74 s/xe. Đồng thời, số xe chưa vào được mạng trung bình tăng từ 69,83 lên 132,17 xe/lượt, completion giảm và teleport tăng. Không thể suy ra toàn bộ người đi đường chờ ít hơn từ trung bình của nhóm về đích.

### Lặp pha và thiếu phục vụ

Trên Cologne1, chuỗi dài nhất trong mỗi lượt trung bình giảm từ **53,83 xuống 42,83 quyết định**, nhưng cả hai chiến lược đều có một chuỗi **107 quyết định, 3.190 giây xanh thực**. Với biến thể e7, seed 73101 repeat 1, agent giữ pha 7 và gặp cùng khóa `(2,2,2,2,0,0,0,0,7)` ở 106 quyết định trong chuỗi. Pha 7 có Q lớn nhất duy nhất và đã được cập nhật 7 lần tại khóa này. Ba movement không nhận `G` trong suốt lượt. Chỉ 122/2.015 chuyến hoàn thành hợp lệ. Cơ chế ưu tiên action mới lúc train vẫn cho phép policy greedy lặp lại một lựa chọn đã học. [Actions của lượt này](../../../TrafficSignalControl-ML-exploration-204c7770/test/cologne1/QL-Improved/exploration_7/seed_73101/repeat_1/actions.csv) · [Kết quả SUMO](../../../TrafficSignalControl-ML-exploration-204c7770/test/cologne1/QL-Improved/exploration_7/seed_73101/repeat_1/result.json)

Trên Ingolstadt1, số quyết định lặp action ngay trước đó giảm từ 28,83 xuống 22,50 lần/lượt, nhưng cả hai vẫn có chuỗi dài nhất 7 quyết định. Gap cực đại của biến thể tăng thêm 414 giây. Gap mô tả tín hiệu cấp cho movement, không đo thời gian chờ của từng xe.

### Coverage và ổn định qua checkpoint

Coverage bằng số cặp thực cập nhật chia cho số state đã gặp nhân số pha hợp lệ. “Đổi greedy” tính trên các state có mặt ở cả checkpoint 14 và 15. Trong bảng, mỗi ô ghi **Original → biến thể**.

| Bản đồ / seed khám phá | Cặp cập nhật / mẫu số | Coverage | Đổi greedy 14→15 | Max ΔQ tuyệt đối |
|---|---|---|---|---|
| Cologne1 / 7 | 467/920 → 492/1.000 | 50,76% → 49,20% | 35/113 → 28/125 | 17,19 → 17,02 |
| Cologne1 / 17 | 472/976 → 507/984 | 48,36% → 51,52% | 31/119 → 28/121 | 18,87 → 13,20 |
| Ingolstadt1 / 7 | 49/51 → 52/57 | 96,08% → 91,23% | 8/17 → 8/19 | 5,42 → 6,69 |
| Ingolstadt1 / 17 | 50/60 → 52/54 | 83,33% → 96,30% | 9/20 → 10/18 | 4,01 → 6,56 |

Trong train Cologne1, biến thể có 472 quyết định khám phá ở state có cả pha đã thử và chưa thử. Cả 472 lần đều chọn pha chưa thử. Tuy vậy, số cặp chỉ nhận đúng một cập nhật tăng từ 234 lên 288 ở e7 và từ 271 lên 312 ở e17. Trên test, tỷ lệ chọn action chưa cập nhật giảm từ 14,28% xuống 6,39% ở Cologne1 và từ 0,67% xuống 0,16% ở Ingolstadt1. Số lần thử một pha và chất lượng ước lượng Q của pha đó là hai phép đo khác nhau. Checkpoint cuối vẫn đổi greedy trên cả bốn model mới. [Replay theo episode](../../../TrafficSignalControl-ML-exploration-204c7770/training.csv) · [Đếm cập nhật và nhánh khám phá](../../../TrafficSignalControl-ML-exploration-204c7770/exploration_checks.csv)

### Biến thiên và giới hạn kết luận

- **Seed khám phá.** Completion test Cologne1 của Original → biến thể là **69,82% → 21,01% ở e7**, còn **29,56% → 63,71% ở e17**. Chọn riêng e17 sẽ đảo kết luận. Trên Ingolstadt1, hai cặp tương ứng là 98,97% → 96,85% và 89,28% → 82,21%. [Theo seed train](../../../TrafficSignalControl-ML-exploration-204c7770/summary_by_training_seed.csv)
- **Cùng SUMO seed.** Cologne1 Original e7, seed 73103, hoàn thành 1.342 và 1.394 chuyến ở hai repeat. Chênh 52 chuyến dù dùng cùng model và demand. Chênh lớn nhất của FT là 11 chuyến. MP và biến thể ghi cùng số chuyến hợp lệ giữa hai repeat trong các cặp test lần này. Đổi seed FT từ 73101 sang 73102 làm `speedFactor` thay đổi ở 1.901/1.952 ID chung trên Cologne1 và 1.648/1.708 trên Ingolstadt1. [Các cặp repeat](../../../TrafficSignalControl-ML-exploration-204c7770/repeat_spread.csv) · [Đối chiếu seed](../../../TrafficSignalControl-ML-exploration-204c7770/seed_effect.csv)
- **Độ phân tán.** Completion test Cologne1 trải từ 7,99–71,41% ở Original và 6,05–68,64% ở biến thể, độ lệch chuẩn lần lượt 23,95 và 24,33 điểm phần trăm. Các khoảng này gộp seed train, seed SUMO và repeat. Hai seed khám phá, ba seed test và một bộ demand mỗi bản đồ chỉ đủ cho kết luận mô tả trong thí nghiệm này.

## Kiểm chứng và vị trí thuyết trình

Toàn bộ 181 test pytest đạt. Công cụ offline đã kiểm đủ 104 lượt evaluate, tính lại metric từ XML/CSV, đối chiếu tín hiệu theo giây, thứ tự và 30 giây xanh đầy đủ của FT, SHA model trước/sau evaluate. Cả hai biến thể QL sử dụng cùng cơ chế chuyển pha và định nghĩa metric với baseline. [Biên bản kiểm chứng](../../../TrafficSignalControl-ML-exploration-204c7770/verified.json)

| Nội dung | Hàm và vai trò |
|---|---|
| State và action | `SumoEnv.get_state()` đọc queue/pha. `encode_state()` cộng queue theo edge và chia bin. `choose_action()` chọn pha. [Môi trường](../../src/sumo_env.py) · [Q-Learning](../../src/q_learning.py) |
| Học và lưu | `update_q_value()` tính Bellman. `run_episode()` tăng bộ đếm sau cập nhật. `save_q_table()` ghi giá trị Q và metadata khám phá. [Q-Learning](../../src/q_learning.py) |
| Điều khiển và reward | `SumoEnv.step()` thực thi chuyển pha và trả reward theo queue. `_run_signal()` gửi lệnh TraCI và lấy mẫu từng giây. [Môi trường](../../src/sumo_env.py) |
| Đánh giá | `run_evaluation()` chạy từng controller, `greedy_steps()` chọn pha theo bảng Q đã khóa, `trip_metrics()` và `signal_metrics()` tính kết quả. [Runner](../../src/evaluate.py) · [Metric](../../src/evaluation_metrics.py) |
