# Stage 6.2 — Vì sao Q-Learning hoạt động kém? — 15/11/2026

## 1. Mục tiêu và kết luận trong phạm vi dữ liệu

Phân tích offline model cuối Stage 5, toàn bộ 6 episode training, 4 lượt evaluate
Stage 5 và 30 lượt chính Stage 6. Không sửa thuật toán, state/reward/tham số,
checkpoint, dữ liệu SUMO hoặc kết quả Stage 1–6.1; không chạy SUMO, train thêm,
cài thư viện, commit hoặc push.

**Cologne1 có một cơ chế lựa chọn đã xác minh từng quyết định:** coverage học
mỏng, Q chưa thử bằng 0 thắng Q âm, state lạ trả hàng 0 và tie-break lấy action
đầu. Trong các state thực gặp khi evaluate, mọi argmax cuối cùng đều là 0–3.
Signals xác nhận các action này chỉ mở movement 0–4/10–14, nên 10 movement còn
lại không có xanh cả giờ. Đây là giải thích trực tiếp chuỗi model → action →
tín hiệu; chưa định lượng được đóng góp nhân quả của từng yếu tố vào completion.

**Ingolstadt1 là đối chứng cho giới hạn của giải thích chỉ bằng tie-break:**
495/511 quyết định có cực đại duy nhất, vẫn cấp xanh đủ 8 movement, nhưng
completion trung bình 93,10% thấp hơn FT/MP trong tập lưu. Vì vậy đổi tie-break
không được xem là giải pháp đã chứng minh cho cả hai map.

## 2. Nguồn dữ liệu, SHA và khả năng tái tạo

Git ban đầu sạch trên `main`. Hai SHA dưới khớp Stage 5, metadata/result của
mọi lượt QL Stage 6 và checkpoint episode 003 (cả bytes). `analysis.json` lưu
SHA-256 của 164 tệp đầu vào được script dùng; `verification.json` ghi kiểm tra
bất biến toàn bộ 1.320 file được Git theo dõi trước nhiệm vụ.

| Map | Model | SHA-256 |
|---|---|---|
| Cologne1 | `../stage5/train/cologne1/cologne1_q_table.json` | `6b824ae7dbe63e93cef9f03bd1fa757d727c3b3153321121f7bdd8ad3bd9b4d9` |
| Ingolstadt1 | `../stage5/train/ingolstadt1/ingolstadt1_q_table.json` | `564a6fae2b8e22044fbaabd8b46ea38631f1f7fa170687eb992843cee1a73daa` |

Đã đọc AGENTS.md, `q_learning.py`, `test_q_learning.py`, `sumo_env.py`,
`evaluation_metrics.py`, `evaluate.py`, README/analysis/summary Stage 5,
README/runs/movement_service và raw runs Stage 6, README Stage 6.1.

- Train: `../stage5/train/<map>/episode_00{1,2,3}/actions.csv`, `signals.csv`,
  `result.json`, `metadata.json`, `q_table.json`; model cuối và training.csv.
- Evaluate cũ: `../stage5/evaluate/<map>/seed_{42,12345}/`; dùng actions/result
  cho phân tích policy; test cũ còn kiểm tra raw signals/statistics.
- Evaluate chính: `../stage6/runs/<map>/<FT|MP|QL>/seed_<seed>/` có
  actions/signals/result/metadata. Đối chiếu `../stage6/runs.csv` và
  `../stage6/movement_service.csv`. Giữ đủ seed 42, 12345, 2026, 2027, 2028.
- Stage 6.1 đã xác nhận sai khác quỹ đạo cùng seed/lịch đèn có thể phát sinh
  phía SUMO 1.28.0. Phân tích này mô tả chính xác file đã lưu; không coi seed
  kiểm soát hết biến thiên, không suy kiểm định nhân quả từ bảng mô tả.

Không cộng 4 lượt evaluate Stage 5 vào trung bình 5 seed Stage 6, không dùng
pilot/probe/lượt lặp thay kết quả chính. Training chỉ so giữa các episode;
không xếp hạng train với evaluate baseline.

## 3. State representation và phần thông tin bị mất

`SumoEnv.get_state()` lấy số xe dừng trên từng boundary lane duy nhất ở thời
điểm quyết định. `q_learning.build_encoding()` xác định lane thuộc edge từ XML,
sắp nhóm theo edge ID. `encode_state()` cộng queue nguyên trên các lane trong
nhóm rồi bin **0 / 1–4 / ≥5**, cuối tuple là **ID pha xanh trước action**.
Không cộng queue theo controlled link, không nhân một lane nhiều lần.

| Map | Incoming groups theo thứ tự | Outgoing groups theo thứ tự | Tuple/action |
|---|---|---|---|
| Cologne1 | -32038056#3; 23429231#1; 27115123#3; 28198821#3 | -28198821#4; 32038051#0; 32038056#0; 32324544#0 | 8 bin + phase; action 0,1,2,3,4,5,6,7 |
| Ingolstadt1 | 104010354; 164051413; 201963537#1 | -164051413; 104010475#0; 124812857#0 | 6 bin + phase; action 0,2,4 |

Phase ID 4 ở Ingolstadt1 là cột thứ 3 của Q, không phải cột thứ 5. Xanh mỗi
lượt yêu cầu 30 s; đổi pha thêm vàng 8/6 s và đỏ ít nhất 2 s, chờ clear nếu
cần; horizon có thể cắt lượt cuối. Vàng/đỏ không phải action.

Phân biệt bốn giới hạn:

1. **Discretization:** 5 và 113 xe cùng bin 2; đã có các tổng edge khác nhau
   cùng encoded state trong log, thống kê bên dưới. Điều này chứng minh mất
   thông tin, chưa chứng minh đổi ngưỡng sẽ tăng throughput.
2. **Gộp lane:** raw có lane queue nhưng mã hóa chỉ giữ tổng. Ví dụ train
   Cologne1, cùng phase 1 và edge queues `(0,32,12,18)`, nhóm `27115123#3`
   có lane `(5,7)` hoặc `(6,6)`, thành cùng tổng 12 trước cả khi bin.
3. **Không quan sát ngay từ SumoEnv:** ý định rẽ/route từng xe, queue theo
   movement trên lane dùng chung, xe đang chạy, tốc độ, tuổi chờ, internal
   queue trong state, thời gian tuyệt đối và nhu cầu tương lai không có trong
   tuple. Không thể phục hồi thông tin này chỉ bằng bin tốt hơn. Clearance
   có quan sát internal để thực thi an toàn nhưng không đưa nó vào state RL.
4. **Ngoài vùng đo:** xe chưa chèn và queue trên edge khác ở phía trước không
   thuộc boundary queues. Đây là thiếu quan sát, không phải lỗi rời rạc hóa.

Tất cả 4/3 chiều outgoing luôn bằng 0 trong cả train, evaluate Stage 5 và QL
Stage 6 ở các **thời điểm quyết định đã lưu**: không giúp phân biệt quan sát
trong tập này. Không suy rằng downstream không bao giờ tắc giữa hai quyết
định, hoặc chiều outgoing vô dụng trong mọi kịch bản. Mọi incoming bin và
phase đều có biến thiên; điều đó không tự chứng minh chúng đủ tính Markov.

Không gian lý thuyết 3^8×8=52.488 và 3^6×3=2.187 chỉ là cận tổ hợp. Với
outgoing cố định, cận tổ hợp trên các chiều còn biến thiên còn 648 và 81;
Ingolstadt1 còn 54 nếu giữ riêng thực tế edge164051413 chỉ có bin 0/1.
Đây không phải số state có thể đạt đã được chứng minh, hoặc mẫu số để tuyên
bố hội tụ. Coverage bên dưới dùng **state thực đã xuất hiện khi train**.

## 4. Phân bố queue bins và bằng chứng aliasing

Mỗi quan sát là một thời điểm quyết định, không phải một giây. `n0/n1/n2`
là số mẫu từng bin; tỷ lệ n2 có mẫu số là số quyết định trong nguồn tương ứng.
`queue_bins.csv` có đủ tỷ lệ cả ba bin, min/max/mean, min/max trong bin 2,
số mẫu **>5** và số giá trị queue khác nhau bị gộp vào bin 2.

| Map | Nguồn | Incoming edge | n0, n1, n2 | Bin 2 | Max xe | Mẫu >5 |
|---|---|---|---|---|---|---|
| cologne1 | stage5/train | -32038056#3 | 29, 22, 230 | 81.85% | 113 | 227 |
| cologne1 | stage5/train | 23429231#1 | 37, 2, 242 | 86.12% | 32 | 241 |
| cologne1 | stage5/train | 27115123#3 | 45, 26, 210 | 74.73% | 14 | 201 |
| cologne1 | stage5/train | 28198821#3 | 72, 31, 178 | 63.35% | 19 | 174 |
| cologne1 | stage5/evaluate | -32038056#3 | 110, 50, 44 | 21.57% | 20 | 41 |
| cologne1 | stage5/evaluate | 23429231#1 | 4, 0, 200 | 98.04% | 32 | 200 |
| cologne1 | stage5/evaluate | 27115123#3 | 2, 6, 196 | 96.08% | 14 | 194 |
| cologne1 | stage5/evaluate | 28198821#3 | 142, 35, 27 | 13.24% | 18 | 19 |
| cologne1 | stage6/QL | -32038056#3 | 279, 118, 112 | 22.00% | 22 | 99 |
| cologne1 | stage6/QL | 23429231#1 | 10, 0, 499 | 98.04% | 32 | 499 |
| cologne1 | stage6/QL | 27115123#3 | 5, 15, 489 | 96.07% | 14 | 484 |
| cologne1 | stage6/QL | 28198821#3 | 353, 86, 70 | 13.75% | 18 | 51 |
| ingolstadt1 | stage5/train | 104010354 | 129, 49, 126 | 41.45% | 14 | 113 |
| ingolstadt1 | stage5/train | 164051413 | 104, 200, 0 | 0.00% | 4 | 0 |
| ingolstadt1 | stage5/train | 201963537#1 | 109, 68, 127 | 41.78% | 44 | 113 |
| ingolstadt1 | stage5/evaluate | 104010354 | 117, 33, 54 | 26.47% | 14 | 48 |
| ingolstadt1 | stage5/evaluate | 164051413 | 53, 151, 0 | 0.00% | 4 | 0 |
| ingolstadt1 | stage5/evaluate | 201963537#1 | 64, 73, 67 | 32.84% | 46 | 56 |
| ingolstadt1 | stage6/QL | 104010354 | 295, 85, 131 | 25.64% | 14 | 113 |
| ingolstadt1 | stage6/QL | 164051413 | 132, 379, 0 | 0.00% | 4 | 0 |
| ingolstadt1 | stage6/QL | 201963537#1 | 154, 188, 169 | 33.07% | 47 | 142 |

Ở Cologne1 train, cả bốn nhóm đều từng có queue 5 trong bin 2, cực đại lần
lượt 113/32/14/19; tỷ lệ bão hòa bin 2 là 63,35–86,12%. Khi evaluate Stage 6,
hai nhóm `23429231#1` và `27115123#3` ở bin 2 tới 98,04% và 96,07%.
Đây là mất độ phân giải thực sự. Ingolstadt1 `164051413` chỉ 0–4 xe trong
mọi mẫu, nên bin 2 của chiều đó không xuất hiện; không nhầm với chiều cố định.

Các chiều outgoing có `n0=n`, `n1=n2=0`: Cologne1 n=281/204/509;
Ingolstadt1 n=304/204/511, theo train/evaluate Stage5/evaluate Stage6.

Bảng sau giữ phase khi so sánh. “Va chạm gộp lane” là số vector edge+phase
nhận ≥2 vector lane+phase khác nhau. “Va chạm bin” là số encoded state nhận
≥2 vector edge+phase khác nhau; không phải số xe hay số cặp quan sát.

| Nguồn | Mẫu | Lane vectors | Edge vectors | Encoded states | Va chạm gộp lane | Va chạm bin |
|---|---|---|---|---|---|---|
| cologne1/stage5/train | 281 | 273 | 265 | 57 | 6 | 29 |
| cologne1/stage5/evaluate | 204 | 103 | 81 | 15 | 17 | 10 |
| cologne1/stage6/QL | 509 | 199 | 145 | 15 | 36 | 12 |
| ingolstadt1/stage5/train | 304 | 184 | 154 | 15 | 25 | 12 |
| ingolstadt1/stage5/evaluate | 204 | 109 | 90 | 13 | 17 | 11 |
| ingolstadt1/stage6/QL | 511 | 188 | 145 | 15 | 31 | 12 |

Ví dụ train Cologne1: edge queues `(0,31,13,18)` và `(0,23,13,19)`, cùng
outgoing `(0,0,0,0)` và phase 1, đều encode thành `(0,2,2,2,0,0,0,0,1)`.
`analysis.json → observations → examples/aggregation_examples` giữ các vector
thật cho từng nguồn. Vector lane trong đó nối incoming, outgoing rồi phase,
theo đúng thứ tự lane của `models → encoding`.

## 5. Q-table và coverage state-action thực tế

`analyze_ql.replay_training()` bắt đầu bảng **tạm trong RAM** bằng 0, đọc lại
mỗi action đã lưu, kiểm tra q_before, next_max, target/done và q_after theo
công thức hiện hành; so toàn bộ bảng với checkpoint từng episode rồi model
cuối. Đây là đối chiếu số học, không gọi train() hay cập nhật file model.

| Map | State gặp train | Row cuối | Updates | Cặp đã cập nhật/ô | Coverage | Cặp 1 update | Max updates/cặp | Q âm/0/dương | State thiếu action | State hòa max |
|---|---|---|---|---|---|---|---|---|---|---|
| cologne1 | 57 | 57 | 281 | 175/456 | 38.38% | 107 | 6 | 175/281/0 | 47 | 47 |
| ingolstadt1 | 15 | 15 | 304 | 40/45 | 88.89% | 10 | 35 | 40/5/0 | 3 | 2 |

Hai map không có row chỉ được tạo bởi bootstrap mà chưa xuất hiện ở đầu
quyết định: 57/15 row đều có train visit. Tuy nhiên **row tồn tại không có
nghĩa mọi action trong row đã được học**.

- Cologne1: 107/175 cặp đã thử chỉ cập nhật một lần (61,14%). Histogram số
  update → số cặp: `1→107, 2→44, 3→15, 4→5, 5→3, 6→1`. Cặp nhiều nhất:
  state `(2,2,2,0,0,0,0,0,0)`, action 4, **6 lần**.
- Ingolstadt1: 10/40 cặp một lần (25%); max **35 lần** ở
  `(2,1,0,0,0,0,2)`, action 0. Histogram đầy đủ trong analysis.json.
- 47/57 và 3/15 state chưa thử hết action; 47/57 và 2/15 có hòa cực đại.
  Số state có bất kỳ Q trùng nhau cũng là 47 và 2. Ties kiểm tra **bằng nhau
  chính xác**, không thêm ngưỡng làm các Q gần nhau thành hòa.
- **Trong đúng hai model này**, replay chứng minh mọi ô đã cập nhật đều âm,
  không có ô đã cập nhật còn 0. Vì thế 281/5 ô bằng 0 đúng là chưa được chọn
  trong train. Đây là kết quả đối chiếu log, không phải quy tắc QL nói chung.
  Test mới có ví dụ action đã cập nhật vẫn Q=0 để giữ sự phân biệt này.

`state_actions.csv` lưu đủ 501 ô (456+45), kể cả ô 0 và update=0; không chỉ
xuất cặp được chọn. `states.csv` lưu train visits, visits Stage6, vector số
update, Q và chênh lệch hai Q lớn nhất; state lạ cũng có dòng riêng.

Ví dụ state train hay gặp của Cologne1 `(2,2,2,0; phase0)` có 36 visit,
nhưng mỗi action mới 3–6 update. Ở Ingolstadt1 `(0,1,1; phase0)` có 46 visit,
145 quyết định Stage6, Q=`[-10.933152, -11.001069, -11.149013]`: argmax duy
nhất 0 nhưng khoảng cách đầu chỉ 0,067917. Không có nhiều lần train độc lập
để kết luận khoảng cách này ổn định hoặc có ý nghĩa thống kê. Phần “chưa đủ
phân biệt” được chứng minh chính xác ở các tie; độ tin cậy ở các gap nhỏ
**không xác định từ dữ liệu hiện có**.

## 6. Epsilon-greedy, Q=0 và tie-break: giải thích chính xác Cologne1

`choose_action()` dùng RNG riêng khi epsilon>0; nếu không rơi vào nhánh
ngẫu nhiên thì argmax, hòa chọn cột đầu trong action order. `get_q_values()`
trả vector 0 cho state chưa thấy mà không chèn row lúc evaluate.
`update_q_value()` giữ Q âm; terminal target chỉ r. Alpha=0,1; gamma=0,95,
discount mỗi quyết định. Epsilon theo episode: **1 → 0,525 → 0,05**.

Phát lại `random.Random(7)` liên tục qua 3 episode kiểm tra được chính xác
nhánh ngẫu nhiên. “Random branch” không đồng nghĩa action khác greedy: bốc
ngẫu nhiên có thể trúng cùng action. Evaluate epsilon=0, không có RNG khám phá.

### 6.1 Các quyết định evaluate, theo từng seed

“State lạ” đếm quyết định trên state ngoài model; “Q0 thắng âm” chỉ các row
đã biết có max=0 và ít nhất một Q âm. Tất cả ties trong các trace này đều chọn
một action chưa cập nhật trong train. Cực đại duy nhất chỉ mô tả thứ tự Q,
không chứng minh action tối ưu thật.

| Map/seed | Quyết định | State lạ | Hòa max | Q0 thắng âm | Max duy nhất |
|---|---|---|---|---|---|
| cologne1/42 | 102 | 25 (24.51%) | 72 (70.59%) | 47 | 30 (29.41%) |
| cologne1/12345 | 102 | 20 (19.61%) | 71 (69.61%) | 51 | 31 (30.39%) |
| cologne1/2026 | 101 | 26 (25.74%) | 69 (68.32%) | 43 | 32 (31.68%) |
| cologne1/2027 | 102 | 23 (22.55%) | 70 (68.63%) | 47 | 32 (31.37%) |
| cologne1/2028 | 102 | 18 (17.65%) | 72 (70.59%) | 54 | 30 (29.41%) |
| ingolstadt1/42 | 101 | 0 (0.00%) | 3 (2.97%) | 3 | 98 (97.03%) |
| ingolstadt1/12345 | 103 | 0 (0.00%) | 4 (3.88%) | 4 | 99 (96.12%) |
| ingolstadt1/2026 | 104 | 0 (0.00%) | 2 (1.92%) | 2 | 102 (98.08%) |
| ingolstadt1/2027 | 101 | 0 (0.00%) | 4 (3.96%) | 4 | 97 (96.04%) |
| ingolstadt1/2028 | 102 | 1 (0.98%) | 3 (2.94%) | 2 | 99 (97.06%) |

Stage5 evaluate được phân tích riêng trong `analysis.json → policy`: Cologne1
seed42/12345 có 25/20 state-lạ decisions, 72/71 ties, 30/31 unique max;
Ingolstadt1 0/0 lạ, 3/4 ties, 98/99 unique max. Không cộng chúng thành seed mới.

### 6.2 Chuỗi lựa chọn thực tế Cologne1

Gộp 5 seed có **509 quyết định** trên 15 encoded state khác nhau:
8 state có trong model, 7 state ngoài model. **49/57 state train không được
thăm ở evaluate.** Action 4–7 từng được chọn tổng 119 lần trong train, không
bị loại khỏi tập action.

| Action | State lạ → zero row | Row đã biết hòa max | Max duy nhất | Tổng lựa chọn |
|---|---|---|---|---|
| 0 | 112 | 122 | 0 | 234 |
| 1 | 0 | 115 | 107 | 222 |
| 2 | 0 | 0 | 48 | 48 |
| 3 | 0 | 5 | 0 | 5 |
| 4 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |

Tổng **354/509 = 69,55%** tie gồm 112 zero-row do state lạ và 242 row đã biết
có Q0 thắng Q âm; **155/509 = 30,45%** là max duy nhất (107 action1, 48
action2). Trong 354 tie này có action ≥4 cùng đứng ở cực đại, nhưng cột đầu
thắng là 0,1 hoặc3. Đây là ảnh hưởng tie-break trực tiếp đã kiểm tra trên log.
Không suy rằng đổi tie-break sẽ sửa completion: quỹ đạo sau action mới chưa chạy.

Các ví dụ cụ thể (rút gọn incoming bins + phase; outgoing đều 0):

| State | Q theo action 0…7 | Lựa chọn và bằng chứng |
|---|---|---|
| `(0,0,0,0; p0)` | `[-.38,0,-.1925,0,0,0,0,-.18]` | action1, đầu trong tập hòa `[1,3,4,5,6]`; quyết định đầu cả 5 seed |
| `(0,0,1,1; p1)` | state ngoài model → tám số 0 | action0, fallback; quyết định thứ hai cả 5 seed |
| `(1,2,1,0; p0)` | `[-3.76,-3.565,-2.5375,0,0,0,0,0]` | action3, đầu tập hòa 3–7; xuất hiện một lần mỗi seed |
| `(0,2,2,0; p1)` | `[-5.8001875,0,0,0,0,-5.285,0,0]` | action1 chưa thử, 106 decisions Stage6; train chỉ thử 0 và5, mỗi action một lần |
| `(1,2,2,0; p0)` | `[0,0,-5.13,0,0,0,0,0]` | action0 chưa thử, 104 decisions; train chỉ có một visit/action2 |
| `(2,2,2,0; p0)` | `[-34.606592,-32.894620,-36.946115,-41.286863,-35.106765,-43.901975,-39.644211,-33.454298]` | action1 là max thật, 107 decisions; không phải tie |
| `(0,2,2,2; p1)` | `[-10.172488,-11.123250,-5.867500,-16.420825,-8.891000,-5.955000,-9.413000,-6.837500]` | action2 là max thật, 48 decisions |

Cơ chế quan sát được là: policy khởi đầu 1→0→3; sau đó các state thực gặp
chỉ trả 0/1/2. Nó không cập nhật hay khám phá khi evaluate, nên những zero-row
hoặc Q0 chưa thử vẫn lặp lại như cũ. State chứa phase nên quỹ đạo pha cũng
thay đổi vùng state được thăm. Đây là mô tả trace, không phải chứng minh tập
state đóng dưới mọi nhu cầu hoặc seed khác.

**Không phải model không bao giờ chọn 4–7:** đếm argmax trên toàn model,
chưa chạy mô phỏng giả định, được số row cho action0…7 là
`[22,15,5,3,8,1,2,1]`. Có **12 row chọn 4–7**, tất cả không xuất hiện trong
trace evaluate Stage6. Ví dụ `(2,2,2,2; p3)` chọn action4; `(2,0,2,2; p4)`
chọn7. Vì vậy kết luận đúng là **Q-table + quy tắc chọn + state thực gặp**
tạo ra 0–3, không phải một lỗi ánh xạ cột hay lệnh cấm action 4–7.

Ingolstadt1: 511 decisions, chỉ 1 state-lạ decision (seed2028), 15 tie trên
row đã biết, **495 max duy nhất**. Action4 thắng max thật 90 lần. Toàn model
có greedy row counts `[6,6,3]` theo action `[0,2,4]`. Tie-break/fallback có
ảnh hưởng nhỏ hơn rõ rệt trong trace nhưng chất lượng vẫn chưa đạt baseline.

## 7. Training qua 3 episode: tiến trình và giới hạn diễn giải

SUMO seed23423 và exploration seed7 giữ nguyên; mỗi episode reset mô phỏng,
Q-table và exploration RNG tiếp tục từ episode trước. Vì epsilon và bảng
cùng đổi, chênh lệch kết quả episode không cô lập riêng tác dụng của học Q.

| Map | Ep | ε | Random branch | Σreward | Queue·s | Rows lũy kế | Updates | Cặp lũy kế | Teleport | Inserted | Arrived* | Chưa chèn | Đang chạy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cologne1 | 1 | 1.000 | 93/93 (100.00%) | -5934.117 | 230243 | 36 | 93 | 75 | 23 | 1897 | 1845 | 118 | 52 |
| cologne1 | 2 | 0.525 | 55/94 (58.51%) | -7291.217 | 280246 | 49 | 94 | 137 | 28 | 1650 | 1571 | 365 | 79 |
| cologne1 | 3 | 0.050 | 4/94 (4.26%) | -8960.242 | 346256 | 57 | 94 | 175 | 20 | 1933 | 1817 | 82 | 116 |
| ingolstadt1 | 1 | 1.000 | 104/104 (100.00%) | -1823.933 | 62914 | 11 | 104 | 25 | 6 | 1698 | 1659 | 18 | 39 |
| ingolstadt1 | 2 | 0.525 | 56/100 (56.00%) | -1145.267 | 41343 | 13 | 100 | 32 | 0 | 1689 | 1661 | 27 | 28 |
| ingolstadt1 | 3 | 0.050 | 4/100 (4.00%) | -671.674 | 24386 | 15 | 100 | 40 | 7 | 1647 | 1624 | 69 | 23 |

`Arrived* = inserted − running` từ statistics của full-hour Stage5: có thể
gồm xe từng teleport, không phải số hoàn thành hợp lệ Stage6. Stage5 không có
tripinfo cùng tiêu chí Stage6 nên **completion hợp lệ và throughput hợp lệ
của train không xác định từ dữ liệu hiện có**; CSV để trống, không điền giả.
Đây không phải thiếu dữ liệu bắt buộc để phân tích policy, nên không chạy
mô phỏng bổ sung chỉ để lấp cột.

Phân bố action từng episode, theo thứ tự `[0…7]` / `[0,2,4]`:

| Map | Episode | Action counts | Greedy branches | Greedy ties |
|---|---|---|---|---|
| cologne1 | 1 | 6, 14, 11, 13, 11, 13, 14, 11 | 0 | 0 |
| cologne1 | 2 | 23, 15, 13, 15, 5, 8, 6, 9 | 39 | 31 |
| cologne1 | 3 | 18, 9, 14, 11, 13, 7, 12, 10 | 90 | 32 |
| ingolstadt1 | 1 | 36, 31, 37 | 0 | 0 |
| ingolstadt1 | 2 | 41, 29, 30 | 44 | 4 |
| ingolstadt1 | 3 | 46, 36, 18 | 96 | 4 |

Tổng train Cologne1 `[47,38,38,39,29,28,32,30]`, Ingolstadt1 `[123,96,85]`.
Action ít nhất tương ứng là 5 (28 lần) và4 (85 lần); **không có action hoàn
toàn vắng trong train**. Điểm yếu là coverage **theo state-action**, không
chỉ tần suất action toàn cục. Signals train cấp xanh đủ 20/20 hoặc8/8 movement
trong cả ba episode; policy cuối greedy không giữ được điều đó ở Cologne1.
`training.csv` có thêm số giây xanh thực từng action, không suy 30×counts
vì lượt cuối có thể bị cắt, kể cả chỉ có vàng/đỏ.

- Cologne1 reward xấu dần -5934→-7291→-8960 và queue·s tăng. Rows 36→49→57,
  cặp 75→137→175, không chứng minh policy tốt hơn. Arrived* 1845→1571→1817
  không đơn điệu; growing Q-table không bảo đảm movement coverage khi greedy.
- Ingolstadt1 reward cải thiện -1824→-1145→-672 và queue·s giảm; Arrived*
  1659→1661→1624, inserted 1698→1689→1647, chưa chèn18→27→69,
  teleport6→0→7. Ngay cả số arrival thô cũng không cải thiện cùng chiều reward.
  Không thể tính mức thay đổi completion hợp lệ mà thiếu cohort từng xe.
- Random branch episode cuối chỉ **4 quyết định mỗi map**; 107 cặp Cologne1
  và10 cặp Ingolstadt1 mới được cập nhật một lần. Có bằng chứng lượng khám
  phá thực tế hữu hạn và coverage mỏng, đặc biệt Cologne1. Chưa có đối chứng
  để chứng minh chỉ cần kéo dài exploration/training sẽ khắc phục kết quả.
- **Không có bằng chứng hội tụ**: chỉ 3 episode, alpha cố định, epsilon cùng
  thay đổi, thiếu nhiều chuỗi train độc lập, chưa đo độ ổn định Q/policy.
  Không nối ba điểm thành “đường cong hội tụ”; chưa thể kết luận overfit,
  số episode tối ưu, action thật tối ưu hay Q ước lượng chính xác.

## 8. Action và movement coverage khi evaluate

`audit_signals()` dùng metric offline hiện có `signal_metrics()`, kiểm tra
mapping runtime G theo metadata trên từng giây, controlled links đủ index,
đối chiếu result.json và từng dòng movement_service.csv. Queue/gap/xanh đều
tính từ signals thực; action counts không được dùng thay movement coverage.

### 8.1 Từng map và seed của QL

Counts và giây xanh theo action `[0,1,2,3,4,5,6,7]` (Cologne1), `[0,2,4]`
(Ingolstadt1). “Chưa xong” là đã chèn nhưng chưa arrival; không gộp với chưa
chèn, không bằng tổng yêu cầu chưa hoàn thành hợp lệ vì còn arrived từng teleport.

| Map | Seed | Action counts | Giây xanh thực | Movement có G | Gap lớn nhất s | Chưa chèn | Chưa xong |
|---|---|---|---|---|---|---|---|
| cologne1 | 42 | 49, 44, 8, 1, 0, 0, 0, 0 | 1470, 1310, 240, 30, 0, 0, 0, 0 | 10 | 3600 | 849 | 113 |
| cologne1 | 12345 | 45, 45, 11, 1, 0, 0, 0, 0 | 1350, 1350, 310, 30, 0, 0, 0, 0 | 10 | 3600 | 854 | 118 |
| cologne1 | 2026 | 46, 45, 9, 1, 0, 0, 0, 0 | 1380, 1350, 260, 30, 0, 0, 0, 0 | 10 | 3600 | 851 | 114 |
| cologne1 | 2027 | 48, 43, 10, 1, 0, 0, 0, 0 | 1440, 1290, 270, 30, 0, 0, 0, 0 | 10 | 3600 | 850 | 116 |
| cologne1 | 2028 | 46, 45, 10, 1, 0, 0, 0, 0 | 1380, 1340, 300, 30, 0, 0, 0, 0 | 10 | 3600 | 844 | 113 |
| ingolstadt1 | 42 | 54, 29, 18 | 1614, 870, 540 | 8 | 418 | 94 | 40 |
| ingolstadt1 | 12345 | 58, 26, 19 | 1740, 780, 540 | 8 | 400 | 86 | 37 |
| ingolstadt1 | 2026 | 61, 25, 18 | 1830, 720, 540 | 8 | 476 | 26 | 31 |
| ingolstadt1 | 2027 | 52, 30, 19 | 1560, 870, 570 | 8 | 418 | 89 | 35 |
| ingolstadt1 | 2028 | 59, 27, 16 | 1766, 810, 480 | 8 | 522 | 97 | 40 |

Cả 5 seed Cologne1 thiếu đúng `{5,6,7,8,9,15,16,17,18,19}`, tất cả gap=3600 s.
Ingolstadt1 không có movement không xanh, nhưng vẫn có gap dài 400–522 s.

### 8.2 Đối chiếu mapping Cologne1

| Action | Movement runtime G |
|---|---|
| 0 | 10,11,12,13,14 |
| 1 | 0,1,2,3,4 |
| 2 | 0,1,2,10,11,12 |
| 3 | 3,4,13,14 |
| 4 | 5,6,7,8,9 |
| 5 | 15,16,17,18,19 |
| 6 | 5,6,7,15,16,17 |
| 7 | 8,9,18,19 |

Hợp tập G của action0–3 chỉ gồm 10 movement nêu trên. Do không chọn4–7,
không có runtime G nào mở hai nhóm edge bên dưới. Mapping và signals khớp
cho từng seed; không dựa vào pha gốc có yielding `g/s` vì runtime đóng chúng.

| Movement | Incoming lane | Outgoing lane | Action có thể cấp G |
|---|---|---|---|
| 5 | 23429231#1_0 | 32038056#0_0 | 4, 6 |
| 6 | 23429231#1_0 | 32038051#0_0 | 4, 6 |
| 7 | 23429231#1_1 | 32038051#0_1 | 4, 6 |
| 8 | 23429231#1_1 | -28198821#4_1 | 4, 7 |
| 9 | 23429231#1_1 | 32324544#0_1 | 4, 7 |
| 15 | 27115123#3_0 | -28198821#4_0 | 5, 6 |
| 16 | 27115123#3_0 | 32324544#0_0 | 5, 6 |
| 17 | 27115123#3_1 | 32324544#0_1 | 5, 6 |
| 18 | 27115123#3_1 | 32038056#0_1 | 5, 7 |
| 19 | 27115123#3_1 | 32038051#0_1 | 5, 7 |

Incoming edges bị thiếu G: **23429231#1** (lane0/1), **27115123#3** (lane0/1).
Không có G không đồng nghĩa chắc chắn có xe chờ trong từng giây; log quyết
định cho thấy queues ở hai nhóm thường ở bin2 nhưng không lưu queue theo
movement hay ý định rẽ để chứng minh xe chờ liên tục cả giờ. Gap không xanh
là metric tín hiệu, **không phải waiting time của xe**. Không truy vết nên
không gán toàn bộ 844–854 xe chưa chèn hoặc113–118 xe chưa xong cho các movement này.

### 8.3 Đối chiếu FT/MP theo từng seed

Tất cả baseline cấp G đủ movement, kể cả MP không chọn một vài action do
movement được phục vụ chồng lấp ở action khác. Đây là ví dụ vì sao chỉ đếm
action không đủ. Counts/xanh của đủ 30 lượt nằm trong `runs.csv`; xanh và gap
của đủ 420 dòng controlled link nằm trong `movements.csv`.

| Map | Seed | FT: có G / gap max s | MP: có G / gap max s | QL: có G / gap max s |
|---|---|---|---|---|
| cologne1 | 42 | 20 / 250 | 20 / 770 | 10 / 3600 |
| cologne1 | 12345 | 20 / 250 | 20 / 690 | 10 / 3600 |
| cologne1 | 2026 | 20 / 250 | 20 / 1010 | 10 / 3600 |
| cologne1 | 2027 | 20 / 250 | 20 / 570 | 10 / 3600 |
| cologne1 | 2028 | 20 / 250 | 20 / 970 | 10 / 3600 |
| ingolstadt1 | 42 | 8 / 84 | 8 / 312 | 8 / 418 |
| ingolstadt1 | 12345 | 8 / 84 | 8 / 312 | 8 / 400 |
| ingolstadt1 | 2026 | 8 / 84 | 8 / 388 | 8 / 476 |
| ingolstadt1 | 2027 | 8 / 84 | 8 / 312 | 8 / 418 |
| ingolstadt1 | 2028 | 8 / 84 | 8 / 350 | 8 / 522 |

Ví dụ MP Cologne1 seed2027 không chọn action3/7, nhưng vẫn phục vụ20/20
movement qua các action còn lại. Coverage đủ không đồng nghĩa bảo đảm
max-wait hoặc an toàn tuyệt đối; giữ nguyên cảnh báo/collision Stage6/6.1.

## 9. Reward và metrics: điều gì được khuyến khích, điều gì không được đo?

`SumoEnv.step()` tính:

```text
q(k) = tổng xe dừng trên các incoming boundary lanes duy nhất, cuối giây k
queue_seconds = Σ q(k) × 1 s, gồm xanh/vàng/đỏ/clearance
r_i = −queue_seconds_i / elapsed_i
R_episode = Σ_i r_i
Average Queue Length = Σ_i queue_seconds_i / 3600
```

Reward trực tiếp ưu tiên queue trung bình **trong vùng đo** thấp trên lượt
điều khiển; hậu quả tương lai được Q phản ánh qua bootstrap discount. Nó
không có số hạng trực tiếp cho chưa chèn, queue ngoài boundary lanes,
completion/arrival, throughput, teleport, thời gian không G hoặc fairness
giữa movement. Movement bị bỏ G có thể tạo queue và penalty **gián tiếp**
khi xe ở vùng đo, nhưng không có bộ đếm age/service hay bảo đảm chống đói.
Không nói reward hoàn toàn không phạt ùn, cũng không nói nó đo toàn mạng.

**Thời lượng quan trọng:** cùng action thường30 s, đổi pha40/38 s hoặc thêm
clearance, lượt cuối có thể ngắn. R_episode cộng các trung bình không trọng
số thời gian nên không bằng âm queue·s, cũng không bảo toàn thứ hạng của
queue mean theo giờ. Ví dụ tính tay, không phải mô phỏng: queue10 trong30 s
và queue20 trong40 s cho R=-30, queue·s=1100 và mean15,714; chia khoảng queue20
thành hai lượt20 s cho R=-50 dù cùng1100 queue·s. Trong hệ thống thực các
lượt phải tuân thủ timing, ví dụ này chỉ minh họa đặc tính phép cộng.

Gamma0,95 discount theo **quyết định**: cùng số bước bootstrap có thể ứng với
số giây khác nhau. Sau một quyết định30 s hay40 s đều nhân0,95; không diễn
giải trực tiếp như cùng discount theo giây, không tự chuyển thành semi-MDP.
Không phát hiện lỗi terminal hay sai cột action trong replay; điều đó không
chứng minh giá trị Q đã học đúng hàm giá trị tối ưu.

So sánh **chỉ evaluate Stage6**, trung bình5 lượt/map/algorithm. R* dưới đây
là giá trị **tái tính offline cùng công thức** cho mọi thuật toán, không phải
reward dùng để train FT/MP và không dùng xếp hạng thành công.

| Map | Alg | R* | Mean queue xe | Completion hợp lệ | Throughput/giờ | Teleport events | Movement thiếu G | Chưa chèn |
|---|---|---|---|---|---|---|---|---|
| cologne1 | FT | -6696.15 | 73.87 | 93.86% | 1891.2 | 0.0 | 0 | 53.8 |
| cologne1 | MP | -2567.13 | 28.39 | 97.66% | 1967.8 | 10.6 | 0 | 6.2 |
| cologne1 | QL | -4830.30 | 47.65 | 49.20% | 991.4 | 65.8 | 10 | 849.6 |
| ingolstadt1 | FT | -873.63 | 9.22 | 97.95% | 1680.8 | 0.2 | 0 | 7.8 |
| ingolstadt1 | MP | -534.48 | 5.64 | 97.28% | 1669.4 | 0.6 | 0 | 19.2 |
| ingolstadt1 | QL | -833.85 | 8.30 | 93.10% | 1597.6 | 3.4 | 0 | 78.4 |

Cologne1 QL có R* ít âm và queue nhỏ hơn FT nhưng completion49,20% so với
93,86%, teleport65,8 so với0. MP có queue/reward tốt hơn QL cùng completion
cao hơn. Ingolstadt1 QL queue8,30 thấp hơn FT9,22 nhưng completion93,10% thấp
hơn97,95%, waiting52,51 s cao hơn32,01 s. Vì vậy reward tốt hơn trên vùng đo
không đủ để gọi policy tốt, và QL không phải kém baseline ở **mọi** metric.

Định nghĩa giữ nguyên Stage6: valid completed là arrived, không vaporized và
chưa từng teleport. Completion=N/requested; throughput=N×3600/horizon (xe/giờ).
Average Waiting Time/Travel Time/Time Loss chỉ trên **cùng cohort N**,
đơn vị s/xe; travel=arrival−actual depart, waitingTime/timeLoss theo SUMO,
không cộng departDelay. Xe chưa chèn/chưa xong không được gán thời gian0
hay đưa vào trung bình chuyến hoàn thành. Queue mean tính đủ3600 mẫu cuối
giây, đơn vị xe, không phải xe/lane. Teleports đếm sự kiện start, có thể một
xe nhiều lần; không đồng nhất với số ID từng teleport. Các metric cohort
này đã được kiểm chứng raw ở Stage6.1 và test offline cũ.

Một giả thuyết phù hợp là queue boundary bị giới hạn trong khi xe dồn ngoài
vùng đo, hoặc teleport làm thay đổi queue, nên reward không phản ánh nhu cầu
bị trì hoãn. Dữ liệu xác nhận nhiều xe chưa chèn và reward không chứa chúng;
chưa truy vết được phần giảm reward do từng cơ chế. Không gọi việc “agent
cố tình khai thác teleport” là nguyên nhân đã được chứng minh.

## 10. Vì sao cùng cấu hình không cho kết quả giống nhau trên hai map?

| Chỉ số | Cologne1 | Ingolstadt1 |
|---|---|---|
| Action IDs | 0–7 | 0,2,4 |
| Incoming/outgoing groups | 4/4 | 3/3 |
| State lý thuyết | 52.488 | 2.187 |
| State train thực gặp / row model | 57/57 | 15/15 |
| Updates / cặp đã học | 281 / 175 | 304 / 40 |
| Coverage trên state train thực gặp | 38,38% | 88,89% |
| Cặp chỉ cập nhật1 lần | 107/175 | 10/40 |
| Incoming bin2 khi train (theo edge) | 81,85%;86,12%;74,73%;63,35% | 41,45%;0%;41,78% |
| Outgoing cố định0 khi quyết định | 4 chiều | 3 chiều |
| State Stage6 biết / lạ (distinct) | 8 / 7 | 14 / 1 |
| Hòa max / tổng decisions Stage6 | 354/509 | 16/511 |
| Movement thiếu G mỗi seed | 10/20 | 0/8 |
| Teleport events trung bình Stage6 QL | 65,8 | 3,4 |
| Completion Stage6 QL | 49,20% | 93,10% |
| Throughput Stage6 QL / giờ | 991,4 | 1597,6 |
| Nhu cầu / giờ | 2015 | 1716 |

Cùng3 episode nhưng Cologne1 có8 action/57 state thực, so với3 action/15 state:
281 update chia trên456 ô khác304 trên45 ô. Ngưỡng chung bão hòa nhiều hơn
ở Cologne1; cùng pha là một chiều nên8 phase cũng phân mảnh dữ liệu hơn3.
Phân bố nhu cầu, số lane/movement, hình học và số giây chuyển tiếp khác nhau;
reward phản ánh queue khác nhau, không chuẩn hóa trực tiếp theo nhu cầu.
Không so độ lớn Q/reward giữa hai map như thang chất lượng chung.

Route demand khác nhau là dữ kiện, nhưng chưa tính nhu cầu theo từng movement
và từng khoảng giờ trong nhiệm vụ này. Không suy “cùng seed cùng số lần train”
đồng nghĩa bài toán học tương đương, hoặc nhiều state hơn là nguyên nhân duy
nhất. Ingolstadt1 coverage cao vẫn có aliasing và chất lượng kém ở nhiều metric.

## 11. Những vấn đề đã có bằng chứng

| Kết luận giới hạn | Bằng chứng trực tiếp | Điều chưa suy ra |
|---|---|---|
| Bin và gộp edge mất thông tin trong log | queue_bins.csv; analysis.observations và các vector thật | Không định lượng mức giảm throughput do encoding |
| Outgoing không phân biệt mẫu ở thời điểm quyết định này | Toàn bộ3 nguồn, tất cả outgoing0 | Không kết luận downstream luôn trống/không cần quan sát |
| Coverage mỏng, đặc biệt Cologne1 | Replay585 updates,6 checkpoint;175/456 ô,107 cặp1 lần | Không biết bao nhiêu episode là đủ |
| Q0/unseen/tie-break quyết định354 lượt Cologne1 | policy_decisions.csv, selection_reasons;112 lạ+242 hòa biết | Không chứng minh thay tie-break sẽ chữa policy |
| 155 lượt Cologne1 còn lại có argmax thật1/2 | Q trong model khớp từng row evaluate | Không coi Q đúng thứ tự là giá trị tối ưu thật |
| Action0–3 không mở10 movement còn lại | Runtime mapping +18.000 mẫu signals QL Cologne1 | Không chứng minh xe chờ liên tục hoặc quy tất cả chưa xong vào chúng |
| Reward không đo trực tiếp service toàn mạng | Công thức SumoEnv.step; cohort/chưa chèn trong result | Không quy phần trăm lỗi cho reward, không suy agent cố ý lợi dụng |
| Reward train tăng không bảo đảm arrival tăng | Ingolstadt1 ep1–3 | Chưa xác định completion hợp lệ lúc train |
| Không phát hiện sai công thức update/action column trong bằng chứng lưu | Phát lại mỗi Q trước/sau, RNG và checkpoint; test Q cũ | Không chứng nhận toàn bộ hành vi SUMO an toàn/tái lập |

## 12. Các giả thuyết và điều chưa xác định

- Encoding thô/gộp lane làm thiếu thông tin cần để chọn pha tốt: đã chứng
  minh mất thông tin, **tác động chất lượng** cần ablation riêng.
- Ba episode/lịch epsilon giảm nhanh làm Q chưa đủ tốt: coverage và4 lượt
  random cuối là dữ kiện; “train dài sẽ thắng” chưa được chứng minh.
- Tie-break/zero initialization duy trì vùng pha hẹp: đã thấy trực tiếp trên
  trace, nhưng hiệu quả thực của random tie-break hay cách xử lý unknown
  khác chưa đo. Không được đổi policy rồi gọi suy đoán là kết quả cũ.
- Reward boundary tạo thiên lệch với unmet demand/fairness/teleport: cấu
  trúc mục tiêu có giới hạn rõ; chưa tách được hiệu ứng của từng thành phần.
- Khác biệt nhu cầu theo hướng, saturation flow, khả năng lan truyền ùn ngoài
  vùng đo hoặc thiếu Markov góp phần vào kết quả: chưa có log đủ theo
  movement/xe để kết luận. Không khẳng định xe của tất cả movement thiếu G
  bị kẹt toàn giờ hoặc mọi xe chưa chèn đều thuộc chúng.
- Overfitting, hội tụ, Q ranking ổn định, số episode cần thiết và trọng số
  nguyên nhân không xác định từ3 episode/một exploration seed. Cùng seed
  SUMO cũng chưa bảo đảm cùng quỹ đạo theo Stage6.1.

## 13. Ba ưu tiên Stage 7 — chỉ đề xuất, chưa triển khai

1. **Bổ sung đo lường và giữ kiểm soát thực nghiệm trước khi sửa học.** Giữ
   model hiện tại làm mốc; giải quyết/định lượng biến thiên cùng seed theo
   Stage6.1. Log queue từng lane/edge theo giây, ý định/nhu cầu từng movement,
   tuổi không được phục vụ, xe pending/chưa chèn và completion hợp lệ cả lúc
   train bằng cùng tiêu chí. Theo dõi visits/updates/TD error, policy và Q
   qua checkpoint. Kiểm chứng bằng partition xe và raw CSV/XML, không bỏ
   teleport/lượt xấu. Thiếu đo lường không phải lý do để sửa dữ liệu SUMO.
2. **Kiểm chứng giả thuyết coverage và lựa chọn trước.** Trên cùng encoding,
   reward, timing và nhu cầu, thử ngân sách train/exploration có nhiều seed
   độc lập; lịch epsilon dựa trên mức bao phủ cần được thiết kế rõ. Ablation
   riêng tie-break và xử lý state lạ/Q chưa thử, luôn giữ action xanh hợp lệ.
   Không thay đồng thời tất cả để mất khả năng giải thích. Tiêu chí: coverage
   theo state thực, ranking Q/policy ổn định, movement G và completion/teleport
   khi evaluate đóng băng; đủ toàn bộ action ở train chưa phải đủ coverage.
   Rủi ro: khám phá nhiều làm training xấu hơn, train dài vẫn có thể học sai
   mục tiêu; random tie-break không bảo đảm công bằng hay service tốt.
3. **Kiểm chứng quan sát và mục tiêu bằng ablation có kiểm soát.** Cân nhắc
   bin bổ sung/cap theo sức chứa cho edge đông, hoặc giữ queue lane để tránh
   mất phân bố; chỉ thêm thời gian/age/movement khi có phép đo đáng tin. Cân
   nhắc reward theo queue·s/chi phí thời gian, penalty unmet demand/tuổi chờ
   hay fairness, và discount phù hợp thời lượng. Thử từng thay đổi riêng,
   giữ cơ chế pha an toàn cho mọi thuật toán và đánh giá trên nhu cầu/seed
   tách training. Rủi ro: state lớn làm coverage loãng; penalty cân sai làm
   ưu tiên lane ít nhu cầu/chuyển pha nhiều; không phạt thiếu G khi không có
   nhu cầu một cách máy móc. Kiểm chứng bằng completion/throughput, waiting
   cohort, queue trong/ngoài vùng, movement service, teleport và safety.

Không ưu tiên tự xóa chiều outgoing chỉ vì0 trong tập hiện tại: nó không
tách state ở đây nên xóa nó cũng không tự giảm số state **thực đã thăm**.
Không có đề xuất nào ở đây đã được triển khai hay chứng minh cải thiện.

## 14. Tệp bàn giao, cách chạy và vị trí thuyết trình

| Tệp | Nội dung / hàm phân tích |
|---|---|
| `../../analyze_ql.py` | Script offline stdlib + metric offline có sẵn, không import traci/sumo_env/q_learning/evaluate |
| `tests/test_analyze_ql.py` | Phân biệt Q0 đã/chưa thử; alias lane/bin; gap đầu/cuối; signal/action sai; thiếu/CSV rỗng |
| `analysis.json` | SHA164 inputs; models, observations/alias examples, policy theo seed, comparisons |
| `queue_bins.csv` | 42 dòng edge/nguồn: bins/tỷ lệ/min/max/chiều cố định |
| `state_actions.csv` | 501 ô Q và số update thật, kể cả chưa được chọn |
| `states.csv` | 80 dòng state trong model và state lạ Stage6, tần suất, Q, action/max/tie/gap |
| `policy_decisions.csv` | 1.428 quyết định của4 lượt Stage5 +10 QL Stage6: Q, tập argmax, lý do và số update action được chọn |
| `training.csv` | 6 episode; RNG branch, reward/queue, updates, action/xanh/movement, số xe |
| `runs.csv` | Đủ30 lượt Stage6; counts/xanh, metrics, reconstructed_reward và policy counts |
| `movements.csv` | 420 dòng controlled links: lane/edge/via, action có G, xanh và gap thực |
| `verification.json`, `tests_offline.log` | Bằng chứng bất biến/test/tái tạo bảng, không khởi chạy simulator |

Chỉ cần Python `.venv` hiện có. Script phân tích toàn bộ trong RAM trước khi
xuất, lỗi khi thiếu file/CSV rỗng, và từ chối thư mục đầu ra đã tồn tại.
`assert` đối chiếu dữ liệu dùng Python bình thường, **không dùng `python -O`**.
Ví dụ tái tạo vào một thư mục mới:

```powershell
.\.venv\Scripts\python.exe tools/analyze_ql.py --output reports/stage6_2_repeat
```

Lệnh này chỉ sinh JSON/CSV, README là phần diễn giải do tác giả viết. Script
không import module chạy SUMO; `signal_metrics` trong evaluation_metrics chỉ
xử lý list mẫu đã lưu. Không cần dependency mới hoặc dashboard/biểu đồ.

Chạy riêng các test phân tích offline từ thư mục gốc.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_analyze_ql.py
```

Lượt kiểm tra offline lịch sử loại 11 ca khởi chạy SUMO hoặc train.
Các ca đó gồm `test_real_train_three_episodes_save_load_and_read_only_evaluation`,
`test_episode_error_closes_real_sumo`,
`test_terminal_transition_is_not_encoded_and_evaluation_does_not_insert` (2 cases),
`test_unseen_greedy_episode_keeps_empty_table`,
`test_runner_closes_real_sumo_on_controller_error_and_refuses_overwrite`,
`test_common_runner_short_real_sumo` (3 algorithms),
`test_tripinfo_output_matches_short_85_second_control`,
`test_probe_records_real_clock_and_does_not_overwrite`.
Không tuyên bố toàn bộ147 test cũ đã chạy lại: baseline offline136 passed,
11 deselected. **Kiểm tra cuối: 140 passed, 11 deselected trong3,16 s**
(136 test cũ offline +4 test mới); không có failure. Xem
`verification.json` và `tests_offline.log`.

Chạy script qua CLI vào thư mục tạm mới tái tạo **7 CSV và1 JSON khớp từng
byte** với bản bàn giao; gọi lại cùng output bị FileExistsError như thiết kế.
Đã đối chiếu bước đầu policy1→0→3 trên cả5 seed Cologne1. SHA của cả1.320
file cũ giữ nguyên; `git diff --exit-code` đạt. SHA tổng manifest trước=sau:
`1624648c373233f87ebbd2d54c0320e291feec03dbbe123b948c80f68a6f3e7b`.

| Kiến thức khi thuyết trình | File / hàm |
|---|---|
| Input queue lane và phase | `sumo_env.py: SumoEnv.get_state()` |
| Nhóm edge/ngưỡng/state tuple | `q_learning.py: build_encoding(), encode_state()` |
| Chọn action, zero-row và tie | `q_learning.py: get_q_values(), choose_action(), episode_epsilon()` |
| Công thức Q/terminal | `q_learning.py: update_q_value()` |
| SUMO nhận lệnh, chuyển pha an toàn | `sumo_env.py: SumoEnv.step(), _run_signal(), _junction_vehicles()` |
| Reward | `sumo_env.py: SumoEnv.step()` |
| Evaluate không học | `evaluate.py: greedy_steps(), run_evaluation()` |
| Phép đo/chọn cohort xe | `evaluation_metrics.py: trip_metrics(), signal_metrics()` |
| Bằng chứng số lần học và RNG | `analyze_ql.py: replay_training()`; state_actions/training CSV |
| Bằng chứng mất thông tin | `analyze_ql.py: observation_analysis()`; queue_bins.csv và analysis.observations |
| Bằng chứng vì sao0–3 | `analyze_ql.py: greedy_evidence(), evaluation_policy(), policy_summary()`; states/policy_decisions CSV |
| Bằng chứng10 movement không G | `analyze_ql.py: audit_signals(), analyze_evaluation_run()`; movements/runs CSV |

Chưa đo can thiệp nhân quả, hội tụ, nhu cầu theo từng movement, quỹ đạo của
mọi xe chưa chèn/chưa xong, hoặc sửa tính không xác định của SUMO. Kết luận
chất lượng bị giới hạn ở model và các lượt đã lưu; không có cải tiến thuật
toán nào được thực hiện trong Stage6.2.
