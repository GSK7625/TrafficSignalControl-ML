# Giai đoạn 5 — Tabular Q-Learning — 15/11/2026

Đã triển khai Q-Learning dạng bảng, dùng nguyên `SumoEnv` của FT/MP. Chạy thực
**3 episode train/map và 2 episode evaluate/map**, mỗi episode đủ 3600 s:
6 train + 4 evaluate. Đây là kiểm tra pipeline, **chưa hội tụ, chưa có bằng
chứng QL tốt hơn baseline**. Không xóa lượt kém; Cologne1 train reward xấu đi.

## 1. Phạm vi và khảo sát trước khi sửa

Đã đọc AGENTS.md, sumo_env.py, ft.py, mp.py, test_sumo_env.py, test_mp.py và
báo cáo giai đoạn 3/4; đối chiếu net/config/route và mapping từ môi trường.
Git ban đầu sạch, nhánh main; baseline **49 passed**. Chỉ thêm q_learning.py,
test_q_learning.py và reports/stage5/. Không sửa môi trường/FT/MP/test cũ,
data gốc, dependency; không commit/push. `source_hashes.json` đối chiếu SHA-256
8 tệp data với giai đoạn 4: không thay đổi.

Python .venv 3.11.9, SUMO 1.28.0, chạy headless. Cologne1: TLS
`cluster_357187_359543`, 20 controlled links, 8 làn vào/8 làn ra, 2015 trip,
horizon [25200,28800). Ingolstadt1: TLS
`cluster_274083968_cluster_1200364014_1200364088`, 8 links, 7 làn vào/6 làn ra,
1716 trip, horizon [57600,61200). SumoEnv.reset đối chiếu controlled links thật
qua TraCI. Không dùng cologne3, neural network, PyTorch hoặc framework RL.

## 2. State và kiểm tra ngưỡng

`get_state()` trả incoming/outgoing theo **lane order của môi trường**, phase
là **ID pha gốc**. `build_encoding()` lấy membership lane→edge từ net.xml,
chỉ lấy lane biên được môi trường đo, từ chối lane trùng. Sắp nhóm theo edge
ID; trong nhóm mỗi lane đóng góp đúng một lần, không cộng theo controlled link.
`encode_state()` cộng queue nguyên trước, rồi rời rạc hóa:

```text
0 xe → 0; 1–4 xe → 1; ≥5 xe → 2
state = (incoming bins..., outgoing bins..., original_phase_id)
```

Thứ tự nhóm đã lưu trong từng model:

| Map | Incoming edges | Outgoing edges |
|---|---|---|
| Cologne1 | -32038056#3, 23429231#1, 27115123#3, 28198821#3 | -28198821#4, 32038051#0, 32038056#0, 32324544#0 |
| Ingolstadt1 | 104010354, 164051413, 201963537#1 | -164051413, 104010475#0, 124812857#0 |

Cologne1 có tuple 9 phần tử; Ingolstadt1 có 7. Ví dụ Cologne1 lane đầu cùng
edge có 2 và 3 xe: tổng 5 → mức 2, không mã hóa riêng rồi cộng các mức.
Không đọc route/ý định xe hoặc dữ liệu tương lai. Terminal có thể phase=None
khi horizon cắt vàng/đỏ: runner và update **không encode/bootstrap terminal**.

Kiểm tra hàng đợi thực tại các thời điểm quyết định, gộp 3 episode train:

| Map / incoming edge | Số mẫu mức 0 / 1 / 2 | Queue lớn nhất (xe) |
|---|---|---:|
| Cologne1 / -32038056#3 | 29 / 22 / 230 | 113 |
| Cologne1 / 23429231#1 | 37 / 2 / 242 | 32 |
| Cologne1 / 27115123#3 | 45 / 26 / 210 | 14 |
| Cologne1 / 28198821#3 | 72 / 31 / 178 | 19 |
| Ingolstadt1 / 104010354 | 129 / 49 / 126 | 14 |
| Ingolstadt1 / 164051413 | 104 / 200 / 0 | 4 |
| Ingolstadt1 / 201963537#1 | 109 / 68 / 127 | 44 |

Ngưỡng đủ để kiểm tra rời rạc hóa, nhưng **rất thô**: Cologne1 nhiều mẫu bão
hòa ở mức 2, 5 xe và 113 xe có cùng mã. Edge 164051413 chưa tới mức 2 trong
các mẫu này. Mọi outgoing queue bằng 0 tại mọi quyết định train/evaluate;
code vẫn đo/mã hóa đủ, test dùng queue ra khác 0. `analysis.json` có phân bố
từng edge theo train/evaluate; chưa thay ngưỡng để làm đẹp kết quả.

Gộp theo edge mất phân bố lane/hướng; không có thời gian còn lại trong state.
Vì vậy state quan sát không được chứng minh là Markov đầy đủ. State-space
lý thuyết: Cologne1 3^8×8=52488, Ingolstadt1 3^6×3=2187; chỉ học 57/15 state.

## 3. Action, reward và công thức

`env.actions`: Cologne1 **(0,1,2,3,4,5,6,7)**; Ingolstadt1 **(0,2,4)**.
`Q[state]` là list giá trị theo đúng thứ tự đó. Chọn cột argmax rồi trả action
ID; cập nhật dùng `actions.index(action)`. Không nhầm action 4 với cột 4.

`choose_action()` dùng epsilon-greedy với `random.Random(7)` riêng cho khám phá.
Hòa lấy cột đầu tiên trong action order; giữ Q âm. Q=0 ban đầu khiến action
chưa thử có thể thắng action đã học Q âm; đây là khởi tạo QL, không phải MP
hoặc heuristic chống đói. Giữ một exploration RNG xuyên các episode; reset
SUMO độc lập mỗi episode bằng seed 23423. Không trộn hai RNG.

`episode_epsilon()` giảm tuyến tính theo episode từ start tới minimum. Với
3 episode: **1.0 → 0.525 → 0.05** (CSV cuối có sai số biểu diễn float nhỏ).
Với đúng 1 episode, dùng epsilon_start. alpha=0.1, gamma=0.95; alpha thuộc
(0,1], gamma thuộc [0,1], epsilon thuộc [0,1]. Lịch ba episode này chỉ để kiểm
tra thay đổi bảng, giảm quá nhanh để coi là lịch huấn luyện đã tối ưu.

`update_q_value()` thực hiện online sau mỗi env.step:

```python
target = reward
if not done:
    target += gamma * max(Q[next_state])
Q[state][column] += alpha * (target - Q[state][column])
```

Ví dụ test tay: Q(s,2)=2, r=-5, max Q(s')=6, alpha=.1, gamma=.95 → **1.87**;
các cột khác giữ nguyên. Terminal target chỉ r dù Q(s') lớn. Q có thể âm.
State trước action và state sau step được giữ riêng trong runner.

Gamma ở đây **discount theo mỗi quyết định, không theo giây**. Quyết định
cùng action dài 30 s; đổi action thêm vàng/đỏ/clearance; call cuối có thể ngắn.
Không dùng semi-MDP discount và không đổi reward để bù thời lượng.

Reward lấy trực tiếp từ SumoEnv: **-queue_seconds/elapsed**, queue tổng trên
incoming lane duy nhất lấy cuối từng bước 1 s, gồm toàn thời gian vàng/đỏ/xanh.
Tổng reward là tổng các queue trung bình âm theo call; **không bằng âm tổng
queue-seconds**, không phải waiting time của mỗi xe.

Thực thi duy nhất qua `env.step(action)`: runtime chỉ G/r như giai đoạn 3;
đổi pha đóng mọi G bằng vàng (8 s Cologne1 / 6 s Ingolstadt1), toàn đỏ tối
thiểu 2 s và chờ hết internal/đuôi xe, rồi yêu cầu đủ 30 s xanh. Cùng action
giữ xanh liên tục. Horizon có thể cắt bất kỳ stage cuối; done theo end cấu
hình, không kết thúc sớm khi ít xe. Không đổi timing/chương trình SUMO.

## 4. Train, evaluate và JSON

`train()` tạo bảng mới, reset SUMO từng episode, ghi CSV và lưu checkpoint
sau mỗi episode. `run_episode()` chứa vòng state→encode→epsilon-greedy→step
→next_state/reward/done→online update; luôn close trong finally, kể cả lỗi.

`save_q_table()` ghi JSON gồm metadata và rows `{state: [...], values: [...]}`;
khi tải, list state thành tuple. Không dùng pickle/eval. Metadata lưu map,
action order, encoding và lane/edge order, alpha/gamma, discount_unit, lịch
epsilon, số episode đã train, SUMO seed, exploration seed, phiên bản SUMO,
horizon/timing/runtime signals và hash data. `load_q_table()` kiểm tra đúng
map/actions/encoding/timing/data, state rời rạc, độ dài và Q hữu hạn.

Model cuối:

- `train/cologne1/cologne1_q_table.json`: 57 state, 175 cặp state-action đã cập
  nhật, tổng 281 update; 175 Q âm và 281 ô còn 0.
- `train/ingolstadt1/ingolstadt1_q_table.json`: 15 state, 40 cặp đã cập nhật,
  tổng 304 update; 40 Q âm và 5 ô còn 0.

Checkpoint `train/<map>/episode_00N/q_table.json` giữ từng giai đoạn; model
cuối giống checkpoint episode 3, SHA-256 ở analysis.json. Đường dẫn output
và file model đã tồn tại đều bị từ chối; dùng tên mới khi chạy lại. Không có
resume hay tùy chọn ghi đè. Không áp model Cologne1 lên Ingolstadt1.

`evaluate()` load model độc lập, epsilon=0, không RNG khám phá, không giảm
epsilon, không gọi update và không chèn state mới. State lạ đọc zero-row tạm,
greedy hòa chọn action đầu. Test giữ snapshot bảng trong RAM và bytes model
để kiểm tra bất biến. Metadata/result tách seed train 23423 và seed evaluate
42/12345; exploration_seed trong lượt evaluate để trống, giữ seed train riêng.

`unseen_decisions` là số quyết định chưa có row trong bảng **lúc chọn action**.
Ở evaluate, đây là số quyết định ngoài tập state của model. Ở train, successor
đã được tạo khi bootstrap từ bước trước nên số này **không phải** số state mới
hoặc coverage học tập; dùng q_states/cặp đã cập nhật trong analysis.json.

## 5. Kết quả thực tế

Các lượt train đều SUMO seed 23423, exploration seed 7; evaluate không dùng
seed khám phá. Tất cả đủ 3600 s, process thoát mã 0, **0 collision, 0 emergency
braking, 0 emergency stop**. Vẫn có **224 teleport** trong 10 lượt.

| Map | Chế độ / episode hoặc seed | Epsilon | Tổng reward | Queue-seconds (xe·s) | State Q | Calls | Teleport |
|---|---|---:|---:|---:|---:|---:|---:|
| Cologne1 | Train 1 | 1 | -5934.117 | 230243 | 36 | 93 | 23 |
| Cologne1 | Train 2 | .525 | -7291.217 | 280246 | 49 | 94 | 28 |
| Cologne1 | Train 3 | .05 | -8960.242 | 346256 | 57 | 94 | 20 |
| Ingolstadt1 | Train 1 | 1 | -1823.933 | 62914 | 11 | 104 | 6 |
| Ingolstadt1 | Train 2 | .525 | -1145.267 | 41343 | 13 | 100 | 0 |
| Ingolstadt1 | Train 3 | .05 | -671.674 | 24386 | 15 | 100 | 7 |
| Cologne1 | Evaluate seed 42 | 0 | -4829.067 | 171024 | 57 | 102 | 67 |
| Cologne1 | Evaluate seed 12345 | 0 | -4838.867 | 171634 | 57 | 102 | 68 |
| Ingolstadt1 | Evaluate seed 42 | 0 | -869.877 | 31328 | 15 | 101 | 3 |
| Ingolstadt1 | Evaluate seed 12345 | 0 | -842.905 | 30224 | 15 | 103 | 2 |

Cologne1 evaluate chỉ chọn action 0–3; **10/20 movement không có G trong cả
giờ**. Có 25/102 và 20/102 quyết định gặp state ngoài bảng. inserted chỉ
1166/1161 trên loaded=2015, running=113/118, waiting chưa chèn=849/854.
Queue đo thấp không thể coi là thành công khi xe bị chặn ngoài vùng đo và
teleport nhiều. Không thêm fairness/MP để che chính sách chưa học tốt.

Ingolstadt1 evaluate còn 3/2 teleport; inserted=1622/1630, running=40/37,
waiting chưa chèn=94/86. Seed 42/12345 thực sự thay đổi queue, action trace
và số xe ngay cả khi policy greedy không có RNG. Nhu cầu trip gốc vẫn cố định;
đây không phải hai bộ nhu cầu độc lập hay bằng chứng ý nghĩa thống kê.

SUMO giữ nguyên collision.check-junctions=true, collision.action=teleport,
step-length=1 s, ngưỡng teleport mặc định 300 s. Thống kê phân loại teleport
trong result.json/XML; không trace từng xe để khẳng định một nguyên nhân duy
nhất. Sáu warning thiếu vàng Cologne1/lượt là khi nạp chương trình gốc như
giai đoạn 3/4; CSV xác nhận runtime đổi pha có vàng/đỏ. Không ẩn cảnh báo.

## 6. Metric và tệp đầu ra

Mỗi lượt có metadata.json, actions.csv, signals.csv, result.json, command.txt,
sumo.log, warnings.log, statistics.xml, collisions.xml; train thêm checkpoint.
Signals giữ schema như FT/MP và từng mẫu 1 s. Actions giữ trường chung về
action/reward/queue/timing, thêm state rời rạc, epsilon, q_before/q_after và
next_max để kiểm tra update. training.csv/evaluation.csv và summary.csv lưu
đúng 10 lượt, không bỏ lượt xấu. analysis.json lưu phân bố bin, hash model/CSV,
coverage action/movement và số lần cập nhật.

Queue-seconds đơn vị xe·s, queue trung bình theo giờ = queue_seconds/3600,
đơn vị xe. Chỉ đo incoming boundary lanes duy nhất, kể cả xe chưa kết thúc
chuyến đang trên lane đó; không gồm xe chưa chèn, internal, hoặc ùn ở edge
trước. loaded/inserted/running/waiting được giữ riêng. Teleport đếm start,
không cộng end thành sự kiện mới; collision/teleport đối chiếu TraCI và XML.

Chưa có phép đo chung **Average Waiting Time** (s/xe), **Travel Time** (s/chuyến),
**Time Loss** (s/xe) và **Throughput** (xe hoàn thành/giờ) cho cả FT/MP/QL.
Không suy các đại lượng đó từ reward hay chỉ mẫu xe đã hoàn thành; chưa có
dữ liệu chuyến để xử lý nhất quán xe chưa đến đích. Cần bổ sung chung và chạy
lại các baseline trước khi so các metric này. Không so reward lúc train với
kết quả evaluate FT/MP, không khẳng định QL phải thắng.

## 7. Kiểm thử và lệnh chạy

`.\.venv\Scripts\python.exe -m pytest -q` → **99 passed**, không skip:
49 test cũ + 50 trường hợp QL, gồm 10 kiểm tra lại trace đủ giờ và 2 replay
toàn bộ chuỗi train/RNG/checkpoint. Test kiểm tra mã hóa đúng thứ tự/ngưỡng,
cộng trước khi bin, lane duy nhất, action ID thật, init độc lập, epsilon=1/0,
hòa/Q âm, công thức tay và terminal, alpha/gamma, save/load/sai map/cấu hình/
row hỏng/ghi đè, evaluation bất biến, kết thúc giữa vàng/đỏ và close khi lỗi.

Test ban đầu dừng vì chưa có module QL; sau triển khai core 79 passed, thêm
các probe terminal/read-only/validation 87 passed; sau 6 trace train 95 passed;
sau đủ evaluate 99 passed. Pytest có các lượt SUMO ngắn trong thư mục tạm
(test rút horizon chỉ ở instance test, không sửa config/data); chúng không
phải kết quả train/evaluate đủ giờ trong bảng. Mỗi full suite hiện chạy thêm
7 episode QL ngắn thành công, 1 probe QL cố ý lỗi trước step; test MP cũ còn
có một probe controller lỗi. Không chạy train dài.

Rà soát độc lập theo skill requesting-code-review không phát hiện Critical/
Important. Reviewer chạy 44 test QL không khởi chạy SUMO (5 deselected ở lượt
review); chỉ ra cách diễn giải unseen_decisions đã làm rõ ở mục 4. Reviewer
không chạy mô phỏng hoặc đánh giá chất lượng policy; kiểm tra trace thực và
full suite do tác giả thực hiện.

Lệnh đã chạy cho cả hai map (output chưa tồn tại):

```powershell
foreach ($mapName in @('cologne1','ingolstadt1')) {
    .\.venv\Scripts\python.exe q_learning.py train $mapName --episodes 3 --seed 23423 --exploration-seed 7 --output "reports/stage5/train/$mapName"
    foreach ($runSeed in @(42,12345)) {
        .\.venv\Scripts\python.exe q_learning.py evaluate $mapName --model "reports/stage5/train/$mapName/${mapName}_q_table.json" --seed $runSeed --output "reports/stage5/evaluate/$mapName/seed_$runSeed"
    }
}
.\.venv\Scripts\python.exe -m pytest -q
```

Chạy thử mới dùng đường dẫn mới, ví dụ:

```powershell
.\.venv\Scripts\python.exe q_learning.py train cologne1 --episodes 1 --output reports/stage5/train_probe_new/cologne1
.\.venv\Scripts\python.exe q_learning.py evaluate cologne1 --model reports/stage5/train/cologne1/cologne1_q_table.json --seed 42 --output reports/stage5/evaluation_probe_new/cologne1
```

Hai lệnh ví dụ mới này chưa chạy. Test trace tự thu result.json dưới stage5;
test policy evaluate tham chiếu model bàn giao trong train/<map>. Khi thử
model khác, lưu ngoài stage5 hoặc cập nhật test reference tương ứng.

## 8. Vị trí chỉ code khi thuyết trình và bước tiếp theo

| Nội dung | Tệp / hàm |
|---|---|
| Input queue từng lane | sumo_env.py: SumoEnv.get_state |
| Mapping lane/edge và thứ tự nhóm được lưu | q_learning.py: build_encoding |
| State tuple rời rạc | q_learning.py: encode_state |
| Khởi tạo Q=0; đọc state lạ không ghi bảng | q_learning.py: get_q_values |
| Epsilon-greedy, tie-break, đổi cột thành ID pha | q_learning.py: choose_action |
| Công thức Q và terminal target | q_learning.py: update_q_value |
| Lịch epsilon, theo episode | q_learning.py: episode_epsilon |
| Train nhiều episode, seed riêng, checkpoint | q_learning.py: train |
| Vòng điều khiển online; env.step(action) | q_learning.py: run_episode |
| Gửi tín hiệu/tiến đồng hồ SUMO | sumo_env.py: SumoEnv._run_signal |
| Vàng/đỏ/clearance, reward/done | sumo_env.py: SumoEnv.step, _junction_vehicles |
| Lưu/tải JSON và kiểm tra tương thích | q_learning.py: save_q_table, load_q_table, make_metadata |
| Evaluate greedy không update/chèn state | q_learning.py: evaluate; nhánh training trong run_episode |
| Phép tính tay/terminal | test_q_learning.py: test_q_update_hand_calculation_correct_column_and_negative_q, test_terminal_does_not_read_or_create_next_state |
| Replay từng update và RNG từ CSV | test_q_learning.py: test_replay_every_training_update_and_rng_against_checkpoints |

**Đủ nền tảng kỹ thuật để bắt đầu DQN trong phạm vi học tập**: state thông tin
gốc/action/reward/timing chung đã rõ, QL có test công thức, train/evaluate và
lưu/tải thực. Chưa đủ để tuyên bố đã học tốt hoặc hoàn tất đánh giá thuật toán.
DQN sau này nên dùng cùng incoming/outgoing/phase thông tin gốc (có thể chuẩn
hóa), action/reward/control interval này và ghi rõ khác biệt với bin của QL.

Chưa kiểm tra hội tụ, train dài, nhiều exploration seed, nhu cầu khác, GUI,
bước dưới 1 s, mọi hình học thân xe hoặc mọi nguyên nhân teleport. Ngưỡng thô,
state aliasing, tie-break đầu tiên và thiếu coverage có thể gây đói pha nặng.
Không có fairness/max-wait; không chứng nhận an toàn tuyệt đối từ các lượt
không collision, không thêm heuristic hoặc chỉnh môi trường để làm đẹp điểm.
