# Stage 7 — Huấn luyện Q-Learning có kiểm soát — 15/11/2026

**Đã hoàn thành thí nghiệm; kết quả khác nhau theo map.** Cologne1 học nhiều ô Q hơn,
ít unseen/tie hơn, cấp xanh đủ movement, nhưng model được chọn **không cải thiện
đáng tin cậy trên test**: completion trung bình giảm và biến thiên cùng seed rất lớn.
Ingolstadt1 cải thiện completion/throughput so với model cũ trong cả sáu lượt test,
nhưng chưa đồng thời tốt hơn FT về waiting, gap hoặc teleport. Không kết luận hội tụ.

## 1. Phạm vi, file và bảo toàn dữ liệu

Đã đọc AGENTS.md, q_learning.py, test_q_learning.py, sumo_env.py, evaluate.py,
evaluation_metrics.py, analyze_ql.py, README Stage 5/6/6.1/6.2 và bốn bảng/JSON Stage 6.2
được yêu cầu. Git ban đầu sạch tại `c42e396e285ef7fbfbafd0d2474f34e626bd4dba`;
remote đúng repository GSK7625/TrafficSignalControl-ML.

- Sửa `analyze_ql.py`: `replay_training(..., training_dir=None)` nhận thư mục train
  tùy chọn; mặc định Stage 5 giữ nguyên. Chỉ 3 dòng thêm, 2 dòng thay.
- Thêm `stage7_summary.py`: đọc log offline, dùng lại replay/metric/greedy evidence cũ,
  thống kê coverage và thay đổi checkpoint, kiểm tra matrix lượt chạy và model đã khóa.
- Thêm `test_stage7.py`: 6 kiểm tra nhỏ cho đường dẫn replay, coverage, epsilon 15ep,
  lựa chọn validation, thay đổi argmax và matrix thiếu/trùng/sai SHA.
- Tất cả model/log mới nằm trong `reports/stage7/`; không commit/push hoặc cài dependency.

SHA của **1.333 file tracked ban đầu** được kiểm tra trước/sau. Chỉ `analyze_ql.py`
thay đổi có chủ đích; toàn bộ data, model và kết quả Stage 1–6.2 giữ nguyên.
Không sửa `q_learning.py`, `SumoEnv`, FT, MP, evaluate.py hoặc evaluation_metrics.py.
Không có DQN, PyTorch, neural network hoặc framework RL.

| Model QL-3ep giữ nguyên | SHA-256 trước = sau |
|---|---|
| Cologne1: `../stage5/train/cologne1/cologne1_q_table.json` | `6b824ae7dbe63e93cef9f03bd1fa757d727c3b3153321121f7bdd8ad3bd9b4d9` |
| Ingolstadt1: `../stage5/train/ingolstadt1/ingolstadt1_q_table.json` | `564a6fae2b8e22044fbaabd8b46ea38631f1f7fa170687eb992843cee1a73daa` |

## 2. Thiết kế đã chốt trước test và số lượt thực tế

[PLAN.md](PLAN.md) được ghi trước pilot/training/test. **10 episode pilot + 60 episode
chính**, tất cả đủ 3.600 giây mô phỏng, đã hoàn thành; không có lượt chính gián đoạn
hay bị loại. Pilot 5 episode/map, exploration seed 7, lưu riêng trong `pilot/`;
không tiếp tục bảng pilot để train chính. Bốn lượt train chính đều bắt đầu bảng rỗng:
2 map × exploration seed 7/17 × 15 episode. Không gộp Q-table.

- SUMO seed **23423 cố định ở mọi episode training**, không tự tăng seed.
- Alpha=0.1, gamma=0.95 theo decision; epsilon tuyến tính 1→0.05 qua 15 episode
  (episode 8 là 0.525). Pilot dùng cùng hai đầu epsilon nhưng chỉ 5 episode.
- Bins 0 / 1–4 / >=5, Q khởi tạo 0, action ID gốc, tie-break first giữ nguyên.
- Reward `-queue_seconds/elapsed`; green 30s; yellow Cologne1 8s/Ingolstadt1 6s,
  all-red >=2s và clearance cũ. Horizon có thể cắt lượt điều khiển cuối.
- Cologne1: TLS `cluster_357187_359543`, action 0–7, 20 movement, 8 lane vào/8 lane ra,
  2.015 yêu cầu, [25200,28800). Ingolstadt1: TLS
  `cluster_274083968_cluster_1200364014_1200364088`, action 0/2/4, 8 movement,
  7 lane vào/6 lane ra, 1.716 yêu cầu, [57600,61200).

Python .venv 3.11.9; SUMO/TraCI/sumolib 1.28.0. Binary
`C:/Program Files (x86)/SUMO/bin/sumo.exe`, SHA
`35d45c1166ceb5b72cce37676216f79124fcfe05920a496b14daee19acdaf9f3`, trước=sau.
Các episode độc lập khởi chạy SUMO qua CLI hiện có. Validation được chạy khi từng
model cuối sẵn sàng; test chạy hai map song song, từng map chạy tuần tự theo manifest.
Không dùng chế độ thời gian thực. Đường dẫn output/cổng TraCI riêng cho từng process.

**B thay cả số episode lẫn lịch khám phá theo thời gian.** Đây là kiểm chứng gói
“training dài hơn + khám phá kéo dài”, không tách tác dụng riêng của hai yếu tố.
SUMO có thể khác quỹ đạo cùng seed; khác biệt giữa exploration seeds cũng chứa
biến thiên SUMO trong training, không được quy hoàn toàn cho RNG khám phá.

## 3. Q-table và coverage trước/sau

Coverage = số cặp đã cập nhật / (số state thực gặp ở quyết định train × số action).
Không dùng toàn bộ không gian state lý thuyết làm mẫu số. Đếm cập nhật từ replay,
**không suy chưa thử chỉ từ Q=0**. “Một lần” là số cặp chỉ nhận đúng một update.

| Map | Model | State | Update | Cặp / mẫu số | Coverage | Cặp một lần |
|---|---|---:|---:|---:|---:|---:|
| Cologne1 | QL-3ep | 57 | 281 | 175/456 | 38.38% | 107 |
| Cologne1 | QL-15ep-e7 | 129 | 1400 | 500/1032 | 48.45% | 276 |
| Cologne1 | QL-15ep-e17 | 121 | 1397 | 461/968 | 47.62% | 261 |
| Ingolstadt1 | QL-3ep | 15 | 304 | 40/45 | 88.89% | 10 |
| Ingolstadt1 | QL-15ep-e7 | 17 | 1515 | 49/51 | 96.08% | 8 |
| Ingolstadt1 | QL-15ep-e17 | 20 | 1513 | 50/60 | 83.33% | 8 |

Cả bốn model mới học nhiều cặp hơn model cũ. Tỷ lệ không luôn tăng: Ingolstadt1 e17
gặp thêm state nên mẫu số tăng nhanh hơn số cặp học. Cologne1 vẫn có hơn nửa số
cặp đã cập nhật chỉ học một lần; “500 cặp” không có nghĩa 500 ước lượng Q đã chính xác.
Pilot cuối có 252 cặp/73 state Cologne1 và 44 cặp/15 state Ingolstadt1.

[analysis/training.csv](analysis/training.csv) có từng episode: epsilon, reward,
queue-seconds, state mới/tích lũy, coverage, action counts, green, movement,
teleport, inserted/arrived thô và SHA checkpoint. [state_actions.csv](analysis/state_actions.csv)
ghi số update **mỗi cặp tại mỗi checkpoint**, gồm cả ô 0 update. Replay khớp toàn bộ
Q trước/sau, RNG, terminal và checkpoint; raw lane queues khớp encoding.
Training không bật tripinfo: arrivals thô không được gọi là completion hợp lệ.

## 4. Validation và khóa model

12 lượt: 2 map × 3 model × seeds **42/12345**. Đây là seed đã được xem ở Stage 5/6,
chỉ gọi là validation. Không train lại QL-3ep. Evaluate epsilon=0, không cập nhật Q,
không tạo state mới; runner kiểm tra cả bảng trong RAM và SHA model trước/sau.

Chỉ xét **hai model cuối 15 episode**, không tìm best checkpoint/reward. Xếp theo
completion trung bình giảm dần (cùng demand/horizon nên cũng là throughput), rồi
min movement coverage giảm dần, mean teleport tăng dần, max gap tăng dần; hòa chọn e7.

| Map | Model | Completion mean | Throughput xe/h | Queue xe | Teleport events | Movement có G mỗi lượt | Gap lớn nhất s |
|---|---|---:|---:|---:|---:|---|---:|
| Cologne1 | QL-3ep | 49.18% | 991.0 | 47.59 | 67.5 | 10/20; 10/20 | 3600 |
| Cologne1 | QL-15ep-e7 **chọn** | 83.97% | 1692.0 | 53.60 | 44.5 | 20/20; 20/20 | 2040 |
| Cologne1 | QL-15ep-e17 | 54.62% | 1100.5 | 91.49 | 51.5 | 20/20; 20/20 | 3030 |
| Ingolstadt1 | QL-3ep | 92.37% | 1585.0 | 8.55 | 2.5 | 8/8; 8/8 | 418 |
| Ingolstadt1 | QL-15ep-e7 **chọn** | 97.12% | 1666.5 | 8.86 | 3.0 | 8/8; 8/8 | 530 |
| Ingolstadt1 | QL-15ep-e17 | 88.14% | 1512.5 | 6.89 | 9.0 | 8/8; 8/8 | 1114 |

Ingolstadt1 e17 có queue nhỏ hơn nhưng completion thấp và teleport cao hơn, nên
không được chọn. Cologne1 e17 seed42 chỉ hoàn thành 680/2015, chọn action5 96/115
quyết định; vẫn giữ nguyên lượt này. Hai exploration seeds cho policy khác rõ rệt.

Khóa lúc **09:24:41 UTC, 16:24:41 giờ Việt Nam 15/11/2026**, trước lượt test đầu tiên.
[models/selection.json](models/selection.json) lưu tiêu chí, path, seed và SHA; không
đổi model sau khi xem test. Model vẫn nằm trong thư mục training, không tạo bản sao:

| Model được chọn | SHA-256 |
|---|---|
| `training/cologne1/exploration_7/cologne1_q_table.json` | `36626479972dd09f0e23582c8f03980daba5c74b6474f7627d5d785651bf4b64` |
| `training/ingolstadt1/exploration_7/ingolstadt1_q_table.json` | `24169b3ebe29d41548fc08daf4f96b4f9aa704a32343363534e4d0ae75a7573a` |

SHA/metadata của cả bốn model mới ở [training_verification.json](training_verification.json).
Model cuối khớp từng byte checkpoint episode015; hai seed tạo bảng khác nhau, mỗi
bảng đã được replay độc lập từ 0. Chưa test model e17 vì không được chọn trên validation.

## 5. Unseen, hòa Q và chọn action chưa thử

Bảng là **số decision / tổng decision** gộp trong partition, không phải phần trăm
state lý thuyết. Unseen là state không có row trong model. Tied là nhiều action cùng
max Q. Untried là action được chọn có **0 update trong training**, kể cả state lạ.
Distinct unseen theo từng lượt nằm ở `analysis/runs.csv`; trace từng decision nằm
ở [policy_decisions.csv](analysis/policy_decisions.csv).

| Map | Nguồn / model | Unseen | Tie | Chọn chưa thử |
|---|---|---:|---:|---:|
| Cologne1 | Stage6 QL-3ep | 112/509 = 22.00% | 354/509 = 69.55% | 354/509 = 69.55% |
| Cologne1 | Validation QL-3ep | 45/204 = 22.06% | 143/204 = 70.10% | 143/204 = 70.10% |
| Cologne1 | Validation QL-15ep-e7 | 5/195 = 2.56% | 37/195 = 18.97% | 37/195 = 18.97% |
| Cologne1 | Test QL-3ep | 133/610 = 21.80% | 422/610 = 69.18% | 422/610 = 69.18% |
| Cologne1 | Test QL-15ep-e7 | 8/647 = 1.24% | 66/647 = 10.20% | 66/647 = 10.20% |
| Ingolstadt1 | Stage6 QL-3ep | 1/511 = 0.20% | 16/511 = 3.13% | 16/511 = 3.13% |
| Ingolstadt1 | Validation QL-3ep | 0/204 | 7/204 = 3.43% | 7/204 = 3.43% |
| Ingolstadt1 | Validation QL-15ep-e7 | 0/206 | 0/206 | 0/206 |
| Ingolstadt1 | Test QL-3ep | 2/614 = 0.33% | 22/614 = 3.58% | 22/614 = 3.58% |
| Ingolstadt1 | Test QL-15ep-e7 | 0/616 | 0/616 | 0/616 |

Các cột tie và untried bằng nhau trong những nhóm này là kết quả thực tế, không phải
định nghĩa tương đương: Ingolstadt1 e17 validation có **0 tie nhưng 2/198 lần chọn
action chưa thử**. Một action Q=0 duy nhất vẫn có thể thắng toàn bộ Q âm đã học.

## 6. Test giữ riêng: so sánh chính thức

Seeds **31001,31002,31003** được xác định trong PLAN trước training; không dùng để
chọn hoặc điều chỉnh model. 2 repeats/seed × 4 controller × 2 map = **48/48 lượt**,
đủ 3600s/lượt. Không chạy lại để chọn kết quả đẹp. Mỗi model/controller/map có 6 lượt.

Bảng dưới lấy trung bình không trọng số giữa sáu lượt; N là số chuyến hoàn thành
hợp lệ trung bình mỗi lượt. Throughput=N×3600/horizon, nên cùng số N với horizon1h.
Movement là min–max giữa lượt, không phải số action. Nhu cầu **2015/lượt Cologne1,
1716/lượt Ingolstadt1**. Số thập phân ở event count là mean, không phải nửa sự kiện.

| Map | Controller | N valid / xe/h | Completion | Queue xe | Teleport events | Xe từng teleport | Movement có G | Gap max qua 6 lượt s |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Cologne1 | FT | 1893.67 | 93.98% | 73.53 | 0.00 | 0.00 | 20/20 | 250 |
| Cologne1 | MP | 1969.17 | 97.73% | 27.76 | 9.83 | 9.83 | 20/20 | 1010 |
| Cologne1 | QL-3ep | 1006.17 | 49.93% | 47.80 | 68.83 | 61.50 | 10–20/20 | 3600 |
| Cologne1 | QL-15ep-e7 | 872.33 | 43.29% | 111.63 | 74.33 | 74.00 | 20/20 | 3520 |
| Ingolstadt1 | FT | 1686.67 | 98.29% | 9.21 | 0.00 | 0.00 | 8/8 | 84 |
| Ingolstadt1 | MP | 1662.67 | 96.89% | 5.72 | 0.33 | 0.33 | 8/8 | 388 |
| Ingolstadt1 | QL-3ep | 1579.00 | 92.02% | 8.11 | 4.00 | 4.00 | 8/8 | 702 |
| Ingolstadt1 | QL-15ep-e7 | 1698.67 | 98.99% | 9.03 | 1.67 | 1.67 | 8/8 | 438 |

Cologne1 mới giảm **133.83 xe/h, 6.64 điểm phần trăm completion** so với QL-3ep
cùng test; queue tăng63.83 xe, teleport events tăng5.50/lượt. Không có bằng chứng
cải thiện đáng tin cậy dù coverage/state lạ/tie tốt hơn. Baseline cũ Stage6 phục vụ
10/20 ở cả5 lượt; trên test mới có **một** lượt seed31002 repeat1 phục vụ20/20,
năm lượt còn lại10/20. Không sửa lịch sử để làm baseline trông nhất quán hơn.

Ingolstadt1 mới tăng **119.67 xe/h, 6.97 điểm phần trăm** so với QL-3ep và giảm
teleport4→1.67; queue tăng8.11→9.03. So FT tăng12 xe/h, 0.70 điểm phần trăm,
nhưng waiting/timeLoss/gap/teleport kém hơn. Đây là cải thiện một số metric trong
tập hữu hạn, không phải thắng toàn diện hoặc bảo đảm khái quát sang nhu cầu khác.

## 7. Cohort thời gian chuyến, insertion và xe chưa xong

Waiting/travel/timeLoss chỉ trên **cùng cohort valid của mỗi lượt**: arrived trong
horizon, không vaporized, chưa từng teleport. Travel=arrival−actual depart;
waiting/timeLoss theo SUMO, không cộng departDelay. Không gán0 cho xe chưa xong.
Cột NΣ là tổng các lần hoàn thành hợp lệ qua6 lượt, **không phải số ID duy nhất
giữa các lượt**. Trung bình thời gian ở đây là mean của6 mean từng lượt, không
phải pooled mean của NΣ chuyến. Raw N từng lượt và mọi metric nằm trong runs.csv.

| Map | Controller | NΣ valid (N min–max/lượt) | Wait s/xe | Travel s/xe | TimeLoss s/xe | Insertion | Unfinished mean | Undeparted mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Cologne1 | FT | 11362 (1870–1906) | 144.88 | 187.98 | 164.94 | 97.47% | 70.33 | 51.00 |
| Cologne1 | MP | 11815 (1964–1973) | 60.87 | 97.02 | 74.13 | 99.69% | 29.67 | 6.33 |
| Cologne1 | QL-3ep | 6037 (987–1086) | 30.11 | 64.73 | 37.30 | 58.60% | 114.83 | 834.17 |
| Cologne1 | QL-15ep-e7 | 5234 (212–1634) | 233.40 | 275.26 | 252.56 | 56.34% | 188.83 | 879.83 |
| Ingolstadt1 | FT | 10120 (1681–1690) | 31.82 | 64.20 | 43.42 | 99.77% | 25.33 | 4.00 |
| Ingolstadt1 | MP | 9976 (1658–1671) | 35.11 | 67.23 | 46.63 | 98.66% | 30.00 | 23.00 |
| Ingolstadt1 | QL-3ep | 9474 (1573–1584) | 55.02 | 86.60 | 66.78 | 94.46% | 38.00 | 95.00 |
| Ingolstadt1 | QL-15ep-e7 | 10192 (1698–1700) | 48.25 | 81.20 | 60.29 | 99.94% | 14.67 | 1.00 |

Insertion=inserted/requested. Unfinished=inserted chưa arrived, undeparted=requested−inserted.
Mean queue=queue-seconds/3600 trên 3600 mẫu1s, gồm xanh/vàng/đỏ/clearance,
là tổng xe dừng trên lane vào biên duy nhất, không phải xe/lane hay queue toàn mạng.
Teleport events là số start; số xe từng teleport đếm unique ID trong **mỗi lượt**.
48 lượt test có collisions=0 và emergencyBraking=0 theo bộ đo chung; raw warnings,
emergency stops và XML vẫn được giữ. Điều này không chứng nhận mọi điều kiện an toàn.

## 8. Movement còn bị bỏ đói không?

Model Cologne1 e7 validation chọn cả action4–7, khác vùng0–3 của Stage6 cũ.
Model mới cấp G đủ20/20 trong cả2 validation và6 test; không còn movement thiếu
G **suốt cả giờ** trong các lượt này. Nhưng gap test lên **3520/3600s**, tức vẫn có
bỏ phục vụ kéo dài. Một lần có G không phải bảo đảm công bằng hoặc đáp ứng nhu cầu.

Ví dụ test seed31001 repeat1 (repeat2 cũng chỉ212 xe): action7 được chọn105/117
lượt, trong đó state `(2,2,2,2,0,0,0,0,7)` xuất hiện104 lần. Ở model đã khóa:

```text
Q theo action0..7:
[-67.7565, -71.5712, -76.0400, -72.2831, -67.7881, -67.6349, -76.2913, -66.6589]
Số update tương ứng: [6,9,11,8,7,6,9,7]
Argmax duy nhất = action7; Q7 đã được cập nhật7 lần.
```

Đây là chuỗi trực tiếp **state bão hòa → argmax đã học7 → giữ pha7 → state mã hóa
vẫn như cũ → chọn7 tiếp**. 104 quyết định này không do state lạ, Q chưa thử hoặc
hòa Q. Giảm tie/unseen và tăng coverage không tự sửa thứ hạng Q tại state này.
Gaps bao gồm biên đầu/cuối horizon; chưa đo nhu cầu/chờ liên tục theo từng movement,
nên không quy toàn bộ879.83 xe chưa chèn trung bình cho một movement cụ thể.

## 9. Biến thiên qua các lần lặp

Hai lần/seed, không đủ ước lượng toàn bộ phân phối ngẫu nhiên. Bảng là N valid thực:

| Map | Seed | QL-3ep r1 / r2 | QL-15ep-e7 r1 / r2 |
|---|---:|---|---|
| Cologne1 | 31001 | 995 / 995 | 212 / 212 |
| Cologne1 | 31002 | 1086 / 987 | 403 / 1214 |
| Cologne1 | 31003 | 987 / 987 | 1559 / 1634 |
| Ingolstadt1 | 31001 | 1580 / 1580 | 1700 / 1700 |
| Ingolstadt1 | 31002 | 1573 / 1573 | 1698 / 1698 |
| Ingolstadt1 | 31003 | 1584 / 1584 | 1698 / 1698 |

Cologne1 mới seed31002 chênh **811 xe, 40.25 điểm phần trăm**, lớn hơn độ lớn chênh
mean cũ–mới133.83 xe. FT có range trong cùng seed tối đa26 xe, MP5 xe,
QL-3ep99 xe. Không dùng một lượt đẹp để gọi cải thiện. Gap3520s/212 xe của seed31001
lặp lại ở cả hai lần, cho thấy một kết quả xấu có thể tái xuất hiện ổn định.

Ingolstadt1: hai lần cùng seed cho các metric kết quả đã ghi trùng nhau ở cả bốn
controller; model mới1698–1700 xe, cao hơn model cũ1573–1584 trên cả3 seed, và hơn FT
ở từng seed. Chỉ kết luận **ổn định trong hai lần đã quan sát**, không bảo đảm SUMO
toàn cục xác định hoặc mọi raw trace trùng byte. [variation.csv](analysis/variation.csv)
ghi chênh queue/teleport/completion trong cùng seed cho cả FT/MP.

## 10. Đối chiếu Stage 6 và dấu hiệu hội tụ

Stage6 là 5 seed cũ, test Stage7 là 3 seed mới ×2 lần; đây là hai tập khác nhau,
không gộp hoặc xem chênh giữa stage là tác động nhân quả. So sánh chính thức vẫn
là QL-3ep/QL-15ep/FT/MP trên cùng test ở mục6.

| Map / controller | Stage6: completion; xe/h; queue; teleport | Test Stage7: cùng thứ tự |
|---|---|---|
| Cologne1 FT | 93.86%; 1891.2; 73.87; 0 | 93.98%; 1893.67; 73.53; 0 |
| Cologne1 MP | 97.66%; 1967.8; 28.39; 10.6 | 97.73%; 1969.17; 27.76; 9.83 |
| Cologne1 QL-3ep | 49.20%; 991.4; 47.65; 65.8 | 49.93%; 1006.17; 47.80; 68.83 |
| Cologne1 QL-15ep-e7 | chưa có | 43.29%; 872.33; 111.63; 74.33 |
| Ingolstadt1 FT | 97.95%; 1680.8; 9.22; 0.2 | 98.29%; 1686.67; 9.21; 0 |
| Ingolstadt1 MP | 97.28%; 1669.4; 5.64; 0.6 | 96.89%; 1662.67; 5.72; 0.33 |
| Ingolstadt1 QL-3ep | 93.10%; 1597.6; 8.30; 3.4 | 92.02%; 1579.0; 8.11; 4.0 |
| Ingolstadt1 QL-15ep-e7 | chưa có | 98.99%; 1698.67; 9.03; 1.67 |

Không có bằng chứng hội tụ. Giữa checkpoint14→15, greedy action đổi trên state
chung: Cologne1 e7 **26/124**, e17 **26/117**; Ingolstadt1 e7 **8/17**, e17 **9/20**.
Max |ΔQ| tương ứng15.49,18.51,5.42,4.01. Năm checkpoint cuối vẫn đổi argmax nhiều;
reward và số row tăng không thay thế kiểm tra policy/value ổn định.

## 11. Kết luận giới hạn và hướng tiếp theo

1. Huấn luyện dài/khám phá kéo dài tăng số cặp đã học ở cả4 model; tỷ lệ coverage
   và chất lượng không nhất thiết tăng cùng nhau.
2. Model e7 giảm unseen/tie/untried rõ trên cả hai map. Cologne1 cải thiện validation
   nhưng thất bại khái quát trên test, còn gap kéo dài; không gọi là cải thiện tin cậy.
3. Ingolstadt1 có bằng chứng cải thiện completion/throughput so với QL-3ep trong tập
   test này; waiting cũng giảm so model cũ. Chưa vượt FT toàn diện: waiting cao hơn,
   gap lớn hơn, vẫn teleport. Không tuyên bố RL phải thắng baseline.
4. Chưa tách tác dụng budget khỏi lịch epsilon; chưa xác định nguyên nhân gốc SUMO
   khác quỹ đạo; chưa chứng minh ảnh hưởng nhân quả của aliasing, reward boundary,
   thứ hạng Q chưa chính xác, phân bố nhu cầu hoặc số update chưa đủ.

**Ưu tiên thí nghiệm tiếp theo: state**, một thay đổi riêng giúp phân biệt trạng thái
bão hòa/tuổi không được phục vụ, với phép đo và thiết kế mới chốt trước. Lý do:
vòng lặp action7 trên test có **unique max đã học**, nên chỉ đổi tie-break không chữa
được trực tiếp104 quyết định đó; model không phân biệt được các queue >=5 trong
cùng tuple. Đây là đề xuất cần ablation, không phải nguyên nhân đã chứng minh hay
bảo đảm thành công. Giữ reward và action-selection ở lần kiểm chứng state đó;
sau đó mới xem xét thí nghiệm riêng về reward/fairness hoặc chiến lược phục vụ tối thiểu.
Không tự thêm state, đổi reward/tie-break hay train tiếp trong Stage7 này.

## 12. Kiểm tra và cách chạy lại

- Trước sửa: `.venv/Scripts/python.exe -m pytest -q` → **151 passed /14.84s**.
- Sau sửa: cùng lệnh → **157 passed /17.50s**, không skip/deselect, có integration SUMO
  ngắn. Các test này không thay cho70 episode và60 lượt evaluation thực3600s.
  Test train cũ cũng tạo episode47s trong thư mục tạm; không tính vào ngân sách60
  episode chính hoặc pilot3600s. `tests_before.log`, `tests_after.log` giữ output.
- Pilot: replay RNG/Q/checkpoint, epsilon, timing, action, terminal đều đạt;
  không lỗi terminal/process. Vẫn giữ warning SUMO, không gọi warning là0.
- Cả60 episode chính khớp replay từ0; model cuối=checkpoint015 theo byte;
  raw queue hợp encoding, valid action IDs, reward/terminal đúng theo log.
- 90 lượt evaluate (30 Stage6 đọc lại +12 validation +48 test): đọc lại tripinfo,
  statistics, collisions, warnings, signals bằng `evaluation_metrics.read_metrics()`;
  **mọi field khớp result đã lưu**. Sáu file controller/metric có SHA giống Stage6
  trong metadata mọi lượt; binary trước=sau. Không khởi chạy lại30 lượt lịch sử.
- Matrix bắt buộc đủ90, không trùng; model test trùng SHA đã khóa, SHA selection
  trùng test manifest; evaluate không làm đổi file hoặc bảng Q trong RAM.
- Review offline riêng đã kiểm tra script/test; lỗi chấp nhận matrix thiếu đã được
  chặn bằng kiểm tra bắt buộc, có regression test. Không dùng review thay test SUMO.

Ví dụ tái tạo các lượt Cologne1 trong thư mục mới, chạy từ gốc dự án.

```powershell
.\.venv\Scripts\python.exe src/q_learning.py train cologne1 --episodes 5 --seed 23423 --exploration-seed 7 --output reports/stage7/pilot_repeat/cologne1
.\.venv\Scripts\python.exe src/q_learning.py train cologne1 --episodes 15 --seed 23423 --exploration-seed 7 --output reports/stage7/training_repeat/cologne1/exploration_7
.\.venv\Scripts\python.exe src/evaluate.py --algorithm QL --map cologne1 --seed 42 --model reports/stage7/training_repeat/cologne1/exploration_7/cologne1_q_table.json --output reports/stage7/validation_repeat/cologne1/QL-15ep-e7/seed_42
.\.venv\Scripts\python.exe tools/stage7_summary.py --output reports/stage7/analysis_from_saved_runs
.\.venv\Scripts\python.exe -m pytest -q
```

Đổi đường dẫn nếu thư mục đầu ra đã tồn tại. `training_manifest.json` ghi4 CLI train chính; `validation_manifest.json`
ghi12 CLI; `test_manifest.json` ghi48 CLI, thời điểm bắt đầu và SHA selection.
Mỗi raw run có `command.txt` SUMO, metadata, actions/signals, XML và warning logs.
Mỗi episode có checkpoint. `provenance_before.json`, `training_verification.json`,
`pilot_verification.json`, `verification.json` ghi nguồn, phiên bản, SHA và kết quả kiểm tra.

`stage7_summary.py` hiện phân tích bộ raw cố định `reports/stage7/`, chỉ `--output`
được chọn mới. Ví dụ tái tạo bảng mà không chạy SUMO vào thư mục tạm mới:

```powershell
.\.venv\Scripts\python.exe tools/stage7_summary.py --output C:/Temp/stage7-analysis-new
```

Không có chức năng resume, auto-tuning, dashboard hoặc framework mới.

## 13. Vị trí thuyết trình trực tiếp

| Kiến thức | File / hàm |
|---|---|
| Input lane queue và pha hiện tại | `sumo_env.py: SumoEnv.get_state()` |
| Nhóm lane→edge, ngưỡng và state rời rạc | `q_learning.py: build_encoding(), encode_state()` |
| Epsilon tuyến tính, action và tie-break | `q_learning.py: episode_epsilon(), choose_action(), get_q_values()` |
| Công thức Q, done chỉ lấy reward | `q_learning.py: update_q_value()` |
| Train độc lập từ0, RNG riêng, checkpoint | `q_learning.py: train(), run_episode(), save_q_table(), load_q_table()` |
| SUMO nhận lệnh/chuyển pha/clearance | `sumo_env.py: SumoEnv.step(), _run_signal()` |
| Reward queue-seconds/elapsed | `sumo_env.py: SumoEnv.step()` |
| Evaluate đóng băng và kiểm tra SHA/RAM | `evaluate.py: greedy_steps(), run_evaluation()` |
| Cohort valid và metric dùng chung | `evaluation_metrics.py: requested_trips(), trip_metrics(), signal_metrics(), read_metrics()` |
| Phát lại từng update, RNG và checkpoint | `analyze_ql.py: replay_training()` |
| Coverage, state mới, số lần học từng cặp | `stage7_summary.py: coverage(), training_evidence()` |
| Thay đổi argmax/Q giữa checkpoint | `stage7_summary.py: checkpoint_changes()` |
| Unseen/tie/đã thử, movement G thực | `analyze_ql.py: greedy_evidence(), evaluation_policy(), audit_signals()`; `stage7_summary.py: run_evidence()` |
| Quy tắc chọn chỉ dùng validation | `stage7_summary.py: choose_model()`; `models/selection.json` |
| Đủ lượt, đúng SHA, tổng hợp không trộn map | `stage7_summary.py: verify_run_matrix(), summarize(), main()` |

Bảng tổng hợp ở [analysis/summary.csv](analysis/summary.csv), từng lượt ở
[analysis/runs.csv](analysis/runs.csv), từng episode ở [analysis/training.csv](analysis/training.csv).
Biểu đồ [overview.png](overview.png) chỉ minh họa dữ liệu đã lưu; khoảng error bar
là min–max quan sát, không phải confidence interval.

Chưa kiểm chứng can thiệp nhân quả, hội tụ, nhu cầu liên tục của từng movement,
khả năng khái quát sang map/demand mới, hoặc sửa tính không xác định của SUMO.
