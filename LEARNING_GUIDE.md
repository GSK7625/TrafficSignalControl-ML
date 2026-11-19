# Đọc và thuyết trình TrafficSignalControl-ML

Đây là bài học Python + SUMO/TraCI cho **một bộ điều khiển trên mỗi map**
Cologne1/Ingolstadt1. Bắt đầu bằng luồng dưới đây; chưa cần đọc các script
điều tra.

## 1. Đọc theo thứ tự nào?

1. [README.md](README.md) để cài đặt và chạy thử, rồi xem [AGENTS.md](AGENTS.md) về quy tắc pha an toàn.
2. Trong [src/sumo_env.py](src/sumo_env.py), đọc `get_state()`, `step()`, rồi mới đọc `reset()` và các
   hàm kiểm tra mạng trong `__init__()`.
3. [src/ft.py](src/ft.py) có `fixed_time()`. [src/mp.py](src/mp.py) có `calculate_pressures()` và `max_pressure()`.
   Cả hai baseline đều gọi `env.step(action)`.
4. Trong [src/q_learning.py](src/q_learning.py), đọc `build_encoding()` → `encode_state()` → `choose_action()`
   → `update_q_value()` → `run_episode()` → `train()`.
5. [src/evaluate.py](src/evaluate.py) có `greedy_steps()` và `run_evaluation()`.
   [src/evaluation_metrics.py](src/evaluation_metrics.py) có `trip_metrics()` và `signal_metrics()`.
6. Đọc [báo cáo Stage 8](reports/stage8/README.md) và [bảng tổng hợp](reports/stage8/summary.csv),
   sau đó kiểm tra bằng [dữ liệu từng lượt](reports/stage8/runs.csv).

Luồng train: **reset → queue/phase → state rời rạc → epsilon-greedy →
step an toàn → reward/state sau → cập nhật Q → lặp → checkpoint**.
Luồng evaluate: **load → queue/phase → state → greedy → step → metric**;
không cập nhật hoặc thêm hàng Q.

## 2. Input và state nằm ở đâu?

`SumoEnv.get_state()` đọc `getLastStepHaltingNumber()` trên các boundary
lane duy nhất, cho hai vector `incoming`, `outgoing` và `phase` hiện tại.
Đây là xe đang dừng trong vùng đo; không bao gồm mọi xe ngoài mạng hoặc xe
chưa chèn, không đo nhu cầu rẽ của từng movement trên lane dùng chung.

`q_learning.build_encoding()` đọc membership lane→edge từ XML; mỗi lane
chỉ xuất hiện một lần. `encode_state()` cộng queue trên các lane cùng edge,
rời rạc hóa bằng ngưỡng trong metadata, rồi nối **ID pha gốc** vào tuple.

| Ngưỡng inclusive `queue_bins` | Các bin |
|---|---|
| `[0,4]` mặc định, model Stage5/7 | 0; 1–4; >=5 |
| `[0,4,14]` thí nghiệm Stage8 | 0; 1–4; 5–14; >=15 |

Thêm bin giảm một phần mất thông tin, vẫn gộp queue15 với120 và các phân bố
lane khác nhau có cùng tổng. Không có bằng chứng đây là ngưỡng tối ưu.
Không xóa chiều outgoing. `load_q_table()` kiểm tra map, action, encoding,
thời gian, trạng thái đèn runtime và SHA dữ liệu trước khi dùng model.

## 3. Agent chọn action và học như thế nào?

`choose_action()` thực hiện epsilon-greedy: xác suất epsilon chọn action
ngẫu nhiên bằng RNG khám phá riêng; còn lại lấy Q lớn nhất. Hòa lấy action
đầu theo thứ tự đã lưu. State chưa biết được đọc như hàng0; evaluate không
chèn hàng này vào bảng. Q=0 không chứng minh action chưa học: xem số update
tái dựng từ log nếu cần kiểm chứng.

Action chỉ là pha xanh hợp lệ: Cologne1 **0–7**, Ingolstadt1 **0,2,4**.
Action4 ở Ingolstadt1 là cột thứ3, không phải cột thứ5.

`update_q_value()` chứa trực tiếp công thức:

```text
target = r                              nếu done
target = r + gamma * max Q(s',a')        nếu chưa done
Q(s,a) += alpha * (target - Q(s,a))
```

Alpha0,1 là tốc độ học; gamma0,95 discount theo **quyết định**, không theo
giây. `episode_epsilon()` giảm tuyến tính từ1 xuống0,05 qua15 episode;
khám phá dùng `random.Random(7)`, còn SUMO seed23423 được giữ từng episode.
`train()` bắt đầu bảng rỗng và lưu checkpoint mỗi episode, không resume.

## 4. SUMO nhận lệnh ở đâu? Reward là gì?

Chỉ `SumoEnv.step()` thực thi action. Đổi pha sẽ đóng xanh bằng vàng,
all-red tối thiểu2s, chờ xe clear rồi xanh30s. Vàng Cologne1=8s,
Ingolstadt1=6s; giữ nguyên pha thì thêm30s xanh. `_run_signal()` gọi
`setRedYellowGreenState()` và tiến từng giây, kiểm đồng hồ/tín hiệu thực.
Không chuyển trực tiếp giữa hai xanh xung đột; horizon có thể cắt lượt cuối.

Reward ở cuối `step()`:

```text
queue_seconds = tổng qua từng giây (xe dừng trên incoming lane duy nhất × 1s)
reward = -queue_seconds / elapsed
```

Tính cả xanh/vàng/đỏ/clearance. Reward không chứa trực tiếp completion,
xe chưa chèn, teleport hay tuổi không G. Tổng reward episode cộng các trung
bình theo decision nên không bằng âm queue_seconds toàn giờ. Chất lượng
phải được kiểm tra bằng nhiều metric, không chỉ reward.

## 5. Chạy training và evaluation

Làm theo [lệnh chạy ở README](README.md#chạy-ft-mp-q-learning-và-guard).
Model Stage 5 và Stage 7 dùng ngưỡng mặc định `0 4`. Model Stage 8
`QL-NewState` dùng `--queue-bins 0 4 14`. `src/evaluate.py` ghi tripinfo và
tính metric chung cho FT, MP và QL. `src/q_learning.py evaluate` phục vụ
chẩn đoán QL bằng trace từng quyết định.

Evaluate khóa Q trong RAM và kiểm SHA model trước/sau. Mỗi lượt ghi
`actions.csv`, `signals.csv`, `metadata.json`, `command.txt`, `result.json`
và XML/log cần đối chiếu. Training có `training.csv` và checkpoint; không
xếp hạng kết quả đang train với baseline evaluate.

Stage8 có biến thể **QL-NewState-Guard**, bật riêng bằng `--guard-seconds 300`
trong `evaluate.py`. `greedy_steps()` theo dõi tuổi nhu cầu edge tại các thời
điểm quyết định; khi quá hạn, ưu tiên edge lâu nhất và action có G cho edge
đó với Q cao nhất. Queue=0 hoặc có G thực thì reset tuổi. Đây là QL có ràng
buộc phục vụ edge, không phải Tabular QL thuần, không bảo đảm từng movement
trên lane dùng chung và không bảo đảm waiting của xe <=300s. `step()` vẫn
quản lý mọi vàng/đỏ/clearance; không có lệnh đèn riêng cho Guard.

## 6. Xem và giải thích kết quả

`evaluation_metrics.py` dùng cùng định nghĩa cho mọi thuật toán:

- Completion=xe hoàn thành hợp lệ/requested; throughput=xe hợp lệ×3600/horizon,
  đơn vị xe/h. Hợp lệ: arrived trong horizon, không vaporized, chưa teleport.
- Waiting/timeLoss theo SUMO; travel=arrival−actual depart, đơn vị s/xe,
  cùng cohort hoàn thành hợp lệ. Không gán0 cho xe chưa xong/chưa chèn.
- Queue mean=queue_seconds/horizon, đơn vị xe, tính đủ từng giây mọi stage.
- Teleport events khác số xe từng teleport. Báo cả unfinished/undeparted.
- Movement coverage đếm movement có `G`; longest no-green gap bao gồm hai
  biên horizon, là metric tín hiệu, không phải waiting time của xe.

Đọc completion/throughput, teleport và khả năng phục vụ trước khi nhìn
waiting. SUMO cùng seed vẫn có thể khác quỹ đạo: giữ mọi repeat và báo range.
Đọc checkpoint bằng thay đổi Q, tỷ lệ đổi greedy, state mới và coverage;
reward tăng hoặc một seed tốt không chứng minh hội tụ.

## 7. File nào chưa cần học trước?

| File | Vai trò, khi nào cần |
|---|---|
| `tools/analyze_ql.py` | Điều tra offline Stage6.2, replay từng update/RNG, mất thông tin state |
| `tools/stage7_summary.py` | Thực nghiệm offline Stage7, coverage/checkpoint/khóa model/matrix |
| `tools/stage8_summary.py` | Thực nghiệm offline Stage8, dùng lại replay và metric, đo chuỗi giữ action |
| `tools/stage6_1_probe.py` | Điều tra sâu SUMO bằng observer và native replay |
| `reports/stage6_1/verify.py` | Kiểm metric độc lập từ XML/CSV |
| `tests/test_*.py`, `tools/smoke_test.py` | Kiểm công thức, tín hiệu, save/load và pipeline |
| `tools/plot_results.py` | Vẽ kết quả đã lưu |

Để thuyết trình, mở trực tiếp năm vị trí: `get_state()` (input),
`choose_action()` (action), `update_q_value()` (công thức), `step()`/
`_run_signal()` (SUMO), `step()` và `read_metrics()` (reward/kết quả).
