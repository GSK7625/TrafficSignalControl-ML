# TrafficSignalControl-ML

Dự án học Python, SUMO và TraCI qua bài toán điều khiển một nút giao trên mỗi bản đồ `cologne1` và `ingolstadt1`. Các bộ điều khiển hiện có gồm Fixed-Time (FT), Max-Pressure (MP), Tabular Q-Learning (QL) và Q-Learning với Guard.

## Cài đặt trên Windows

Cài Git, Python 3.11 và SUMO. Mở PowerShell tại thư mục gốc của bản clone Git. Đảm bảo `sumo.exe` có trong `PATH` để Python khởi chạy được trình mô phỏng. Nếu cần thêm đường dẫn, dùng `$env:Path = 'C:\Program Files (x86)\SUMO\bin;' + $env:Path` với thư mục cài đặt thực tế. [Hướng dẫn cài SUMO](https://sumo.dlr.de/docs/Installing/index.html#windows)

```powershell
git --version
sumo --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Nếu đã có `.venv`, dùng lại môi trường đó. Môi trường kiểm tra dùng Python 3.11.9 và SUMO, TraCI, sumolib 1.28.0. Bộ test có các ca khởi chạy SUMO thật.

## Chạy FT, MP, Q-Learning và Guard

Chạy các lệnh dưới đây từ gốc repository. Mỗi lần chạy nhóm lệnh sẽ tạo một đường dẫn mới trong thư mục tạm. Có thể xem vị trí kết quả bằng `$runRoot`. Ví dụ dùng `cologne1`; để chạy `ingolstadt1`, đổi tên bản đồ trong các lệnh và tên tệp Q-table cho khớp.

```powershell
$runRoot = Join-Path $env:TEMP ("traffic-signal-" + [guid]::NewGuid().ToString("N"))

.\.venv\Scripts\python.exe src/evaluate.py --algorithm FT --map cologne1 --seed 42 --output "$runRoot/ft"
.\.venv\Scripts\python.exe src/evaluate.py --algorithm MP --map cologne1 --seed 42 --output "$runRoot/mp"

.\.venv\Scripts\python.exe src/q_learning.py train cologne1 --episodes 1 --seed 23423 --exploration-seed 7 --output "$runRoot/train"
$model = "$runRoot/train/cologne1_q_table.json"
.\.venv\Scripts\python.exe src/evaluate.py --algorithm QL --map cologne1 --seed 42 --model $model --output "$runRoot/ql"
.\.venv\Scripts\python.exe src/evaluate.py --algorithm QL --map cologne1 --seed 42 --model $model --guard-seconds 300 --output "$runRoot/guard"
```

Một episode dùng để kiểm tra quy trình train, lưu và nạp Q-table. `--seed` là seed SUMO, còn `--exploration-seed` điều khiển khám phá khi train. Evaluate dùng bảng Q cố định và chọn tham lam. Guard dùng cùng bảng Q và có thể đổi action để phục vụ một edge đầu vào đã chờ lâu. Mốc 300 giây là ngưỡng xét ở thời điểm quyết định và áp dụng cho edge có hàng đợi quan sát được.

`--queue-bins` mặc định là `0 4`. Nếu train hoặc dùng model Stage 8 `QL-NewState`, chỉ định `--queue-bins 0 4 14` khi train model mới và khi evaluate model đó. Tham số evaluate phải khớp metadata của model.

Để thử ưu tiên pha chưa học, thêm `--exploration-strategy untried-first` vào lệnh train. Khi khám phá, agent lấy mẫu đều trong các pha chưa từng cập nhật Q ở state hiện tại. Sau khi thử đủ pha, agent lấy mẫu đều trong toàn bộ pha hợp lệ. Bộ đếm dùng chung qua các episode của một lượt train. Mặc định `uniform` dùng cách khám phá của QL-Original. Cả hai model dùng cùng lệnh evaluate tham lam.

Thí nghiệm state trên Cologne1 dùng lệnh train với `--overdue-state` và cách khám phá mặc định `uniform`. Khóa Q có thêm một số sau ID pha. Số 0 nghĩa là chưa có edge vào nào có queue quan sát được và thiếu xanh ưu tiên trong ít nhất 300 giây. Các số 1–4 chỉ edge chờ lâu nhất theo thứ tự `encoding.incoming_groups` trong model. Tuổi bắt đầu khi thấy queue dương tại một quyết định, đặt lại khi thấy queue bằng 0 hoặc khi một movement của edge thực nhận `G`. Model lưu phiên bản `overdue-edge-v1` và ngưỡng thời gian. Lệnh evaluate tự đọc encoding và chọn pha tham lam. Để đo riêng tác động của state, dùng lệnh evaluate QL mặc định với Guard tắt.

Cờ train `--busiest-state` thử một cách thu nhỏ bảng Q trên Cologne1. Bốn hướng vào được mã hóa bằng bốn dấu hiệu có xe dừng và một mã hướng có queue dài nhất. Mã 0 nghĩa là cả bốn hướng đều trống. Nếu nhiều hướng cùng dài nhất, lấy hướng đứng đầu theo thứ tự edge trong model. Các bin đầu ra, pha hiện tại và mã hướng quá hạn 300 giây vẫn nằm trong khóa. Lệnh evaluate đọc phiên bản `overdue-busiest-v1` từ model.

## Đọc kết quả

Train ghi `training.csv` và Q-table theo episode. Các lượt qua `src/evaluate.py` dùng chung cách tính metric và lưu `result.json`, `metadata.json`, `actions.csv`, `signals.csv`, XML cùng log SUMO. Dùng `result.json` để xem các chỉ số sau.

| Chỉ số | Cách đo |
|---|---|
| Average Waiting Time, Travel Time, Time Loss | Giây/xe trên nhóm xe hoàn thành hợp lệ. Waiting Time và Time Loss lấy từ SUMO. Travel Time bằng thời điểm đến trừ thời điểm xe thực sự xuất phát. |
| Average Queue Length | Số xe dừng trung bình theo thời gian trên các làn vào được đo, tính cả lúc vàng và đỏ. |
| Throughput, Completion Rate | Số xe hoàn thành hợp lệ quy đổi ra xe/giờ, và tỷ lệ số xe đó trên tổng lượt xe yêu cầu trong khoảng mô phỏng. |
| Teleport | `teleports` đếm sự kiện bắt đầu teleport. `teleported` đếm số xe khác nhau từng teleport. |

Chuyến hợp lệ phải đến đích trong thời gian quan sát, không bị xóa và chưa từng teleport. `unfinished` đếm xe đã vào mạng mà chưa đến đích, trong đó `removed_without_arrival` đếm riêng xe đã bị SUMO loại. `undeparted` đếm xe chưa vào mạng. Nếu nhóm xe hoàn thành hợp lệ rỗng, các metric thời gian theo xe có giá trị `null`. Đọc completion và teleport cùng các trung bình thời gian vì mỗi lượt có thể hoàn thành một nhóm xe khác nhau.

Các lượt Cologne1 đã lưu cho thấy cùng seed SUMO vẫn có thể cho kết quả khác nhau. Khi đánh giá, giữ số liệu từng lượt lặp và báo độ biến thiên. [Bằng chứng Stage 6.1](reports/stage6_1/README.md)

## Luồng đọc mã nguồn

1. Mở [môi trường SUMO](src/sumo_env.py) và đọc `SumoEnv.get_state()` để thấy hàng đợi vào, hàng đợi ra và pha hiện tại. Đọc `step()` và `_run_signal()` để thấy TraCI thực thi đèn, lấy mẫu theo giây và tính reward từ hàng đợi vào.
2. Mở [Fixed-Time](src/ft.py) và đọc `fixed_time()` để thấy thứ tự pha xanh. Mỗi lượt xanh đầy đủ kéo dài 30 giây theo đồng hồ SUMO, với vàng và đỏ tính riêng. Mốc kết thúc mô phỏng có thể cắt lượt cuối.
3. Mở [Max-Pressure](src/mp.py) và đọc `calculate_pressures()` rồi `choose_action()`. Điểm pha là tổng chênh lệch queue vào và ra có trọng số trên các movement được cấp xanh ưu tiên. Trọng số chia đều theo các làn đích của từng làn vào.
4. Mở [Q-Learning](src/q_learning.py) và đọc `encode_state()`, `choose_action()`, `update_q_value()` rồi `run_episode()` để theo dõi rời rạc hóa, epsilon-greedy và cập nhật Bellman. Khi kết thúc episode, target chỉ gồm reward.
5. Mở [evaluation](src/evaluate.py) để đọc `greedy_steps()` và `run_evaluation()`, rồi xem `trip_metrics()` và `signal_metrics()` trong [mã tính metric](src/evaluation_metrics.py).

[Hướng dẫn thuyết trình](LEARNING_GUIDE.md) đi sâu vào từng hàm. [Mục lục báo cáo](reports/README.md) dẫn tới dữ liệu và các lượt thực nghiệm đã lưu. Công cụ phân tích nằm trong `tools/`.
