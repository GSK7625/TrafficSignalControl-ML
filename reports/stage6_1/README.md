# Stage 6.1 — kiểm chứng độ tin cậy — 15/11/2026

**Metrics đối chiếu được với raw; tính tái lập của SUMO 1.28.0 chưa đạt.**
Đã tìm được bước đầu tiên khác quỹ đạo, tái hiện cả hai sự kiện an toàn và
khoanh vùng sai khác về phía thực thi SUMO. Chưa xác định dòng lỗi C++ cụ thể,
chưa tuyên bố đã sửa tính không xác định hoặc nguyên nhân gốc của hai sự kiện.

Không sửa sáu file production `sumo_env.py`, `evaluate.py`,
`evaluation_metrics.py`, `ft.py`, `mp.py`, `q_learning.py`. SHA của **832 file**
Stage 1–6 và data giữ nguyên, gồm cả model Stage 5 và mọi kết quả Stage 6.
Không train, DQN, sửa network/timing/seed, thêm dependency, commit hoặc push.

## 1. Hiện trạng đã xác nhận

Đã đọc AGENTS.md; sáu file production trên; `test_evaluation.py`, các test
môi trường/tín hiệu liên quan; Stage 6 README, PLAN, provenance,
safety_analysis, reproducibility, observer_control, observer_repeat, diagnostic.
Git ban đầu sạch. Sáu file production khớp SHA bàn giao trong provenance Stage 6.

- Python `.venv`: 3.11.9; SUMO, TraCI, sumolib 1.28.0; pytest 9.2.0;
  NumPy 2.4.6; Matplotlib 3.11.2. TraCI/sumolib thực được import từ
  `C:/Program Files (x86)/SUMO/tools/`; metadata ghi cả đường dẫn module.
- Binary thực: `C:/Program Files (x86)/SUMO/bin/sumo.exe`, SHA-256
  `35d45c1166ceb5b72cce37676216f79124fcfe05920a496b14daee19acdaf9f3`,
  khớp provenance Stage 6.
- Config gốc: Cologne1 [25200,28800), Ingolstadt1 [57600,61200), step 1 s.
  Probe dừng sớm qua `env.end`/đóng TraCI; không sửa end trong sumocfg gốc.
- FT xanh 30 s; vàng Cologne1 8 s/Ingolstadt1 6 s; đỏ tối thiểu 2 s,
  thêm clearance theo cơ chế cũ. Không chuyển trực tiếp xanh xung đột.
- Stage 6 FT Cologne1 seed42: **1873 chuyến hợp lệ, 272036 queue-s**;
  observer_repeat: **1899, 269654**. Bốn probe cũ đúng là
  **51644, 51664, 51644, 51644**. Diagnostic True/False đã lưu có positions
  và departures giống nhau; điều đó không xóa được sai khác ở lượt khác.
- Baseline `.venv`: **142 passed / 12.80 s**. Python hệ thống chỉ đạt
  139 passed, 3 failed vì thiếu Matplotlib; không sửa code để né lỗi môi trường.

## 2. Thí nghiệm tái lập và output

Tất cả lượt bên dưới: FT, Cologne1, seed42, 700 s, cùng binary/config/net/route,
không `--random`, không đổi tham số xe hay thuật toán. Mỗi thư mục có
`command.txt` và `metadata.json.argv` ghi argv Popen thực, gồm remote-port.
Đường dẫn output và cổng TraCI được cấp riêng từng tiến trình; đây là khác
biệt vận hành được công bố, không phải thay seed hoặc điều khiển.

| Nhóm | Thư mục trong reproducibility/ | Queue-seconds |
|---|---|---|
| A: không tripinfo, observer chuẩn | 0_A, 1_A, 2_A | 51664, 51664, 51644 |
| A: không thêm observer | 5_A, 6_A, 7_A | 51664, 51664, 51644 |
| A: đảo thứ tự đọc lane/subscribe xe mới | 8_A | 51664 |
| B: tripinfo, observer | 3_B, B_repeat_0, B_repeat_1 | 51664, 51664, 51644 |
| C: tripinfo + CSV metrics, observer | 4_C, C_repeat_0, C_repeat_1 | 51644, 51644, 51664 |

Observer chuẩn subscribe lane/position/speed/acceleration/xy/angle cho xe mới,
đọc tập xe đang hoạt động, queue từng lane, teleport và collision sau từng
bước. Hash sử dụng float đầy đủ, ID được sắp xếp; không so thứ tự dictionary.
Chỉ giữ chi tiết toàn xe quanh [25830,25870] ở năm lượt đầu. Không có setter
cho xe, không thêm simulationStep. Đảo thứ tự chỉ kiểm tra lane reads và
thứ tự subscription cho departures, không phải mọi hoán vị API có thể có.

A/B vẫn giữ sampling và clearance vốn có của SumoEnv, không ghi thêm CSV
metrics; C ghi cùng các cột signals.csv của runner chính. Metrics XML/CSV
được tính **sau khi SUMO đóng**. Không loại sampling an toàn khỏi môi trường
để tạo một baseline giả có cơ chế clearance khác.

Kết quả có đúng hai trace SHA cho các lượt 700 s được quan sát. Mỗi nhánh
đều xuất hiện trong A, B, C; các trace cùng nhánh khớp từng bước, không chỉ
khớp queue tổng. Không phát hiện thay đổi lịch đèn trong cả 700 s giữa hai
nhánh. Các mẫu có/không observer cũng khớp hoàn toàn khi rơi vào cùng nhánh.

**Kết luận trong phạm vi thử:** tripinfo, recorder CSV và observer bổ sung
không phải điều kiện cần để sai khác xuất hiện; không có bằng chứng chúng
tạo sai lịch điều khiển. Chưa chứng minh tính thụ động tuyệt đối cho mọi
output/seed/map; vẫn có thể có tác động bố trí bộ nhớ trong simulator không
xác định. Một lượt B khác một lượt A không phải bằng chứng nhân quả.

## 3. Bước khác nhau đầu tiên và điều tra tiếp

So `0_A` với `2_A`, theo đồng hồ **sau bước TraCI**:

| Thành phần | Thời điểm khác đầu tiên |
|---|---:|
| Timestamp / tín hiệu đèn | không khác trong 700 s |
| Trạng thái xe (sáu trường subscribe) | **25854** |
| Departed IDs / active IDs | 25859 |
| Queue theo lane | 25874 |
| Teleport / collision | không có ở các probe này |

Tất cả trường quan sát khớp đến 25853. Sai khác đầu tiên xảy ra trong bước
**[25853,25854)**, chỉ ở xe `142128_415_0` vào
`:cluster_357187_359543_13_0`, movement rẽ trái index13. Action0 vẫn có xanh
ở index10–14; cùng timestamp, departures và tập xe đang hoạt động.

| Giá trị tại TraCI 25854 | 0_A | 2_A |
|---|---:|---:|
| Lane position (m) | 1.0457200692672046 | 1.8301295573440441 |
| Speed (m/s) | 10.052142789202104 | 10.836552277278948 |
| Acceleration (m/s²) | 1.4792590611469603 | 2.2636685492238033 |

Sai khác sớm nhất nhìn được trong CSV Stage 6 là [25856,25857), ở
`junction_before`; CSV không chứa quỹ đạo nên không thể suy đó là bước gốc.
`foes_0..2` (660 s) xác nhận lại bước 25854 khi đọc thêm leader/nextTLS/foes
cho xe liên quan. Getter junction-foes trả rỗng trong vùng đó: không dùng
kết quả này làm bằng chứng rằng không có xe xung đột hình học.

Đã tiếp tục điều tra bằng hai phép tách biệt:

1. **Native replay `native_0..7`, mỗi lượt 660 s:** tái dựng đúng các setter
   đèn từ samples, kể cả reset, từng đoạn và đỏ kéo dài; gọi simulationStep
   với cùng lịch cố định. Không dùng SumoEnv, đọc queue, trạng thái xe hay
   recorder Python. SUMO tự ghi FCD nén, chỉ từ 25845. Bốn lượt có speed
   10.836552277279 và bốn lượt 10.052142789202 tại FCD 25853. Sai khác đầu
   trong cửa sổ FCD ở 25853, tương ứng TraCI 25854. FCD dùng timestamp đầu
   bước, không phải có thêm một giây lệch điều khiển.
2. **Native replay `rng_0..3`:** thêm output state tại 25853, precision17,
   lưu RNG. Toàn bộ XML root state, gồm rngState và vehicle/lane/route/TLS,
   giống hệt giữa bốn lượt; bước sau vẫn có cả hai nhánh tốc độ. Đây là
   trạng thái SUMO **đã xuất**, không phải ảnh chụp toàn bộ bộ nhớ tiến trình.

`initial_0..4` chỉ chạy 1 s: `lane.getInternalFoes()` cho thấy thứ tự hai
foe bổ sung ở lane `_21_0` hoặc `_25_0` có thể khác ngay sau reset, trước
khi xe chạy. `initial_trajectory_0..3` (660 s) cho thấy cùng thứ tự công khai
vẫn có thể đi vào hai nhánh. Vì vậy không kết luận thứ tự này là nguyên nhân.

[Source SUMO v1_28_0 MSLink.cpp](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/MSLink.cpp)
dùng tập cặp con trỏ khi recheck/khởi tạo các conflict bổ sung. Đây là hướng
điều tra có cơ sở cho trạng thái khởi tạo phụ thuộc địa chỉ, **chưa phải lỗi
được chứng minh**. Các getter distance và queue đã xem source không có
setter xe/đèn. SHA các source đã đọc và phân biệt bằng chứng/giả thuyết nằm
trong [upstream_review.json](upstream_review.json).

Đã chứng minh sai khác quỹ đạo tồn tại ở phía SUMO với cùng lịch đèn và cả
không có observer/metrics Python. Chưa chỉ ra private field, phép tính hay
dòng C++ gây khác. Cần debug build của chính bản này, ghi conflict geometry,
leader candidates và thứ tự xử lý tại 25853; không đổi seed/network/timing
để ép khớp. Không tự nâng cấp binary hoặc coi một issue upstream tương tự là
đã giải thích được ca này. SUMO cũng mô tả kỳ vọng lặp cùng arguments/input
trong [tài liệu Randomness](https://sumo.dlr.de/docs/Simulation/Randomness.html).

## 4. Collision Ingolstadt1 — đã tái hiện, nguyên nhân gốc chưa chốt

`collision_analysis/observed`: FT seed2026, 550 s, nhận sự kiện qua TraCI
ở 58135; XML ghi **58134**, đúng hai xe và tọa độ Stage 6. Cả **550 samples**
khớp Stage 6, gồm queue, signal, clearance, teleport và collision.
Xem [analysis.json](collision_analysis/analysis.json) và raw collisions.xml.

- Collider `carIn2369:1`: route `653473569#5 → 164051413 → 124812857#0`;
  movement3, đi thẳng từ lane `653473569#5_1`, link state M.
- Victim `carIn72005:1`: route `25149219#1 → 391891458#0 → -653473569#5`;
  movement2, rẽ trái từ đường nhánh, link state m.
- Junction `cluster_1526094852_194342371` là **priority**, không phải TLS.
  Request2 response `11001`: nhường index0,3,4; movement3 có response00000.
  Hai movement2/3 là foes. Thứ tự bit SUMO đọc index0 từ bên phải.
- Trước bước va chạm, tại TraCI58134 collider ở pos71.7484 m, v4.6129 m/s;
  victim dừng trên internal `_2_0`, pos3.3654 m, ít nhất từ58110 đến58134.
- Trong bước [58134,58135), victim bắt đầu chạy 2.1251 m/s; collider đi vào
  `_3_0` với 5.6577 m/s. XML ghi chồng lấn hình xe tại x≈212968.7,y≈451449.2,
  rồi teleport collider. Đây là junction collision, không phải chỉ một cảnh
  báo queue hoặc việc đếm hai lần một xe.
- **Tác động gián tiếp từ TLS có dấu vết thực:** edge164051413 chỉ dài8.93 m
  nối priority junction đến TLS. Xe trước collider `h7703c2:3` dừng tại
  pos1.9113 m trên edge này đến58133. TLS mở movement3 tại58132, xe đó bắt
  đầu rời hàng ở58134; collider tăng tốc và victim cũng bắt đầu chuyển động.
  Queue gần TLS ảnh hưởng khoảng trống cho collider, dù va chạm ở nút khác.

Đã xác nhận cấu hình ưu tiên, hình học và chuỗi chuyển động dẫn tới chồng
lấn. Chưa chứng minh vì sao mô hình junction cho phép victim/collider cùng
tiến vào vùng va chạm ở bước đó; chưa có can thiệp nhân quả để quy lỗi cho
FT, keep-clear, network hoặc bước1 s. Không sửa bất kỳ thành phần nào.
Định nghĩa junction collision bằng chồng lấn hình xe được mô tả trong
[SUMO Safety](https://sumo.dlr.de/docs/Simulation/Safety.html).

## 5. Emergency braking Cologne1 — tái hiện ở một trong hai lượt

FT seed2028, 1640 s: `observed` không có sự kiện; `repeat_0` tái hiện đúng
**174028_427_0, 26823, internal_13_0**, không collision/teleport. Cả **1640
samples của repeat_0** khớp Stage 6. Không bỏ lượt không tái hiện.
Xem [analysis.json](emergency_braking_analysis/analysis.json).

- Trip gốc depart26429; route `28198821#3 → 32038051#0`, rẽ trái index13,
  đi qua internal13 dài8.76 m rồi internal24 dài19.77 m.
- Tại TraCI26823: lane13,pos6.63785 m,v12.00002 m/s. Sau bước sự kiện
  (TraCI26824): lane24,pos0.87787 m,v3.00002 m/s,a=**−9 m/s²**; tiếp đó dừng.
  Đúng emergencyDecel9 của xe, lớn hơn decel thường4.5.
- Action0 xanh liên tục **[26800,26830)**; không có vàng, đổi pha hoặc mở
  xanh khi chưa clear tại sự kiện. Warning ghi lane trước khi xe sang24.
- Xe cùng route phía trước `172772_426_0` có gap30.86 m tại26823 và đang đi
  xa dần. Xe gần `170713_426_0` đổi internal11_0→11_1 trong [26822,26823),
  sát vùng tách đường thẳng/rẽ trái; `215436_445_0` cũng trên internal11_0.

Đã loại được giải thích trực tiếp “đèn vừa đổi đỏ buộc phanh” trong ca đã
lưu. Tương tác với xe đi thẳng/đổi lane nội bộ là giả thuyết cần kiểm tra
bằng leader/conflict diagnostics trong SUMO; getter leader chỉ cho xe theo
route chưa đủ để chứng minh đó là xe gây phanh. Không đồng nhất emergency
braking với collision; không điều chỉnh vàng/đỏ/xanh để loại warning.

## 6. Metrics và kiểm thử

`verify.py` tính độc lập từ XML/CSV, không gọi trip_metrics/signal_metrics:
**36 lượt đạt** (30 Stage 6 + 6 C mới). Kiểm tra requested, inserted, arrived,
valid completed, unfinished, undeparted, teleport ID/events, ba trung bình,
queue, throughput và completion. Các assert chi tiết ở verification.json.

- Requested đọc trip/vehicle có depart dự kiến trong [begin,end); không dùng
  inserted làm mẫu số. Valid=arrived không vaporized, chưa từng teleport.
- Waiting/timeLoss lấy XML, travel=arrival−actual depart, cùng tập valid;
  xe chưa xong/chưa chèn không nhận travel0, không được đưa vào trung bình.
  Không cộng departDelay. Không có valid thì trung bình null.
- Teleports là số lần bắt đầu, teleported là số ID duy nhất. Test mới raw
  XML/CSV có 2 events cùng ID, 1 xe valid, 1 unfinished, 1 undeparted và
  arrival đúng end; đối chiếu cả statistics và các công thức tính tay.
- Queue là số xe dừng trên incoming lane duy nhất, cuối từng giây, gồm mọi
  stage. Test cũ kiểm tra lane đã dedup; probe còn so tổng queue từng lane
  với từng dòng CSV, không cộng theo controlled-link bị trùng.
- Queue mean đơn vị xe; waiting/travel/loss đơn vị giây/xe valid;
  throughput=valid×3600/horizon, completion=valid/requested.
- Ở probe collision550 s, request `h3063c2:1` depart58149.8 nằm trong mẫu số
  nhưng chưa đến bước chèn khi dừng58150 (bước cuối bắt đầu58149). Có 1
  undeparted dù SUMO waiting0. Đây là khác phạm vi nhóm, không thất lạc xe.
  Trong 30 full-hour runs, các partition vẫn khớp như Stage 6 đã báo cáo.

Test baseline **142 passed**; kiểm tra cuối **147 passed / 14.33 s**
(xem [tests_final.log](tests_final.log)).
Không phát hiện lỗi production cần sửa. Thay đổi gồm script probe, test mới,
thêm một test metrics vào test_evaluation.py và bằng chứng Stage6.1.
Review độc lập script/test không thấy lỗi chặn; đã bổ sung lưu `watched` trong metadata
(test RED→GREEN) và kiểm tra tái dựng setter với clearance/horizon cắt ngang,
cùng một replay SUMO47 s trong test. Các lượt cũ trong thư mục này chưa có
trường watched: danh sách dùng cho safety là hai ID collision/một ID braking;
foes_0..2 dùng 142128_415_0,138240_412_0,164298_422_0; các lượt khác không có.
Lượt review độc lập bổ sung cho toàn báo cáo không hoàn tất do giới hạn sử
dụng của reviewer; báo cáo đã được tác giả đối chiếu lại với raw và verifier.

## 7. Chạy lại và vị trí code để thuyết trình

Các lệnh probe phải dùng **thư mục mới**; script từ chối ghi đè:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/stage6_1_probe.py --map cologne1 --seed 42 --horizon 700 --mode A --observe --detail-window 25830 25870 --output reports/stage6_1/new_A
.\.venv\Scripts\python.exe tools/stage6_1_probe.py --map ingolstadt1 --seed 2026 --horizon 550 --mode C --observe --detail-window 58110 58145 --watched carIn2369:1 carIn72005:1 --output reports/stage6_1/new_collision
.\.venv\Scripts\python.exe tools/stage6_1_probe.py --map cologne1 --seed 2028 --horizon 1640 --mode C --observe --detail-window 26785 26840 --watched 174028_427_0 --output reports/stage6_1/new_braking
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'tools'); from stage6_1_probe import replay_signals; replay_signals('reports/stage6_1/reproducibility/foes_0','reports/stage6_1/reproducibility/new_native',save_rng=True)"
```

Lệnh replay đọc đường dẫn tuyệt đối và SHA của SUMO, config và input từ lượt
`foes_0` đã lưu. Để chạy trên máy có đường dẫn khác, tạo một lượt probe
Cologne1 660 giây mới rồi dùng thư mục của lượt đó làm nguồn replay.

`reports/stage6_1/verify.py` là công cụ kiểm chứng lịch sử. Nó yêu cầu mã
nguồn đúng phiên bản Stage 6.1 và ghi vào `verification.json` trong cùng thư
mục. Xem [kết quả đã lưu](verification.json) để đối chiếu.

Đổi mode B/C trong lệnh đầu để kiểm tra output; bỏ --observe để đối chứng.
Metadata/argv và log console giữ chính xác các lượt đã chạy. Tổng thực tế:
**40 tiến trình ngắn**, 25475 giây mô phỏng cộng dồn; gồm 13 lượt A/B/C700 s,
12 native replay660 s, 5 init1 s, 7 chẩn đoán660 s và 3 lượt safety.
Không chạy lại 30 lượt full-hour Stage 6.

| Kiến thức | File / hàm |
|---|---|
| Input TLS/lane/action, reset/seed/command | sumo_env.py: SumoEnv.__init__, reset |
| FT chọn action cố định | ft.py: fixed_time |
| Lệnh SUMO, vàng/đỏ/clearance | sumo_env.py: step, _run_signal, _junction_vehicles |
| State và reward −queue_seconds/elapsed | sumo_env.py: get_state, step |
| Observer/hash từng giây | stage6_1_probe.py: run_probe / observed_step |
| Bước khác đầu và native replay | stage6_1_probe.py: first_difference, signal_commands, replay_signals |
| Phép đo chính, phân loại xe | evaluation_metrics.py: requested_trips, trip_metrics, signal_metrics, read_metrics |
| Recorder chính, đóng SUMO rồi tính metrics | evaluate.py: run_evaluation |
| Đối chiếu độc lập và bảo vệ kết quả cũ | reports/stage6_1/verify.py: raw_metric_check, main |

## 8. Có tiếp tục tối ưu Q-Learning được chưa?

**Chưa đủ để dùng thay đổi nhỏ giữa các lần chạy làm bằng chứng cải thiện
QL.** Bộ metric đã có bằng chứng nhất quán; phần chưa đạt là tính lặp của
simulator và nguyên nhân gốc hai sự kiện. Có thể phân tích policy/bảng kết
quả đã đóng băng, nhưng không coi seed là đã kiểm soát hết biến thiên.

Phép thử tiếp theo: debug native replay giữ nguyên seed42/lịch đèn và so
conflict/leader/private state quanh25853, sau đó xác nhận trên hai safety
replay. Chỉ khi hiểu/kiểm soát được sai khác mới thiết kế đánh giá cải thiện
QL. Chưa huấn luyện hoặc tối ưu QL trong nhiệm vụ này.
