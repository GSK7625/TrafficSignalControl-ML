# Giai đoạn 3 — Fixed-Time — 15/11/2026

Đã triển khai FT trên môi trường dùng chung sau sửa chữa tối thiểu giai đoạn 2.
Sáu lượt FT, mỗi lượt 3600 s, ba seed trên mỗi bản đồ: **0 collision, 0 teleport,
0 emergency braking, 0 emergency stop**. Đây là kết quả quan sát trong các lượt
đã chạy, không phải chứng nhận an toàn tuyệt đối hay cho giao thông thực.

Đã đọc AGENTS.md, hai báo cáo trước, code/test/smoke và các bằng chứng va chạm.
Git ban đầu sạch, nhánh main. Không sửa dữ liệu gốc, cài gói, train, commit/push,
triển khai MP/QL/DQN hoặc so chất lượng thuật toán. Python .venv 3.11.9,
SUMO/TraCI/sumolib 1.28.0; chạy headless trên Windows.

## 1. Nguyên nhân và cách sửa môi trường

Giai đoạn 2 có vàng, đỏ và clearance đuôi xe nhưng vẫn có 11 collision Cologne1:
tất cả ở xanh action 2 `GGGggrrrrrGGGggrrrrr`. Bảy sự kiện có collider trên
internal lane kết thúc `_1_1`, ba trên `_13_0`, một trên `_11_1`.
Trace ở `../stage2/cologne_collision_probe/trace.json` cho thấy xe đi thẳng
link 2 (G) đâm xe rẽ trái link 13 (g) đang chờ ngay trong vùng giao cắt, lúc
25599 s. Xe rẽ trái vào nút trong chính pha 2; không phải xe sót từ pha trước.
Vì vậy chỉ tăng all-red giữa các pha không ngăn được cơ chế này.

Ingolstadt1 smoke cũ không collision nhưng trace giai đoạn 1 lúc 58594 s ghi nhận
bus `60R.42`, đi thẳng link 7 (G), va ô tô `carIn86966:1` rẽ trái link 2 (g),
cùng ở pha 0. Chưa phân tách được đóng góp sâu hơn của hình học, điểm chờ,
mô hình nhường và bước thời gian; không tuyên bố đã sửa hình học của mạng.

Phương án runtime: **giữ G, đổi toàn bộ g/s thành r** trong các pha action.
Không biến g thành G và không thêm quan hệ ưu tiên mới. Không xóa action.
Xe rẽ/chuyển động trước đây được nhường vẫn đi ở pha đã có G bảo vệ khác.
Quy tắc nằm trong `SumoEnv.__init__`, áp dụng chung cho FT và mọi thuật toán
sau này dùng môi trường này. ID vẫn là chỉ số pha gốc; ý nghĩa chuỗi tín hiệu
đã thay đổi và không được trộn kết quả với môi trường giai đoạn 2.

| Map / action | Chuỗi xanh runtime | Links có G |
|---|---|---|
| Cologne1 / 0 | `rrrrrrrrrrGGGGGrrrrr` | 10–14 |
| Cologne1 / 1 | `GGGGGrrrrrrrrrrrrrrr` | 0–4 |
| Cologne1 / 2 | `GGGrrrrrrrGGGrrrrrrr` | 0–2, 10–12 |
| Cologne1 / 3 | `rrrGGrrrrrrrrGGrrrrr` | 3,4,13,14 |
| Cologne1 / 4 | `rrrrrGGGGGrrrrrrrrrr` | 5–9 |
| Cologne1 / 5 | `rrrrrrrrrrrrrrrGGGGG` | 15–19 |
| Cologne1 / 6 | `rrrrrGGGrrrrrrrGGGrr` | 5–7,15–17 |
| Cologne1 / 7 | `rrrrrrrrGGrrrrrrrrGG` | 8,9,18,19 |
| Ingolstadt1 / 0 | `GGrGrGGG` | 0,1,3,5,6,7 |
| Ingolstadt1 / 2 | `GGGrrrrr` | 0,1,2 |
| Ingolstadt1 / 4 | `rrrGGGrr` | 3,4,5 |

Đã kiểm tra ma trận foes trên **tất cả** các pha trên: không có cặp foes cùng
mở G. Ánh xạ TLS index/request index được đối chiếu qua sumolib và TraCI.
Môi trường từ chối nếu có foes cùng mở hoặc một movement không còn pha G.
Foes chỉ là kiểm tra theo khai báo mạng, không chứng minh mọi thân xe đều tránh nhau.

Tác động từng thay đổi g/s (số là linkIndex, không phải lane index):

| Map / action bị đổi | Link đóng | Vẫn có G trong action |
|---|---|---|
| Cologne1 / 0 | 0 (s) | 1,2 |
| Cologne1 / 1 | 10 (s) | 0,2 |
| Cologne1 / 2 | 3,4 (g) | 1,3 |
| Cologne1 / 2 | 13,14 (g) | 0,3 |
| Cologne1 / 4 | 15 (s) | 5,6 |
| Cologne1 / 5 | 5 (s) | 4,6 |
| Cologne1 / 6 | 8,9 (g) | 4,7 |
| Cologne1 / 6 | 18,19 (g) | 5,7 |
| Ingolstadt1 / 0 | 2 (g) | 2 |

`network_audit.json` lưu toàn bộ 20/8 movement: incoming/outgoing/via, các pha
G thay thế, thay đổi từng ký tự, tốc độ, độ dài internal, foes gốc và thông số
các vType thực sự được nhu cầu sử dụng. Không mất connection/đường đi; thời gian
được phép đi của các luồng nhường giảm, có thể tăng chờ và chặn đầu hàng trên
làn nhiều hướng. Không suy từ coverage rằng mọi xe chắc chắn được phục vụ
trước end hoặc rằng lịch bất kỳ sẽ tránh được ùn tắc.

## 2. Vàng, all-red và điều kiện giải phóng

Đổi action: đóng mọi G của pha nguồn bằng y, giữ vàng cố định, chuyển toàn đỏ
ít nhất 2 s, kiểm tra toàn bộ internal và đuôi xe rồi mới mở pha đích. Đóng cả
link nguồn/đích dùng chung. Cùng action: giữ xanh, không chèn vàng giả. Vàng và
đỏ là chuyển tiếp do môi trường quyết định, không phải action cho agent.

| Thông số thiết kế | Cologne1 | Ingolstadt1 |
|---|---:|---:|
| Tốc độ đường lớn nhất, m/s (cả mạng ngoài internal) | 19.44 | 13.89 |
| Bao tốc độ thiết kế: 2 × tốc độ đường, m/s | 38.88 | 27.78 |
| Decel nhỏ nhất của các vType được dùng, m/s² | 4.5 | 4.0 (bus) |
| Giảm tốc dùng khi chọn vàng, m/s² | 3 | 3 |
| Vàng runtime mới, s | 8 | 6 |
| Đỏ nền, s | 2 | 2 |
| Internal path dài nhất, m | 33.54 | 26.06 |
| Chiều dài xe lớn nhất, m | 4.3 | 12 |

Chọn vàng bảo thủ, cố định trước khi chạy: `ceil(1 + v_design / (2 * 3))`,
1 s phản ứng tương ứng bước mô phỏng. Bao hệ số tốc độ 2 dựa trên cận trên
phân bố mặc định `normc(1,0.1,0.2,2)` của passenger; đầu vào hiện tại không có
ghi đè speedFactor/departSpeed. Bus hiện tại có speedDev=0. Dùng decel=3 nhỏ
hơn decel của mọi vType được dùng, không dựa vào emergencyDecel. Đây là lựa
chọn thiết kế có bao dự phòng cho hai bộ dữ liệu này, **không phải công thức
sinh vàng mặc định của SUMO** hoặc hiệu chuẩn ngoài thực địa. Phải kiểm tra lại
nếu đổi route/vType/speed distribution/tốc độ đường/step-length.

Không dùng quãng đường / tốc độ tối đa để kết luận đỏ 2 s luôn đủ: xe có thể
đi chậm hoặc bị chặn. `_junction_vehicles()` giữ nguyên cơ chế theo dõi internal
và odometer tới khi đuôi thoát nút, kể cả bus 12 m vượt edge ra 8.93 m.
Nếu còn xe thì kéo đỏ từng giây, có thể tới hết episode; không có timeout mở
xanh cưỡng bức. Trong hai smoke mới và sáu FT mới, đỏ thực tế đều 2 s; test nhỏ
vẫn kiểm tra kéo đỏ, nút bị chặn và bus còn đuôi trong nút.

Đã đối chiếu tham số thực qua TraCI trong `network_audit.json`, phép tính ở
`timing_audit.json`, tín hiệu/thời gian thực ở CSV và cảnh báo ở log. Không
ghi nhận emergency braking/stop trong các lượt mới. Chưa chứng minh bao thiết
kế này đủ cho mọi hình học hoặc mọi sự kiện giữa hai mẫu 1 s.

Tham khảo: [tín hiệu và ưu tiên SUMO](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html),
[kiểm tra va chạm và an toàn](https://sumo.dlr.de/docs/Simulation/Safety.html),
[phân bố speedFactor](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#speed_distributions).

## 3. Cổng kiểm tra môi trường trước FT và teleport còn lại

Chạy lại script smoke có sẵn, bật `collision.check-junctions=true`, seed 23423,
đủ 3600 s/map. Chuỗi đi mọi cặp action có thứ tự, sau đó chọn ngẫu nhiên;
**đây không phải FT**. Không đổi lịch smoke để giảm số liệu xấu.

| Smoke runtime mới | Cologne1 | Ingolstadt1 |
|---|---:|---:|
| Cặp action đã kiểm tra | 64/64 | 9/9 |
| Lời gọi step | 93 | 101 |
| Collisions | 0 | 0 |
| Teleports | 36 | 2 |
| Phân loại SUMO | 32 yield, 3 jam, 1 wrongLane | 2 yield |
| Emergency braking / stop | 0 / 0 | 0 / 0 |
| Warning lines / thiếu vàng lúc nạp | 78 / 6 | 4 / 0 |

Các teleport mới đều là cảnh báo xe chờ quá lâu trên **làn ngoài nút**, không
phải collision teleport. SUMO mặc định `time-to-teleport=300`, giữ nguyên;
nhãn `yield` không có nghĩa chuỗi TLS runtime vẫn còn g/s. Không coi teleport
là cách giải phóng an toàn hoặc bằng chứng không mất movement.

Ví dụ Cologne1: xe `123965_406_0` bị teleport lúc 25545 s trên
`-32038056#3_0`; từ sau pha 0 đầu episode đến thời điểm này lịch smoke chưa
phục vụ action 1/2 cần cho hướng đi thẳng của làn đó. Ingolstadt1: `h6245c2:1`
bị teleport lúc 58175 s trên `164051413_2`, khi action 4 chưa có xanh hơn
300 s; xe `h1956c1:1` gặp khoảng thiếu pha 4 dài khác lúc 60103 s.
Đây là bằng chứng lịch chẩn đoán để hướng xe chờ lâu. Với các sự kiện còn lại,
log xác nhận chờ yield/jam/wrong lane; chưa trace từng xe để tách định lượng
đói pha, thiếu năng lực thông hành và chặn làn. Không quy tất cả 38 sự kiện
cho một nguyên nhân đã chứng minh.

`smoke_teleport_analysis.json` giữ đủ 38 sự kiện, lane/reason/thời gian/stage và
khoảng từ lần G gần nhất của bất kỳ movement trên lane (không phải waitingTime
của xe). `environment_smoke/` giữ toàn bộ log/CSV/XML, cả reset probe.

Quyết định tiếp tục FT: cơ chế va chạm giữa các luồng cùng xanh đã được loại
khỏi tập runtime theo foes; tất cả movement còn G, mọi cặp chuyển tiếp qua
vàng/đỏ, không có collision/phanh/dừng khẩn cấp trong smoke. Teleport do chờ
của lịch smoke vẫn là giới hạn thật cần công bố, nhưng không có bằng chứng
ở các lượt này về xe bị kẹt vĩnh viễn do mất action hoặc mở xanh khi chưa clear.
FT dùng lịch tuần hoàn đã được giao để kiểm tra riêng; không tối ưu lịch theo xe.

## 4. FT chọn pha và kiểm tra 30 s

`fixed_time(env)` chỉ tăng `action_index` tuần hoàn trong `env.actions`:
Cologne1 `0→1→2→3→4→5→6→7→0`, Ingolstadt1 `0→2→4→0`.
Không đọc queue/reward để chọn action, không train và không gọi TraCI trực tiếp.
`run_ft()` tạo/reset môi trường rồi tiêu thụ vòng lặp trên, xuất log, cuối cùng
đóng trong finally kể cả khi chạy lỗi. Không ghi đè thư mục kết quả đã tồn tại.

Mỗi `env.step()` chạy riêng 30 s xanh sau chuyển tiếp. `_run_signal()` lấy
`simulation.getTime()` trước/sau mỗi bước và đọc lại tín hiệu từ SUMO, kiểm tra
dt=1, chuỗi khớp và không vượt end; không chỉ tin `green_seconds=30`.
`signals.csv` ghi tín hiệu TraCI thực tế ở cuối bước, đại diện [start,end).
`test_recorded_sumo_signals_match_ft_schedule()` đọc lại từng CSV để kiểm tra
chuỗi, thứ tự xanh/vàng/đỏ, cộng thời gian thực, coverage 3600 mẫu và đối chiếu
`actions.csv`/`result.json`. Đây là kiểm tra tín hiệu đọc qua TraCI, không phải
một bộ ghi TLS XML độc lập thứ hai. Số collision còn đối chiếu với XML SUMO.

## 5. Kết quả FT thực tế

Mọi lượt bật kiểm tra collision trong nút, giữ collision.action=teleport và
time-to-teleport mặc định; không tắt teleport để đổi số liệu. Khoảng giờ giữ
nguyên cấu hình: Cologne1 [25200,28800), Ingolstadt1 [57600,61200).

| Map | Seed | Thời gian s | Calls | Xanh s | Vàng s | Đỏ s | Collision | Teleport | E. braking | E. stop | Warning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cologne1 | 23423 | 3600 | 91 | 2700 | 720 | 180 | 0 | 0 | 0 | 0 | 6 |
| Cologne1 | 42 | 3600 | 91 | 2700 | 720 | 180 | 0 | 0 | 0 | 0 | 6 |
| Cologne1 | 12345 | 3600 | 91 | 2700 | 720 | 180 | 0 | 0 | 0 | 0 | 6 |
| Ingolstadt1 | 23423 | 3600 | 95 | 2848 | 564 | 188 | 0 | 0 | 0 | 0 | 0 |
| Ingolstadt1 | 42 | 3600 | 95 | 2848 | 564 | 188 | 0 | 0 | 0 | 0 | 0 |
| Ingolstadt1 | 12345 | 3600 | 95 | 2848 | 564 | 188 | 0 | 0 | 0 | 0 | 0 |

Cologne1 có **90 lượt xanh × 30 s**. Call thứ 91 yêu cầu action 2 nhưng episode
kết thúc sau 8 vàng + 2 đỏ, chưa mở xanh. Lượt xanh cuối thực sự là action 1,
[28760,28790); không báo nhầm call cuối thành lượt xanh 30 s.
Ingolstadt1 có **94 lượt × 30 s + lượt cuối 28 s**, action 2 xanh
[61172,61200). Không vượt end, không kết thúc sớm vì tạm hết xe.

Sáu warning Cologne1 mỗi lượt phát khi SUMO nạp chương trình gốc `program 0`,
trước runtime do TraCI điều khiển. Không xóa/ẩn chúng. Log actual signals xác
nhận mọi lần đổi action runtime đều qua vàng rồi all-red. Không có warning
khác ở các lượt FT này. Ingolstadt1 không warning.

Seed thực sự đổi mô phỏng (FT không có RNG riêng). Theo thứ tự 23423/42/12345:
Cologne1 inserted=1976/1970/1951, running=73/71/69, waiting chưa chèn=39/45/64;
Ingolstadt1 inserted=1715/1708/1709, running=26/28/28, waiting chưa chèn=1/8/7.
Queue-seconds cũng khác: Cologne1 266303/269654/270265; Ingolstadt1
33344/32841/33463. File summary lưu hash của CSV. Nhu cầu OD/thời điểm yêu cầu
khởi hành vẫn cố định; seed tạo biến thiên hành vi/chèn xe, không tạo bộ nhu cầu
mới hay chứng minh tính độc lập thống kê của ba lượt.

Queue trong CSV là tổng `getLastStepHaltingNumber` trên tập incoming lane duy
nhất, xe tốc độ <0.1 m/s, đơn vị xe; lấy cuối mỗi bước 1 s. Queue-seconds là
tích phân rời rạc, đơn vị xe·s; gồm xe chưa đến đích nếu đang trên lane đo,
không gồm xe chưa chèn/internal/ùn trên edge trước. Reward = âm queue trung
bình theo thời gian trong toàn call, gồm vàng/đỏ, chỉ ghi để dùng chung môi
trường; FT bỏ qua. Chưa tính Average Waiting Time, Travel Time, Time Loss hay
Throughput; không dùng mẫu xe hoàn thành để suy cho toàn bộ nhu cầu.

## 6. Tệp, cách chạy và vị trí thuyết trình

Sửa `sumo_env.py`, `test_sumo_env.py`. Thêm `ft.py`, `test_ft.py` và thư mục
`reports/stage3/`. Giữ nguyên `smoke_test.py` và báo cáo/dữ liệu giai đoạn trước.
`PLAN.md` ghi phạm vi; `source_hashes_before.json` và `source_hashes_after.json`
xác nhận SHA-256 cả 8 tệp nguồn (gồm LICENSE) không đổi; smoke cũng có hash riêng.
`ft_summary.csv` tổng hợp từng lượt. Mỗi `ft/<map>/seed_<seed>/` chứa command,
metadata, actions/signals CSV, result JSON, sumo/warnings log, statistics và
collisions XML. Không có lượt chạy xấu nào bị xóa.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe smoke_test.py --output reports/stage3/environment_smoke_repeat
.\.venv\Scripts\python.exe ft.py cologne1 --seed 23423 --output reports/stage3/ft_repeat/cologne1
.\.venv\Scripts\python.exe ft.py ingolstadt1 --seed 23423 --output reports/stage3/ft_repeat/ingolstadt1
```

Các lượt bàn giao chạy `python ft.py <map>` với seed mặc định 23423 và thêm
`--seed 42`, `--seed 12345`; mặc định lưu vào `reports/stage3/ft/<map>/seed_<seed>`.
Đường dẫn đã tồn tại sẽ bị từ chối; dùng đường dẫn mới khi chạy lại. Pytest
kiểm tra các lượt nằm dưới `ft/*/seed_*`; muốn kiểm tra lượt mới bằng test CSV
này, đặt `--output` theo cùng mẫu đó. Phiên chạy không thấy SUMO thì đặt
SUMO_HOME/PATH/PYTHONPATH như hướng dẫn giai đoạn 2, không cần cài gói mới.

| Nội dung thuyết trình | Hàm |
|---|---|
| Đọc cấu hình, mapping, tập action, kiểm tra foes/coverage, thời gian | `SumoEnv.__init__` |
| Khởi chạy/reset và nhận input queue | `SumoEnv.reset`, `SumoEnv.get_state` |
| Công thức FT: `action_index = (action_index + 1) % len(env.actions)` | `ft.fixed_time` |
| Chọn action và gửi sang môi trường | `ft.fixed_time`: `env.step(action)` |
| Tạo vàng, clearance gồm đuôi xe, chuyển tiếp | `yellow_state`, `SumoEnv._junction_vehicles`, `SumoEnv.step` |
| SUMO nhận lệnh, đồng hồ/tín hiệu thực tế | `SumoEnv._run_signal`: `setRedYellowGreenState`, `simulationStep` |
| Reward và thời gian từng đoạn | cuối `SumoEnv.step` |
| Vòng đời episode, xuất kết quả và đối chiếu số liệu | `ft.run_ft` |
| Thứ tự cố định dù queue đổi, cắt horizon | các test trong `test_ft.py` |
| Kiểm tra 30 s từ log thực | `test_recorded_sumo_signals_match_ft_schedule` |

## 7. Kiểm thử và điều kiện giai đoạn 4

Baseline 13 test passed. Đã thấy hai test runtime thất bại trước sửa g/s;
sau sửa 15 passed. Test FT ban đầu không import được vì chưa có ft.py; sau
triển khai 20 passed, 1 skipped do chưa có dữ liệu FT. Sau sáu lượt đủ giờ:
**26 passed**, không skip; gồm sáu test kiểm tra lại CSV thực tế, test lịch
không đổi khi queue biến thiên, kết thúc ở vàng/đỏ/xanh, cùng action, reward
tính tay, action sai, đỏ kéo dài và bus-tail. Không chạy train.

Rà soát độc lập chỉ đọc đã kiểm tra code, 26 test, hash nguồn, raw XML và CSV;
không phát hiện lỗi cần sửa. Reviewer không chạy thêm SUMO hoặc kiểm tra GUI.

**Đủ điều kiện bắt đầu giai đoạn 4 trong phạm vi học tập trên hai dữ liệu này**,
dùng đúng runtime action/chuyển tiếp/reward/metric chung đã ghi ở trên.
Chưa triển khai MP hay quyết định cách phân bổ queue làn nhiều movement.
MP vẫn phải kiểm tra riêng lịch chọn pha: môi trường không đảm bảo lịch bất
kỳ tránh đói pha/teleport; số liệu smoke 36/2 là cảnh báo thực tế cho giới hạn đó.

Chưa kiểm tra GUI, bước thời gian nhỏ hơn 1 s, mọi hình học thân xe, nhu cầu
khác, seed ngoài ba seed FT, hay trace chi tiết từng teleport smoke. Không dùng
vài lượt không va chạm để tuyên bố an toàn tuyệt đối. Vàng 8/6 s có thể bảo thủ
và tăng chờ; không hiệu chỉnh sau chạy để làm đẹp kết quả. Chưa so sánh chất
lượng FT với bất kỳ thuật toán nào.
