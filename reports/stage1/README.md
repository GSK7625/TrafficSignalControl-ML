# Khảo sát Giai đoạn 1 — 15/11/2026

Đã đọc `AGENTS.md`. Git ban đầu sạch, nhánh `main`; repository chỉ có dữ liệu, `AGENTS.md`, `requirements.txt`, `.gitignore`, chưa có code điều khiển. Có `.venv` và ba thư mục dữ liệu; chỉ khảo sát `cologne1`, `ingolstadt1`. Không sửa dữ liệu, cài thư viện, train, commit hoặc push. Các tệp mới đều là báo cáo/đầu ra trong `reports/stage1/`.

## Môi trường đã kiểm tra

- Python hệ thống và `.venv`: **3.11.9**, Windows x64. Nên dùng `.\.venv\Scripts\python.exe`.
- `SUMO_HOME=C:\Program Files (x86)\SUMO\`; `sumo.exe`, `sumo-gui.exe`: **1.28.0**, chạy `--version` thành công, mã thoát 0. Chưa mở giao diện mô phỏng SUMO-GUI.
- Đã import thật trong `.venv`: sumolib/TraCI 1.28.0, NumPy 2.4.6, Matplotlib 3.11.2, pytest 9.2.0. sumolib/TraCI thực tế được nạp từ `SUMO\tools`, không phải từ thư mục package trong `.venv`. Không có PyTorch.
- Python hệ thống tìm được sumolib/TraCI nhưng không tìm được NumPy/Matplotlib; không nên dùng nhầm interpreter.
- Đã kết nối TraCI tới SUMO trên từng bản đồ, lấy TLS/controlled links, đối chiếu toàn bộ mapping với XML, tiến một bước. Giao thức 22; `deltaT=1 s`; đồng hồ tăng đúng 1 s.
- Chi tiết: `environment.json`, `inspection.json`; lệnh chạy SUMO: `commands.txt`. Các lần chạy không truyền seed: giữ mặc định **23423**, đã xác nhận từ chính executable bằng `sumo --save-template reports/stage1/sumo_defaults.xml`. Tệp defaults cũng xác nhận step-length=1, collision.check-junctions=false, collision.action=teleport.

## Hai bản đồ và nhu cầu xe

| Thuộc tính | Cologne1 | Ingolstadt1 |
|---|---|---|
| Khoảng cấu hình [begin,end), giây | [25200,28800), 07:00–08:00 | [57600,61200), 16:00–17:00 |
| Thời gian chạy | 3600 s | 3600 s |
| TLS ID | `cluster_357187_359543` | `cluster_274083968_cluster_1200364014_1200364088` |
| Chương trình | static, programID=0, offset=0 | static, programID=0, offset=0 |
| Hướng tiếp cận / làn xe vào / làn xe ra được TLS điều khiển | 4 / 8 / 8 | 3 / 7 / 6 |
| Controlled links / request indices | 20 / 0–19 | 8 / 0–7 |
| Pha xanh ứng viên / pha có vàng / tất cả đỏ | 8 / 0 / 0 | 3 / 3 / 0 |
| Chu kỳ gốc đo thực tế | 40 s | 90 s |
| Số chuyến khai báo | 2015 | 1716 |
| Loại phương tiện | 2015 passenger (`pkw`) | 1699 passenger + 17 bus |
| Loại khai báo trong rou.xml | 1 vType, 2015 trip | 45 vType, 1716 trip; chỉ 8 vType được dùng |
| Khởi hành yêu cầu đầu/cuối | 25205 / 28799 s | 57600.2 / 61198 s |

Cả hai `sumocfg` chỉ tham chiếu net/route và begin/end, không đặt step-length hay seed. Route chứa **trip theo OD**, không có flow/vehicle/route cố định; SUMO tìm đường khi chạy. Depart đã sắp thứ tự, không có chuyến ngoài cửa sổ. Thời điểm yêu cầu lẻ ở Ingolstadt1 được thực thi theo bước 1 s, không đồng nghĩa thời điểm xe thật sự vào mạng.

Số chuyến theo 12 cửa sổ 5 phút liên tiếp, tính từ begin:

- Cologne1: **192, 224, 130, 158, 231, 191, 134, 130, 180, 143, 154, 148**.
- Ingolstadt1: **135, 106, 165, 168, 147, 121, 130, 173, 169, 140, 148, 114**.

Phân bố nguồn chuyến, theo **edge xuất phát**, không được nhầm với làn sát TLS:

- Cologne1: `-32038056#3`: 572; `23429231#1`: 688; `28198821#3`: 438; `27115123#2`: 204; `130165204`: 112; `32324544#0`: 1.
- Ingolstadt1: `201963537#1`: 620; `104010354`: 463; `653473569#5`: 421; `25149219#1`: 212.

Có chuyến cùng edge đi/đến: Cologne1 có 3 chuyến `130165204` và 1 chuyến `32324544#0`; Ingolstadt1 có 1 chuyến `201963537#1`. Không tự loại chúng. Bảng OD, đích, vType và độ dài/tốc độ từng làn nằm trong `inspection.json`.

## Làn và controlled links

Cologne1: mỗi edge vào `-32038056#3`, `23429231#1`, `28198821#3`, `27115123#3` có làn `_0`, `_1`. Các edge ra `32038051#0`, `-28198821#4`, `32324544#0`, `32038056#0` cũng có `_0`, `_1`. Không có làn vào nằm ngoài mapping TLS.

Ingolstadt1:

- Vào: `201963537#1_1`, `_2`, `_3`; `164051413_1`, `_2`; `104010354_1`, `_2`.
- Ra: `104010475#0_1`, `_2`; `-164051413_1`; `124812857#0_1`, `_2`, `_3`.
- `incLanes` còn có `201963537#1_0`, `164051413_0`, `104010354_0`: **chỉ cho pedestrian**, không phải ba làn xe bị bỏ sót. Dữ liệu nhu cầu không có person.

Mapping đầy đủ từng linkIndex → lane vào → lane ra → via → hướng rẽ có trong `cologne1/controlled_links.csv` và `ingolstadt1/controlled_links.csv`. Đã kiểm tra bằng `sumolib` rằng TLS linkIndex trùng junction request index trên **hai mạng này**, rồi xác nhận qua `traci.trafficlight.getControlledLinks`. Không suy rộng quy tắc này cho mạng khác. Các ký tự `state=o/O` trên thẻ connection là quy tắc ưu tiên của connection, không phải lịch pha TLS.

Làn ngắn đáng lưu ý: Cologne1 `27115123#3` dài 41.48 m; Ingolstadt1 `164051413` và `-164051413` dài **8.93 m**, `104010475#0` dài 22.04 m. Queue chỉ đo ngay sát TLS có thể bỏ qua xe ùn lên edge trước. Cologne1 có đường vào 13.89 và 19.44 m/s; Ingolstadt1 các làn xe được khảo sát là 13.89 m/s. Không lấy cùng một công thức thời gian chuyển pha mà bỏ qua khác biệt này.

## Các pha và chuyển tiếp

Chỉ số pha bắt đầu từ 0. `G`: xanh ưu tiên; `g`: xanh nhường; `s`: mũi tên phải, phải dừng rồi nhường; **s không phải vàng**. `y`: vàng, `r`: đỏ. `minDur/maxDur` dùng cho actuated; chương trình static thực thi `duration`. Tham khảo [tài liệu TLS SUMO](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html).

| Cologne1: pha | Duration | State | Link xanh G/g |
|---|---:|---|---|
| 0 | 5 s | `srrrrrrrrrGGGGGrrrrr` | 10–14; thêm s ở 0 |
| 1 | 5 s | `GGGGGrrrrrsrrrrrrrrr` | 0–4; thêm s ở 10 |
| 2 | 5 s | `GGGggrrrrrGGGggrrrrr` | 0–4, 10–14 |
| 3 | 5 s | `rrrGGrrrrrrrrGGrrrrr` | 3,4,13,14 |
| 4 | 5 s | `rrrrrGGGGGrrrrrsrrrr` | 5–9; thêm s ở 15 |
| 5 | 5 s | `rrrrrsrrrrrrrrrGGGGG` | 15–19; thêm s ở 5 |
| 6 | 5 s | `rrrrrGGGggrrrrrGGGgg` | 5–9,15–19 |
| 7 | 5 s | `rrrrrrrrGGrrrrrrrrGG` | 8,9,18,19 |

**Cả tám pha 5 s của Cologne1 đều là pha có xanh, không phải pha vàng.** Có `minDur=5,maxDur=50` nhưng chương trình static không tự điều chỉnh theo nhu cầu. `tls_states.xml` và `observed_phases.csv` ghi nhận 720 đoạn pha, đều dài 5 s, chạy 0→1→…→7→0. Ví dụ 25200–25205 là pha 0, 25205–25210 pha 1. Sáu chuyển tiếp có G/g/s→r không qua y: **0→1, 2→3, 3→4, 4→5, 6→7, 7→0**. 1→2 và 5→6 cũng đổi quan hệ ưu tiên G/g; không được mặc định là chuyển tiếp không cần kiểm tra.

| Ingolstadt1: pha | Duration | State | Vai trò |
|---|---:|---|---|
| 0 | 38 s | `GGgGrGGG` | Xanh, có nhường ở link 2 |
| 1 | 3 s | `yygyryyy` | Chuyển tiếp có vàng; link 2 vẫn g, không phải action |
| 2 | 6 s | `GGGrrrrr` | Xanh, bảo vệ link 2 |
| 3 | 3 s | `yyyrrrrr` | Vàng |
| 4 | 37 s | `rrrGGGrr` | Xanh |
| 5 | 3 s | `rrryyyrr` | Vàng |

Đo 240 đoạn pha, lặp đúng **38/3/6/3/37/3 s**, theo 0→1→2→3→4→5→0. Không thấy xanh→đỏ bỏ qua vàng trong chuỗi gốc. **Không có all-red và chưa thể kết luận an toàn tuyệt đối**; ba pha vàng chỉ áp dụng cho cạnh chuyển tiếp tương ứng. Ví dụ không dùng nguyên pha 1 để chuyển thẳng 0→4 vì còn link 2 xanh.

Đối chiếu ma trận foes: không có cặp **G/G** xung đột trong các pha gốc. Có cặp G/g xung đột ở Cologne1 pha 2/6 và Ingolstadt1 pha 0: đây là chuyển động nhường, không phải các luồng được bảo vệ hoàn toàn. Kiểm tra ma trận không chứng minh tránh mọi va chạm hình học, nhất là lúc chuyển pha hoặc xe đang ở trong nút.

## Kết quả chạy thực tế

Hai lượt chạy gốc dùng `sumo -c data/<map>/<map>.sumocfg` kèm tùy chọn xuất log, summary, tripinfo, statistics và thêm một `timedEvent` **chỉ quan sát TLS**. Không thay đổi nhu cầu, tín hiệu hay thời gian mô phỏng. Lệnh đầy đủ có trong `commands.txt`; không xem đây là FT 30 s đã triển khai.

| Kết quả gốc, 1 lượt/mạng | Cologne1 | Ingolstadt1 |
|---|---:|---:|
| Exit code / hết thời gian cấu hình | 0 / có | 0 / có |
| Loaded / inserted | 2015 / 2014 | 1716 / 1715 |
| Đến đích / còn trên mạng / chưa được chèn | 1979 / 35 / 1 | 1694 / 21 / 1 |
| Mean waitingTime, xe đến đích | 29.94 s/xe | 17.53 s/xe |
| Mean duration, xe đến đích | 79.57 s/xe | 48.97 s/xe |
| Mean timeLoss, xe đến đích | 56.77 s/xe | 28.17 s/xe |
| Mean departDelay, xe đến đích | 18.55 s/xe | 2.58 s/xe |
| Mean halting toàn mạng, 3600 mẫu | 16.73 xe | 8.39 xe |
| Collision / teleport, kiểm tra mặc định | 0 / 0 | 0 / 0 |
| Emergency braking / emergency stop | 9 / 5 | 0 / 0 |
| Cảnh báo thiếu vàng | 6 | 0 |

Các trung bình trên được tính lại từ `tripinfo.xml` chỉ với **arrival>=0**; `baseline_metrics.csv` lưu số chính xác. SUMO stdout/statistics của lượt này tính trên cả xe đã vào mạng nhưng chưa hoàn thành vì đã bật `write-unfinished`, nên trung bình có khác. Đã xuất cả xe chưa được chèn; không trộn `depart<0` vào trung bình chuyến. `summary.meanWaitingTime` phản ánh depart delay, không dùng làm Average Waiting Time trên hành trình.

Định nghĩa: waitingTime là thời gian tốc độ ≤0.1 m/s, không tính dừng theo lịch; duration là từ depart thực tế đến arrival, không gồm departDelay; timeLoss là thời gian mất so với vận tốc lý tưởng, không gồm dừng theo lịch. Throughput ở đây là số **đến đích trong cửa sổ** (1979 và 1694 xe/giờ), không phải số qua riêng TLS. Mean halting là trung bình số xe dừng trên **toàn mạng**, không phải queue chỉ trên làn vào TLS. Các đại lượng này và số xe chưa hoàn thành phải được giữ riêng khi đánh giá. Xem [TripInfo SUMO](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html).

### Chẩn đoán bổ sung về an toàn

Chạy riêng cùng cấu hình, thêm `--collision.check-junctions true`. Đầu ra có tiền tố `junction_diagnostic_`; không gộp metric với lượt gốc. SUMO mặc định không kiểm tra toàn bộ va chạm trong nút; tùy chọn này kiểm tra chồng lấn hình xe. Va chạm mặc định dẫn tới teleport. Xem [Safety SUMO](https://sumo.dlr.de/docs/Simulation/Safety.html).

- **Cologne1: 17 collision, 17 teleport, 13 emergency braking, 6 emergency stop; exit 0.** Có cả phanh gấp khi chuyển sang đỏ và va chạm trong nút; chưa quy mọi va chạm cho một nguyên nhân duy nhất.
- **Ingolstadt1: 1 collision, 1 teleport, không emergency braking/stop; exit 0.** Lúc 58594 s: bus `60R.42` trên internal lane kết thúc `_6_1` va với `carIn86966:1`. Đã tái hiện bằng TraCI, ghi `collision_reproduction.xml`, `collision_trace.json` và log. TLS đang ở **pha 0 `GGgGrGGG`**, không phải lúc đổi pha: bus đi thẳng link 7 (G), ô tô rẽ trái link 2 (g), là cặp foes. Đây là bằng chứng không thể chỉ thêm vàng rồi khẳng định đã giải quyết an toàn. Chưa kết luận nguyên nhân sâu là hình học mạng, mô hình nhường, bước thời gian hay phối hợp các yếu tố.

## Đề xuất cho giai đoạn tiếp theo — chưa triển khai

1. **Action:** dùng các pha xanh làm ứng viên: Cologne1 0–7, Ingolstadt1 0/2/4. Không dùng pha 1/3/5 của Ingolstadt1 làm action, dù pha 1 còn một đèn g. Các ứng viên có g/s chưa được chứng nhận an toàn. Có thể khảo sát phương án chỉ giữ G, đổi g/s thành r trong chương trình runtime riêng để bỏ chuyển động nhường; không sửa net gốc. Đây là một thay đổi cần kiểm chứng và ghi rõ, không phải kết quả đã chạy.
2. **Chuyển pha:** cùng pha thì giữ xanh; đổi pha phải có chuyển tiếp theo cặp nguồn–đích, xử lý cả thay đổi G↔g, rồi khoảng đỏ giải phóng nút trước xanh mới. Cologne1 cần bổ sung chuyển tiếp hoàn toàn; Ingolstadt1 có thể kế thừa pha vàng cho đúng chuỗi gốc, không áp dụng mù quáng khi nhảy pha. Khởi điểm khảo sát: vàng **4 s Cologne1, 3 s Ingolstadt1**, all-red **ít nhất 2 s**, có kiểm tra xe còn trong vùng xung đột. Các con số này là giả định thử nghiệm, **chưa chốt hoặc chứng minh đủ an toàn**, nhất là Cologne1 có đường 19.44 m/s và Ingolstadt1 có va chạm ngay trong pha xanh. Không dùng nguyên chuỗi gốc làm cơ chế an toàn chung.
3. **Queue thô:** `traci.lane.getLastStepHaltingNumber(lane)` trên tập làn xe vào/ra duy nhất. Tổng mỗi lane một lần, loại pedestrian và internal khỏi queue đo ở biên; internal vẫn phải được quan sát riêng cho an toàn. Giữ mapping movement vào→ra cho MP; với một làn phục vụ nhiều movement phải chốt cách phân bổ/trọng số, không cộng lặp queue của làn cho từng link rồi gọi đó là tổng queue.
4. **State dễ học:** queue vào và ra tổng theo mỗi edge được điều khiển + pha xanh hiện tại: Cologne1 4+4 queue; Ingolstadt1 3+3 queue. Q-Learning rời rạc hóa mỗi queue thành `0`, `1–4`, `>=5` và lưu pha; DQN dùng đúng các queue đó chia 10 và one-hot pha. Ngưỡng/chia 10 chỉ là khởi điểm, chưa được chọn qua train. Q-table tối đa 8×3^8=52488 state và 3×3^6=2187 state trước xét trạng thái không tới được. Tổng theo edge làm mất thông tin hàng đợi theo hướng rẽ và bỏ qua ùn trên edge trước; cần trình bày rõ giới hạn.
5. **Reward:** âm của queue vào trung bình theo thời gian trong toàn bước điều khiển: `r = -sum_t(sum_unique_in_lanes q_l(t) * dt) / elapsed_seconds`. Đơn vị xe; tính cả vàng/đỏ, không chỉ lấy mẫu cuối. QL/DQN dùng chung định nghĩa; không dùng metric chỉ của xe đã đến đích làm reward tức thời.
6. **Thời gian:** khởi đầu đơn giản là 30 s xanh mỗi lần chọn action cho FT/MP/QL/DQN; vàng/đỏ tính riêng. FT mỗi pha xanh đúng 30 s, độc lập các duration gốc 5 hoặc 6/38/37 s. Giữ cùng pha thì không chèn vàng giả. Vì vậy khoảng giữa hai quyết định có thể là 30 s hoặc 30 s cộng chuyển tiếp; phải ghi elapsed thực tế, reward bình quân như trên và nói rõ gamma tính theo mỗi quyết định nếu dùng công thức QL/DQN chuẩn. Không giới thiệu đây là điều khiển cách đúng 30 s theo đồng hồ. Episode kết thúc đúng end; bước cuối có thể bị cắt ngắn.

Chưa viết môi trường hay thuật toán; chưa có các hàm input/action/update/SUMO/reward để chỉ vào khi thuyết trình. Hiện có thể thuyết trình từ ba tệp dữ liệu của mỗi map, bảng mapping và log pha đã đo.

## Đã kiểm tra và còn thiếu

Đã kiểm tra XML, nhu cầu/OD, ma trận foes và mapping request, import thư viện, TraCI một bước trên cả hai map, hai lượt gốc đủ một giờ, hai lượt chẩn đoán va chạm đủ một giờ, một lượt TraCI tái hiện va chạm Ingolstadt1 đến 58597 s. Không có lỗi làm dừng mô phỏng; các cảnh báo an toàn nêu trên là lỗi/giới hạn thực tế cần xử lý, không được bỏ qua vì exit code 0.

Chưa kiểm chứng: giao diện SUMO-GUI khi render/chạy; cơ chế chuyển pha mới; độ đủ của vàng/all-red; nguyên nhân sâu và cách khắc phục va chạm; tác động bước thời gian; độ biến thiên nhiều seed; tính công bằng giữa thuật toán; PyTorch/DQN; test FT/MP/QL/DQN. Chỉ chạy một seed mặc định, không so chất lượng giữa hai map khác nhu cầu. Không chạy train.

SHA-256 trước/sau của 8 tệp trong hai thư mục dữ liệu (gồm LICENSE) giống nhau; bằng chứng ở `source_sha256_before.json`, `source_sha256_after.json`. Git không có thay đổi tracked file sau khảo sát; chỉ có thư mục báo cáo mới.
