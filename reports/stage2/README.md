# Giai đoạn 2 — môi trường SUMO dùng chung — 15/11/2026

**Trạng thái: prototype. Chức năng môi trường đã chạy và được kiểm tra; an toàn giao thông chưa được giải quyết hoàn chỉnh. Chưa đủ điều kiện chốt giai đoạn 2 để bắt đầu giai đoạn 3 (Fixed-Time).**

Không sửa `data/`, không cài dependency, không train, không triển khai FT/MP/QL/DQN, không commit/push. Git lúc bắt đầu sạch, nhánh `main`. Thay đổi chỉ gồm ba tệp Python ở gốc và đầu ra/báo cáo trong thư mục này.

## Căn cứ và thiết kế trước khi viết code

Đã đọc toàn bộ `AGENTS.md`, `reports/stage1/README.md`, `environment.json`, parse `inspection.json`, hai bảng `controlled_links.csv`, log chẩn đoán và trace Ingolstadt1; đối chiếu với net/route/config gốc. Dùng lại khảo sát giai đoạn 1, không tạo lại toàn bộ khảo sát. `source_audit.json` lưu đối chiếu cấu hình, pha, mapping và SHA-256 của cả 8 tệp nguồn.

Thiết kế tối thiểu: một class `SumoEnv` dùng chung, đọc mapping từ XML rồi xác nhận lại qua TraCI; một tệp kiểm tra logic và một script chạy chẩn đoán. Giữ nguyên tín hiệu của các pha xanh gốc. Khi đổi action: đóng mọi luồng đang mở bằng vàng, tất cả đỏ tối thiểu 2 s, chờ xe và đuôi xe ra hết nút rồi mở pha đích. Cùng action: tiếp tục xanh 30 s. Reward lấy queue ở cuối từng bước SUMO 1 s và tính trung bình trên toàn khoảng action.

Giả định thử nghiệm: vàng Cologne1 4 s, Ingolstadt1 3 s; đỏ nền 2 s và kéo dài theo quan sát. Đây **không phải các giá trị đã được hiệu chuẩn theo vận tốc, giảm tốc, hình học và mọi loại xe**. Chọn cách dừng toàn bộ trước mỗi lần đổi pha, kể cả G↔g, để không phải giả định hai pha cùng mở một movement có thể đổi ưu tiên ngay. Kiểm tra dự kiến và đã thực hiện: ví dụ reward tính tay, cắt episode ở các đoạn chuyển tiếp, reset, theo dõi từng giây, mọi cặp action và hai lượt đủ một giờ bật kiểm tra va chạm trong nút.

## Chạy trên Windows

Trong thư mục gốc dự án, dùng môi trường SUMO đã khảo sát ở giai đoạn 1:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe smoke_test.py
```

`smoke_test.py` mặc định tạo `reports/stage2/smoke/`. Đầu ra đã tồn tại thì dừng để giữ bằng chứng; muốn chạy lại:

```powershell
.\.venv\Scripts\python.exe smoke_test.py --output reports/stage2/smoke_repeat
```

Nếu terminal mới chưa có SUMO trong môi trường:

```powershell
$env:SUMO_HOME = 'C:\Program Files (x86)\SUMO'
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"
```

Không cần cài thêm gói. Đã chạy thực tế bằng Python 3.11.9 trong `.venv`, SUMO/TraCI/sumolib 1.28.0. Không dùng GUI.

Ví dụ API, không phải một thuật toán điều khiển:

```python
from sumo_env import SumoEnv

env = SumoEnv("ingolstadt1")
try:
    state = env.reset()
    state, reward, done, info = env.step(2)
    print(state, reward, done, info["elapsed"])
finally:
    env.close()
```

`output_dir` tùy chọn của môi trường dùng để xuất log SUMO; reset cùng đối tượng dùng lại đường dẫn log đó, chỉ giữ log episode sau cùng. Script smoke tách riêng probe reset và lượt đủ một giờ. Không gọi `simulationStep` hoặc thay tín hiệu từ bên ngoài khi dùng môi trường.

## Action và tính hợp lệ của pha

Theo [SUMO Traffic Lights](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions): `G` xanh ưu tiên; `g` xanh phải nhường; `s` mũi tên phải, phải dừng rồi nhường; `y` vàng, xe đủ xa phanh còn xe gần có thể đi; `r` đỏ. `s` không phải vàng. `dir="s"` trong connection là hướng đi thẳng, khác ký tự tín hiệu `s`.

Action ID là **chỉ số pha gốc**, không phải số thứ tự trong một danh sách nén:

| Mạng | Action hợp lệ theo cấu trúc | Xanh có quan hệ nhường/xung đột |
|---|---|---|
| Cologne1 | 0,1,2,3,4,5,6,7 | 0/1/4/5 có s; 2/6 có g và các cặp G/g foes |
| Ingolstadt1 | 0,2,4 | 0 có g ở link 2, nhường links 5/6/7 |

Các chuỗi xanh được giữ nguyên từng ký tự trong `metadata.json`. Các action này **hợp lệ về mapping và quy tắc ưu tiên, chưa được chứng nhận không va chạm**; Cologne1 action 2 có bằng chứng không an toàn. Không đưa pha vàng 1/3/5 của Ingolstadt1 vào action dù pha 1 có một đèn g. Không có lựa chọn vàng/tất cả đỏ cho agent.

`__init__` đối chiếu TLS index với junction request index bằng sumolib trên hai mạng, rồi đọc bitstrings `foes/response` từ phải sang trái (index 0 ở bên phải). Không có cặp G/G foes; mọi cặp đang mở là foes đều phải có một nhánh `g/s` và bit response tương ứng yêu cầu nhường. Có 8 cặp G/g ở mỗi pha 2/6 Cologne1 và 3 cặp ở pha 0 Ingolstadt1. Chi tiết từng cặp ở `smoke/<map>/metadata.json`. Kiểm tra này không chứng minh hình học thân xe không chồng lấn.

## Chuyển pha và giới hạn an toàn

Đổi A→B luôn tạo vàng từ **A**, không sao chép một pha vàng Ingolstadt1 cho mọi cặp: toàn bộ `G/g/s` của A thành `y`, các tín hiệu đỏ giữ `r`; sau đó tất cả `r`; sau khi giải phóng nút mới đưa nguyên chuỗi xanh B. Các đèn dùng chung giữa A/B cũng đóng để giải phóng xe trước khi thay quyền ưu tiên. Do đó vàng có thể giống nhau cho nhiều đích nhưng chỉ vì tất cả các luồng của cùng nguồn đều được đóng. Không đổi g/s thành r trong **pha xanh action**.

Việc s→y trong chuyển tiếp tạm thay quy tắc dừng rồi nhường bằng quy tắc vàng; đây cũng là một giả định cần kiểm chứng, không phải tuyên bố mọi quan hệ nhường còn nguyên trong vàng. Chưa thấy va chạm trong vàng của lượt đã chạy.

Đỏ tối thiểu 2 s **không tự mở xanh khi hết giờ**. `_junction_vehicles` kiểm tra các internal lane thuộc đúng TLS, gồm đoạn sau vị trí chờ nhường. Với xe đã quan sát, odometer và chiều dài xe xác định khi nào đuôi ra khỏi đoạn internal cuối; theo dõi tiếp kể cả đầu xe đã vượt edge ra ngắn. Cũng kiểm tra xe có đầu trên làn ra mà đuôi chưa qua biên. Có test bus 12 m vẫn được giữ khi đầu xe đã vượt edge ra 8,93 m. Quan sát được thực hiện trước mỗi bước 1 s, trên cả xanh/vàng/đỏ. Nếu nút bị chặn thì giữ đỏ đến khi trống hoặc đến end, không có timeout cho phép mở xanh khi còn xe.

Đây là phép kiểm tra giải phóng theo tọa độ dọc làn/mô hình SUMO; chưa phải kiểm tra hình chữ nhật thân xe theo hình học thực. Bước 1 s cũng không chứng minh mọi sự kiện giữa hai mẫu đều được bắt. Đỏ thích ứng không khắc phục va chạm phát sinh giữa các luồng đang cùng xanh.

## State, thời gian và reward

- `reset()` mở process/kết nối mới, kiểm tra begin/step-length/TLS/mapping và đặt xanh action 0 trước bước đầu. Không tiến mô phỏng trong reset. State ban đầu toàn 0 vì chưa thực thi bước chèn xe.
- State là dict: `incoming` tuple số nguyên, `outgoing` tuple số nguyên, `phase` chỉ số pha gốc. Thứ tự là **lane ID sắp xếp từ điển**, công khai ở `incoming_lanes/outgoing_lanes` và metadata. Cologne1 có 8+8 giá trị queue, Ingolstadt1 có 7+6. Giữ dữ liệu theo làn để các thuật toán sau có thể dùng chung; chưa rời rạc hóa hay chuẩn hóa.
- Mỗi làn xuất hiện một lần dù controlled links lặp lại. Lọc/kiểm tra làn cho passenger/bus, không dùng pedestrian/internal làm queue biên. Không cộng queue theo từng movement rồi đếm lại cùng lane. Mapping movement vẫn có trong `links`; chưa tính pressure hay quyết định cách phân bổ queue của làn nhiều hướng.
- Queue là `getLastStepHaltingNumber`, xe tốc độ **<0,1 m/s**, không phải chiều dài vật lý của hàng đợi. Giới hạn: chỉ làn sát TLS, bỏ qua ùn lên edge trước. Xem [Lane Value Retrieval](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html).
- Bước điều khiển lấy mẫu sau từng bước SUMO: mẫu tại t+1 đại diện cho [t,t+1). State tiếp theo lấy sau bước cuối của action. Trước reset chưa có state hợp lệ; sau close phải reset lại.
- `reward = -sum(queue_in(t+dt) * dt) / elapsed`, đơn vị xe. Tính cả xanh/vàng/đỏ và thời gian đỏ kéo dài. `queue_seconds` có đơn vị xe·giây. Ví dụ 4 s queue=2, 2 s queue=4, 30 s queue=1 cho reward `-46/36`; chỉ lấy queue cuối sẽ cho sai thành -1.
- Mỗi action yêu cầu 30 s xanh **ngoài** chuyển tiếp. Hai lần cùng action liên tiếp cho 60 s xanh liên tục. Thời gian quyết định không cố định 30 s; luôn trả `elapsed` thực tế. Chưa có công thức RL hay gamma.
- Cologne1 chạy [25200,28800), Ingolstadt1 [57600,61200). `done` chỉ khi đồng hồ đạt end; không dừng sớm vì tạm hết xe. Mọi đoạn đều bị cắt tại end, không phát bước vượt end. Nếu kết thúc ở vàng/đỏ, `phase=None`; `info.action` vẫn là pha yêu cầu nhưng có thể chưa nhận xanh. Không được gọi `step` sau done.

## Kết quả thực tế

`python -m pytest -q`: **13 passed**. Test logic dùng đồng hồ/queue xác định để kiểm tra reward tính tay, giữ pha, đổi pha, đỏ kéo dài, nút bị chặn, action sai, và cắt tại vàng/đỏ/xanh. Kiểm tra mạng đọc XML thật; smoke sử dụng SUMO thật, không dùng đồng hồ giả.

Smoke dùng một chuỗi chẩn đoán đi qua mọi cặp có thứ tự, gồm cả giữ pha, rồi chọn action bằng `random.Random(23423)` tới end. Đây **không phải FT hay baseline**, và không tối ưu để giảm queue. SUMO cũng dùng seed 23423, step-length 1 s, `--collision.check-junctions true`, collision.action=teleport như giai đoạn 1. `command.txt` trong mỗi thư mục lưu tham số thực tế.

| Kiểm tra/lượt đủ 3600 s | Cologne1 | Ingolstadt1 |
|---|---:|---:|
| Số bước action / cặp đã đi qua | 102 / 64 trên 64 | 107 / 9 trên 9 |
| Xanh, bước không cuối | 30 s | 30 s |
| Vàng, khi đổi pha | 4 s | 3 s |
| Đỏ quan sát, khi đổi pha | 2/3/6/7/8 s | 2/3/5/6 s |
| Số lần kéo đỏ dài hơn 2 s | 6 | 4 |
| Bước cuối | 4 vàng + 2 đỏ + 5 xanh = 11 s | 29 xanh = 29 s |
| Đồng hồ cuối | 28800 s | 61200 s |
| Collisions: giai đoạn 1 → lượt này | 17 → **11** | 1 → **0** |
| Teleports: giai đoạn 1 → lượt này | 17 → **30** | 1 → **0** |
| Emergency braking: giai đoạn 1 → lượt này | 13 → 0 | 0 → 0 |
| Emergency stop: giai đoạn 1 → lượt này | 6 → 0 | 0 → 0 |
| Warning lines / thiếu vàng | 66 / 6 | 0 / 0 |

Sáu cảnh báo thiếu vàng Cologne1 được phát khi SUMO **nạp chương trình gốc**, trước khi TraCI đặt chương trình online. Không xóa/ẩn cảnh báo; `signals.csv` ghi chuỗi runtime có vàng. Không suy từ cảnh báo nạp chương trình rằng runtime đã bỏ vàng, cũng không dùng exit 0 làm bằng chứng an toàn.

30 teleport Cologne1 gồm 11 collision + 17 yield + 2 jam. Không loại bỏ các lượt/số liệu xấu. Lượt này có loaded/inserted/running/waiting = 2015/1892/60/123; Ingolstadt1 = 1716/1701/42/15. Vì lịch tín hiệu khác giai đoạn 1, không diễn giải chênh lệch các con số trên như kết quả so sánh hiệu năng thuật toán.

Toàn bộ 11 collision Cologne1 xảy ra trong **xanh action 2** `GGGggrrrrrGGGggrrrrr`. Không ghi nhận collision ở vàng/đỏ trong hai lượt; không có emergency braking/stop. `collision_context.json` lưu từng sự kiện TraCI, đối chiếu số đếm với statistics.xml; `collisions.xml` là đầu ra SUMO độc lập, có thời điểm và hình học.

Đã chạy lại cùng tiền tố chuỗi action để ghi trace sự kiện đầu ở `cologne_collision_probe/trace.json`: lúc 25599 s (TraCI đọc sau bước tại 25600 s), xe `137244_412_0` đi thẳng link 2 G va với `136669_411_0` đang dừng trên internal link 13 g, vị trí khoảng 8,66 m. Xanh 2 bắt đầu lúc 25590 s; xe thứ hai vào nhánh nhường trong chính pha này, không phải xe sót từ pha trước. Lúc 25602 s lại có xe đi thẳng đâm cùng xe đang dừng. Trace cho thấy điểm chờ rẽ trái và thân xe giao cắt với luồng ưu tiên; chưa kết luận nguyên nhân sâu chỉ là hình học, vị trí chờ, mô hình nhường hay bước thời gian. Thêm all-red giữa pha không giải quyết được trường hợp này.

Ingolstadt1 một lượt không collision **không xóa bằng chứng** collision bus/link 7 G với link 2 g ở xanh 0 của giai đoạn 1. Cần tái kiểm tra nhiều tình huống/seed và giải quyết cơ chế nhường/hình học trước khi chứng nhận.

## Tệp bằng chứng và các hàm để thuyết trình

- `sumo_env.py`: `__init__` đọc map, action, làn, foes/response; `reset/close` quản lý episode/process; `get_state` lấy input; `yellow_state` tạo vàng; `_junction_vehicles` kiểm tra giải phóng; `_run_signal` gửi lệnh `setRedYellowGreenState` và tiến SUMO; `step` nhận action từ bên ngoài, thực thi chuyển tiếp, tính reward và trả kết quả. Môi trường **không tự chọn** action.
- `test_sumo_env.py`: ví dụ tính tay và các nhánh dễ sai; `smoke_test.py`: reset độc lập, chuỗi action chẩn đoán, assert tín hiệu/thời gian/state/đầu cuối và xuất bằng chứng.
- `smoke/results.json`, `<map>/result.json`: số tổng hợp; `<map>/metadata.json`: thứ tự state, pha, foes, khoảng cách internal đến biên; `signals.csv`: 3600 mẫu từng giây/mạng; `actions.csv`: từng bước, reward, state và thời gian đo thực; `sumo.log`, `warnings.log`, `statistics.xml`, `collisions.xml`: log thô.
- `smoke/source_hashes.json`: hash trước/sau lượt chạy; `source_audit.json`: đối chiếu giai đoạn 1 và nguồn gốc.

Điều kiện để sang giai đoạn 3: xử lý hoặc thống nhất rõ phương án tín hiệu runtime cho những luồng nhường có va chạm, kiểm chứng tác động lên tất cả movement; hiệu chuẩn/kiểm tra thời gian vàng và điều kiện giải phóng theo từng mạng; chạy lại có kiểm tra va chạm và báo cáo cả teleport. Nếu chọn loại g/s hoặc tách pha bảo vệ, đó là thay đổi thiết kế action cần giải thích và kiểm tra riêng, không được âm thầm áp dụng.

Chưa kiểm tra SUMO-GUI, nhiều seed, mọi tình huống hình học/thân xe giữa hai mẫu 1 s và tính đủ của vàng theo thông số xe. Không tuyên bố đã chứng minh an toàn; chưa triển khai giai đoạn 3.
