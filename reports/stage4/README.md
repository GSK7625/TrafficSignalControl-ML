# Giai đoạn 4 — Max-Pressure — 15/11/2026

Đã triển khai và kiểm chứng **MP theo queue làn, phân bổ đều xấp xỉ cho làn
nhiều movement** trên Cologne1/Ingolstadt1. Không gọi đây là MP đo trực tiếp
queue theo movement hoặc saturation flow đã hiệu chuẩn. Công thức, mapping,
chọn action và thực thi đã được kiểm tra; không chứng minh tối ưu thông lượng,
ổn định hàng đợi hay tránh đói pha của MP lý tưởng.

Sáu lượt cuối, mỗi lượt đủ 3600 s: **0 collision, 0 emergency braking,
0 emergency stop**; còn **43 teleport**. Không xóa lượt xấu, đổi seed, tắt
teleport hay điều chỉnh môi trường để cải thiện kết quả. Không kết luận MP
tốt hơn FT từ queue-seconds riêng lẻ.

## 1. Khảo sát và phạm vi thay đổi

Đã đọc AGENTS.md, sumo_env.py/ft.py và test, hai báo cáo giai đoạn 2/3,
network_audit.json, hai controlled_links.csv và kết quả smoke/FT giai đoạn 3.
Đối chiếu cấu hình/net/nhu cầu đang dùng; SumoEnv tiếp tục kiểm tra lại mapping
qua TraCI khi reset. Git ban đầu sạch, nhánh main; 26 test baseline passed.

Chỉ tạo mp.py, test_mp.py và reports/stage4/. Giữ nguyên môi trường, FT,
test cũ, dữ liệu gốc và nhu cầu. source_hashes.json đối chiếu SHA-256 cả 8 tệp
data với reports/stage3/source_hashes_after.json: không đổi. git diff xác nhận
không sửa mã cũ. Không cài gói, train, triển khai QL/DQN, commit hoặc push.
Python .venv 3.11.9, SUMO/TraCI/sumolib 1.28.0; chạy headless trên Windows.

## 2. Công thức và ý nghĩa chính xác của trọng số

Định nghĩa movement là **cặp lane biên** `(i,j)`, không phải một linkIndex
riêng biệt. Mỗi cặp được tính một lần, kể cả khi có nhiều link cùng cặp.

```text
M    = các cặp (incoming_lane, outgoing_lane) duy nhất trong env.links
d_i  = số outgoing_lane khác nhau của làn i trong toàn bộ M
w_ij = 1/d_i
M(a) = các cặp có ít nhất một linkIndex mang G trong env.green_states[a]
P(a) = Σ[(q_i - q_j)/d_i] trên M(a)
a*   = action có P(a) lớn nhất trong env.actions
```

q_i và q_j đều là queue **nguyên làn**, đơn vị xe, từ
getLastStepHaltingNumber; xe có tốc độ dưới 0.1 m/s được coi là dừng theo
[tài liệu SUMO](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html).
Không dùng số xe đang chạy trên làn thay cho số xe dừng.

Trọng số này là cách phân bổ đều theo cấu trúc được chọn trong các phương án
xấp xỉ của yêu cầu. Không có dữ liệu turning ratio/năng lực phục vụ đã đo
trong state hiện tại; **không khẳng định các hướng có cùng lưu lượng hoặc
cùng saturation flow thực tế**. Không đọc route tương lai hoặc bổ sung cảm
biến riêng cho MP. Không diễn giải `q_i/d_i` là số xe đã đo muốn rẽ theo hướng j.

Ví dụ làn có 12 xe, ba hướng: mỗi hướng được phân bổ 4 xe trong mô hình.
Nếu một pha chỉ mở một hướng thì chỉ lấy 1/3 đóng góp; không chia lại toàn
bộ 12 xe cho hướng duy nhất đang xanh. Mẫu số luôn lấy **toàn mapping**.
Vì `Σ_j w_ij = 1`, tổng phần queue đầu vào của cùng làn trên các movement
duy nhất không vượt queue làn. Đây là kiểm tra không nhân queue theo số link,
không phải bằng chứng phân bổ phản ánh đúng ý định của từng xe.

Downstream bị trừ với cùng trọng số: `w_ij*q_j`. Nếu hai làn vào khác nhau
cùng đổ vào làn ra j, queue j tác động lên cả hai chênh lệch pressure; đó là
hai movement khác nhau. Không gọi tổng các hạng downstream trong pressure
là tổng số xe duy nhất của mạng. Tổng queue/reward của môi trường vẫn lấy
tập incoming_lanes duy nhất, hoàn toàn không cộng theo movement.

Ví dụ tay trong test: i_b có hai hướng đến o_y/o_x, i_a có một hướng đến o_y;
queue đầu vào `(12,5)`, đầu ra `(2,8)`:
`P(4)=(12-2)/2+(5-2)=8`, `P(2)=(12-8)/2=2`,
`P(0)=8+2=10`; chọn action 0. Link trùng i_b→o_y không làm tăng P.

Mô hình MP lý tưởng dùng queue theo hướng, tỷ lệ rẽ và khả năng phục vụ;
xem [Lioris, Kurzhanskiy, Varaiya, mục 2](https://connected-corridors.berkeley.edu/sites/default/files/Control%20Experiments%20for%20Signalized%20Intersections%20using%20Q%20Simulator.pdf).
Công thức ở đây là lựa chọn baseline theo làn trong phạm vi yêu cầu, không
phải phép suy ra định lý ổn định của mô hình đó. Làn chung có thể bị xe đầu
hàng chặn; mô hình không biết hướng đi và thứ tự của từng xe, không hiệu
chỉnh khác biệt bus/ô tô, độ dài làn hoặc năng lực thông hành từng hướng.

## 3. Mapping, state, lựa chọn pha và thực thi

Cologne1: 20 link/cặp lane, 8 làn vào/8 làn ra, action 0..7.
Mỗi làn index 0 có hai movement (trọng số 1/2), index 1 có ba (1/3).
Ingolstadt1: 8 link/cặp, 7 làn vào/6 làn ra, action **0,2,4**;
104010354_1 có hai movement (1/2), sáu làn vào còn lại mỗi làn một (1).

Ghép queue bằng `dict(zip(env.incoming_lanes, state['incoming']))` và tương
tự cho outgoing; không tự sắp xếp lại state. SumoEnv đã sắp lane ID và loại
internal/pedestrian khỏi tập queue biên. Kiểm tra độ dài state trước khi zip.
Test mapping đối chiếu CSV khảo sát và green_states của audit giai đoạn 3.

Chỉ ký tự **G runtime** được phục vụ. Không dùng g/s trong xanh gốc, không
coi vàng/đỏ là action. metadata.json mỗi lượt lưu lane order, trọng số,
link_indices và danh sách action của từng cặp để chỉ trực tiếp khi trình bày.

Giữ nguyên pressure âm. Tất cả 0 hoặc nhiều pha bằng nhau: lấy action đầu
trong env.actions, không ưu tiên pha hiện tại hoặc thêm thưởng. Dùng Fraction
của thư viện chuẩn khi cộng để giữ phép hòa toán học; chỉ đổi tổng sang float
khi trả/lưu pressure. Với queue nguyên và mẫu số 1/2/3 ở hai mạng này, không
để sai số 1/3 quyết định tie-break. Reward không tham gia chọn pha.

Không có max-wait hay chống đói pha; thêm chúng sẽ là một biến thể khác.
Sau khi chọn action, max_pressure chỉ gọi env.step(action), lấy state sau
bước đó rồi quyết định tiếp. run_mp reset trước vòng lặp, đóng trong finally
kể cả khi controller lỗi, và từ chối thư mục đầu ra đã tồn tại.

Giữ nguyên cấu hình chung giai đoạn 3: 30 s xanh/lần chọn; vàng Cologne1 8 s,
Ingolstadt1 6 s; đỏ tối thiểu 2 s và chờ clear cả đuôi xe. Cùng action giữ
xanh liên tục, không thêm vàng. Không đổi collision checking, step-length=1 s,
collision.action=teleport hoặc time-to-teleport mặc định 300 s. Horizon có thể
cắt vàng/đỏ/xanh của call cuối; không kéo mô phỏng quá end.

## 4. Kết quả cuối theo seed

Thời gian giữ nguyên: Cologne1 [25200,28800), Ingolstadt1 [57600,61200).

| Map | Seed | Calls | Xanh/vàng/đỏ (s) | Collision | Teleport | Warning | Queue-seconds (xe·s) |
|---|---:|---:|---|---:|---:|---:|---:|
| Cologne1 | 23423 | 91 | 2710/712/178 | 0 | 15 | 36 | 110116 |
| Cologne1 | 42 | 91 | 2710/712/178 | 0 | 9 | 24 | 93207 |
| Cologne1 | 12345 | 91 | 2710/712/178 | 0 | 16 | 38 | 108655 |
| Ingolstadt1 | 23423 | 95 | 2848/564/188 | 0 | 3 | 6 | 19350 |
| Ingolstadt1 | 42 | 95 | 2848/564/188 | 0 | 0 | 0 | 20064 |
| Ingolstadt1 | 12345 | 95 | 2848/564/188 | 0 | 0 | 0 | 20364 |

Cả sáu lượt không có emergency braking/stop. Đỏ mọi lần chuyển pha thực tế
là 2 s, không có lần phải kéo dài; logic kéo đỏ vẫn được test cũ kiểm tra.
Cologne1 có 90 call đầy đủ 30 s xanh và call cuối 10 s xanh; Ingolstadt1 có
94 call 30 s và call cuối 28 s. Call cuối đều chọn action 0, có chuyển tiếp.

Cologne1 chọn action 0 hai lần đầu liên tiếp, xanh 60 s [25200,25260), hai
lần quyết định cách nhau 30 s và không có vàng giữa chúng. Đây là chuỗi lặp
dài nhất của cả ba seed; Ingolstadt1 không có action lặp liên tiếp. Không
ghi nhận một pha giữ xanh hàng trăm giây trong các lượt này.

Tuy nhiên khoảng **không có G của một movement** có thể rất dài dù mỗi call
chọn pha khác: lớn nhất Cologne1 730/770/690 s, Ingolstadt1 350/312/312 s theo
seed 23423/42/12345. Khoảng này tính cả biên horizon và khi có thể chưa có
xe, không phải waiting time đã đo của một xe. Các action 3/7 Cologne1 ít
hoặc không được chọn; không tự kết luận mất service vì movement của chúng
cũng có G trong action khác. Chi tiết coverage/interval từng cặp trong analysis.json.

**Mọi outgoing queue tại các thời điểm quyết định của sáu lượt đều bằng 0.**
Vì vậy các lượt thực này chưa kiểm chứng phản ứng với tắc downstream khác 0;
test tay có queue ra lớn/âm kiểm tra nhánh đó. Công thức vẫn đọc và trừ
queue ra, không cố định nó bằng 0 trong code. Không sửa mạng để tạo điểm đẹp.

Seed có tạo biến thiên mô phỏng dù MP không có RNG: Cologne1 inserted
2009/2009/2008, running 33/30/29, waiting chưa chèn 6/6/7; Ingolstadt1 inserted
1683/1708/1715, running 26/36/20, waiting chưa chèn 33/8/1. Queue và hash CSV
cũng khác. Nhu cầu OD/thời điểm yêu cầu đi vẫn cố định, không phải ba bộ nhu
cầu độc lập; không dùng ba seed để khẳng định ý nghĩa thống kê.

## 5. Teleport, ùn tắc và phạm vi kết luận nguyên nhân

Cologne1 theo ba seed: **15 yield; 9 yield; 15 yield + 1 wrong lane**.
Ingolstadt1 seed 23423: **2 yield + 1 jam**; hai seed kia không teleport.
Không có collision teleport. Nhãn yield của SUMO không có nghĩa runtime
còn mở g/s. Mỗi teleport start được đối chiếu TraCI, warnings.log và
statistics.xml; không đếm teleport end thành một sự kiện mới.

Ví dụ Cologne1 seed 42: xe 115444_403_0 teleport lúc 25601 trên 28198821#3_1.
Cả ba movement của làn không có G trong 300 s trước đó; G gần nhất kết thúc
25260, cách 341 s. Đây là bằng chứng khoảng thiếu phục vụ trên làn đó, không
chỉ suy từ action 3 không được chọn. Các seed 23423/12345: xe 151372_418_0
teleport lúc 25535 trên 27115123#3_1; hướng link 18/19 chưa có G từ đầu đến
đó (335 s), còn link 17 đang G. Log tái chèn xe trên 32038051#0, phù hợp
hướng link 19; chưa trace route/ý định lane trực tiếp nên chỉ coi là bằng
chứng phù hợp, không khẳng định đã chứng minh toàn bộ hành vi xe.

Ingolstadt1 seed 23423: carIn68842:1 lúc 60392 và carIn104001:1 lúc 60973 trên
164051413_2, làn chỉ có movement link 4. Link 4 không có G trong 330/303 s
trước sự kiện và không được phục vụ trong 300 s trước đó; G kế tiếp còn
20/9 s. Đây là bằng chứng cụ thể về thiếu phục vụ hướng đó. Cùng lúc 60973,
carIn89871:1 bị jam trên 653473569#5_2, ngoài tập lane biên TLS: có thể liên
quan ùn ngược, nhưng không có trace đầy đủ để tách nguyên nhân từng xe.

Wrong-lane Cologne1 seed 12345: 182638_431_0 lúc 27028 trên 27115123#3_0;
hai movement từ làn này đều được G 90 s trong 300 s trước đó, G gần nhất
cách 88 s. Không quy sự kiện này chỉ cho đói pha; cần trace đổi làn/route để
giải thích sâu hơn. analysis.json giữ đủ 43 sự kiện, mọi movement ứng viên,
thời gian G gần nhất/kế tiếp và G trong 300 s trước. Không suy ý định xe
từ toàn queue làn, không quy tất cả teleport cho cùng một nguyên nhân.

Sáu warning thiếu vàng Cologne1/lượt vẫn là warning khi nạp chương trình
gốc trước runtime, như giai đoạn 3. Các warning còn lại là start/end teleport;
Ingolstadt1 chỉ có 6 warning tương ứng 3 sự kiện ở seed 23423. Tín hiệu đọc
lại từng giây xác nhận runtime có vàng/đỏ đúng khi đổi action.

## 6. Kiểm thử, lỗi đã sửa và bằng chứng

Lệnh: `.\.venv\Scripts\python.exe -m pytest -q` → **49 passed**, không skip
sau khi có đủ dữ liệu cuối. Gồm 26 test cũ và 23 trường hợp MP: ví dụ tay,
đầu vào/ra lớn nhỏ, link trùng/làn nhiều hướng, thứ tự state, sai kích thước,
G runtime, action đúng tập, hòa/0/âm, reward độc lập, tiếp tục xanh, cắt mọi
stage ở horizon, đóng SUMO thật khi controller lỗi và từ chối ghi đè.

Test CSV của sáu lượt tính lại công thức từ controlled_links.csv khảo sát
(không gọi lại helper MP), kiểm tra mọi action, thời gian/signal TraCI, clearance
trước xanh, queue-seconds/reward, 3600 mẫu liên tục, đầu/cuối episode và số
collision/teleport. Runner chờ process thoát và kiểm tra exit code 0.

Test ban đầu không import được vì chưa có mp.py; sau triển khai có 41 passed.
Kiểm tra CSV độc lập của lượt đầu phát hiện lỗi hòa do floating point:
Cologne1 seed 23423, decision 24, action 1/4 có pressure toán học bằng 10,
nhưng action 1 thành 9.999999999999998 nên action 4 thắng sai quy tắc.
Test hồi quy test_fractional_weights_do_not_break_an_exact_pressure_tie đã
thất bại với ví dụ đó trước sửa; dùng Fraction rồi test qua. Giữ nguyên
toàn bộ **sáu lượt đầu** ở initial_float_runs/, không dùng chúng làm kết quả
bàn giao. Chạy lại sáu lượt ở mp/ bằng code cuối; teleport Cologne1 seed
23423 đổi 14→15, các tổng teleport còn lại không đổi. Không che kết quả xấu.

Rà soát độc lập chỉ đọc theo skill requesting-code-review: 49 test passed,
không có finding Critical/Important/Minor, hash nguồn đúng. Reviewer không
chạy thêm mô phỏng; README được tác giả đối chiếu với đầu ra cuối sau review.
review.json lưu phạm vi và những điều reviewer không kết luận.

## 7. Đầu ra, lệnh chạy và vị trí thuyết trình

Mỗi mp/<map>/seed_<seed>/ có metadata.json, actions.csv, signals.csv,
result.json, command.txt, sumo.log, warnings.log, statistics.xml, collisions.xml.
actions lưu input queue theo lane order metadata, pressure mọi action, thời
điểm quyết định/thời lượng từng stage/reward/done. signals lưu từng giây,
queue tổng, xe còn trong nút và collision/teleport thô. mp_summary.csv là tổng
hợp cuối; analysis.json là phân tích khoảng phục vụ/chuỗi lặp/teleport.

Queue-seconds đơn vị xe·s, tổng queue incoming duy nhất ở cuối từng bước 1 s;
reward = âm queue-seconds / thời gian toàn call, gồm vàng/đỏ. Gồm xe chưa tới
đích nếu đang trên lane đo; không gồm xe chưa chèn, internal hoặc queue ở
edge trước. running/waiting giữ riêng trong statistics. Chưa tính Average
Waiting Time, Travel Time, Time Loss hay Throughput bằng một quy trình chung;
không dùng số xe hoàn thành để bỏ qua xe chưa hoàn thành. Cần thu thêm dữ liệu
xe/chuyến ở giai đoạn đánh giá, chạy lại FT/MP cùng cách đo nếu cần.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
foreach ($mapName in @('cologne1','ingolstadt1')) {
    foreach ($runSeed in @(23423,42,12345)) {
        .\.venv\Scripts\python.exe src/mp.py $mapName --seed $runSeed --output "reports/stage4/mp_repeat/$mapName/seed_$runSeed"
    }
}
```

Báo cáo đã lưu đủ sáu lượt. Đổi `--output` nếu đường dẫn chạy lại đã tồn tại.
Test CSV tự thu lượt nằm ở reports/stage4/mp/*/seed_*/result.json; nếu muốn
kiểm tra lượt mới bằng test đó, dùng thư mục seed khác cùng mẫu trong mp/.
Cài môi trường theo [README gốc](../../README.md#cài-đặt-trên-windows).

| Nội dung | File / hàm |
|---|---|
| Input lane queue theo đúng thứ tự | sumo_env.py: SumoEnv.get_state |
| Cặp movement duy nhất và trọng số | mp.py: movement_weights |
| Công thức và lọc G runtime | mp.py: calculate_pressures |
| Argmax, pressure âm và tie-break | mp.py: choose_action |
| Lấy state, chọn action, gọi env.step | mp.py: max_pressure |
| SUMO nhận tín hiệu/tiến đồng hồ | sumo_env.py: SumoEnv._run_signal |
| Vàng/đỏ, clearance, reward, done | sumo_env.py: SumoEnv.step, _junction_vehicles |
| Reset/close process, CSV/JSON và đối chiếu | mp.py: run_mp |
| Ví dụ tay và lỗi sai số hòa | test_mp.py: test_hand_pressure_uses_both_queues_and_state_lane_order; test_fractional_weights_do_not_break_an_exact_pressure_tie |
| Kiểm tra lại sáu trace thực | test_mp.py: test_recorded_mp_decisions_pressures_timing_reward_and_safety |

## 8. Điều kiện sang giai đoạn 5

**Đủ điều kiện bắt đầu Q-Learning trong phạm vi học tập của hai bản đồ**,
dùng chung state thông tin gốc/action/reward/chuyển pha. Giai đoạn 4 hoàn
thành với mô hình MP xấp xỉ đã công bố: công thức và mapping có test tính tay
và kiểm tra lại trace, sáu lượt đủ giờ, không thay môi trường giữa FT/MP.
Phải gọi đúng tên mô hình khi so sánh, không gán bảo đảm MP lý tưởng cho nó.

Giới hạn còn thật: teleport do chờ/ùn/sai làn; không có fairness; phân bổ đều
không biết hướng từng xe; queue downstream bằng 0 tại mọi quyết định thực;
chưa trace mọi xe teleport, chưa kiểm tra GUI, bước dưới 1 s, mọi hình học
thân xe, nhu cầu/seed khác hoặc toàn bộ metric đánh giá. Không chứng nhận an
toàn tuyệt đối, không tuyên bố đã loại bỏ ùn tắc hoặc MP thắng FT.
