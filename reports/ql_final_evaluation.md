# Giai đoạn 6 — Kiểm chứng Q-Learning và khóa đánh giá cuối

**Chọn QL-Overdue cho Cologne1 và QL-Original cho Ingolstadt1.** Trong các biến thể QL thuần đã thử, chỉ Overdue có bằng chứng validation đủ một giờ để thay Original trên Cologne1. Dùng một checkpoint cho mỗi policy đã chọn. Cả 70 lượt trên năm seed test mới còn chờ chạy. Các số giao thông dưới đây đến từ validation, test lịch sử và smoke, với nguồn và ngân sách riêng.

## 1. Từ quan sát đến cập nhật Q

Mỗi giây, SUMO cho biết số xe dừng trên từng làn biên. Tại thời điểm quyết định, Original cộng các làn theo từng edge vào/ra, chia bin 0, 1–4, ≥5 rồi nối ID pha. Một hàng của bảng Q có một cột cho mỗi pha xanh hợp lệ theo đúng thứ tự metadata. Cologne1 có 8 pha, Ingolstadt1 có 3. Overdue nối thêm mã edge có xe dừng đang thiếu xanh lâu nhất từ 300 giây, hoặc 0 khi chưa có edge quá hạn. Đồng hồ tuổi bắt đầu khi quan sát queue dương, đặt lại khi queue hết hoặc một movement của edge thực nhận xanh ưu tiên.

Khi train, với xác suất ε agent lấy mẫu đều một pha, còn lại chọn pha có Q lớn nhất. Cấu hình đã chọn khởi tạo Q=0 và lấy cực đại đầu tiên khi hòa. SUMO thực thi pha rồi trả về reward và trạng thái tiếp theo. Gọi $q_k$ là tổng số xe dừng trên các làn vào phân biệt tại cuối giây thứ $k$. Với action dài $d$ giây, $r=-\sum_{k=1}^{d}q_k/d$. Tích lũy gồm cả xanh, vàng, đỏ và thời gian chờ giải tỏa. Cập nhật một ô bằng $Q(s,a)\leftarrow Q(s,a)+\alpha[y-Q(s,a)]$, với $y=r+\gamma\max_b Q(s',b)$ hoặc $y=r$ khi kết thúc episode. [Môi trường](../src/sumo_env.py#L238) · [Bellman](../src/q_learning.py#L169)

**Hai vấn đề thấy trong trace.** Các đội xe có độ dài rất khác nhau có thể cùng vượt ngưỡng 5 và nhận cùng khóa. Một policy đã giữ xanh 3.190 giây, trong đó cùng một state lặp 106 quyết định với một cực đại Q duy nhất. Đồng thời, một action chưa cập nhật còn Q=0 có thể thắng những action đã học có Q âm. Tỷ lệ state mới đếm khóa vắng trong checkpoint. Tỷ lệ chọn action chưa học dùng số lần cập nhật của cặp state–action đến checkpoint, kể cả lần chọn ở state mới. Một ô đúng bằng 0 vẫn có thể đã học. [Đối chiếu log và checkpoint](q_learning_review.md) · [Phân tích chọn pha](ql_action_selection.md)

## 2. Đồng hồ SUMO, clearance và discount

Đọc lại toàn bộ 30 tệp `actions.csv` của bộ FT/MP/QL lịch sử (5 seed × 3 controller mỗi map). Đối chiếu 108.000 dòng `signals.csv` với đồng hồ, tín hiệu yêu cầu và thời lượng từng pha; ở mọi lần đổi có mở xanh, danh sách xe còn chiếm nút trước giây xanh đầu tiên đều rỗng. Đếm riêng quyết định cuối của từng lượt vì horizon có thể cắt ở bất kỳ giai đoạn nào. Với các quyết định còn lại, phân loại giữ pha bằng `previous_phase == action`. [Các lượt gốc](stage6/runs/) · [Đối chiếu từng giây với tín hiệu và XML](../tests/test_evaluation.py)

| Đại lượng theo đồng hồ mô phỏng | Cologne1 | Ingolstadt1 |
|---|---:|---:|
| Giữ pha, số quyết định trước quyết định cuối × thời lượng | 243 × 30 s | 174 × 30 s |
| Đổi pha, số quyết định trước quyết định cuối × thời lượng | 1.161 × 40 s | 1.272 × 38 s |
| Vàng + tất cả đỏ thực đo khi đổi trong các quyết định trên | 8 + 2 s | 6 + 2 s |
| Clearance từ lúc đóng xanh cũ đến lúc mở xanh mới | 10 s | 8 s |
| Đỏ kéo dài thêm ngoài mức tối thiểu 2 s, trong mẫu trên | 0 s | 0 s |
| Thời lượng các quyết định cuối, kể cả quyết định bị cắt | 10–30 s | 4–36 s |

Nếu còn xe bên trong nút hoặc đuôi xe chưa qua điểm ra, môi trường tiếp tục đỏ từng giây. Vì vậy 40/38 giây là thời lượng đủ xanh khi nút đã trống sau đỏ tối thiểu. Test dựng trường hợp xe chưa thoát buộc Cologne1 giữ đỏ 5 giây và hoàn thành action sau 43 giây. Mốc kết thúc có thể đến ngay trong vàng hoặc đỏ, khi đó pha xanh đích chưa mở. Đồng hồ TraCI kiểm tra mỗi bước đúng 1 giây. [Theo dõi đuôi xe](../src/sumo_env.py#L192) · [Thực thi chuyển tiếp](../src/sumo_env.py#L238) · [Test clearance](../tests/test_sumo_env.py#L109)

**Giữ γ=0,95 mỗi quyết định cho đối chứng này.** Cùng queue 10 xe ở mọi giây, cả action 30 s và 40 s đều nhận −10 và cùng hệ số bootstrap 0,95. Điều đó phù hợp khi câu hỏi là chất lượng mỗi lần chọn pha, và cho phép so với các checkpoint đã validation. Nếu mục tiêu là tổng xe·giây qua một giờ, thời lượng phải có trọng số. Ví dụ QL lịch sử seed 42 Cologne1 có trung bình queue cho mỗi quyết định 47,344 xe nhưng trung bình theo giây 47,507 xe. Chênh lệch này kiểm tra được trên một quỹ đạo; ảnh hưởng của cách discount lên policy cần một đối chứng train lại. [Trace seed 42](stage6/runs/cologne1/QL/seed_42/actions.csv)

Với mục tiêu theo thời gian thực, một biến thể Semi-Markov nhất quán có thể chọn $\beta=0{,}95^{1/30}$ mỗi giây. Khi đó $R_d=\sum_{k=0}^{d-1}\beta^k(-q_{k+1})$, bước thường dùng $R_d+\beta^d\max_b Q(s',b)$, bước cuối chỉ dùng $R_d$. Bootstrap của 30 s là 0,95 và của 40 s khoảng 0,934. Reward lúc này tích lũy chi phí của từng giây, làm đổi cả mục tiêu và thang Q. Chưa có validation đối chứng cho mục tiêu đó; giai đoạn này giữ nguyên Bellman, reward và các bảng đã học. [Sutton, Precup và Singh, 1999](https://www.sciencedirect.com/science/article/pii/S0004370299000521)

## 3. Các thay đổi đã thử và chi phí

Ngân sách dưới đây tính **trên mỗi model** bằng số episode × giây mô phỏng. Số dòng là số state có trong bảng; số ô bằng số dòng × số action. Coverage dùng cặp đã cập nhật trên các ô của bảng. Các kết quả smoke chỉ cho biết tác động trong horizon ngắn được ghi ở từng hàng.

| Thay đổi, công thức hoặc quy tắc | Chi phí và thay đổi chất lượng học | Kết quả giao thông và quyết định |
|---|---|---|
| Original → Overdue, nối mã edge quá hạn | 15 × 3.600 s, exploration 7/17. Cologne1 từ 115/122 lên 176/157 dòng. Với seed 7, cặp cập nhật tăng 467→587, coverage giảm 50,76→41,69%. | Validation completion 55,65→79,98% qua cả hai model. Gap cực đại giảm 3.440→3.280 s ở seed 42, 3.600→3.130 s ở 12345. **Đạt trên Cologne1.** [Overdue](ql_overdue_state/README.md) |
| Ưu tiên pha có số cập nhật bằng 0 trong nhánh khám phá | 15 × 3.600 s. Cologne1 e7 từ 115→125 dòng. Tỷ lệ chọn cặp chưa học trên test giảm 14,28→6,39%. | Completion validation Cologne1 55,83→42,36%, Ingolstadt1 92,63→88,51%. Loại trên cả hai map. [Untried-first](ql_untried_first/README.md) |
| Chia thêm bin 15+; Guard là controller lai, có quyền đổi action | 15 × 3.600 s. Original Stage 8 có 129 dòng; thêm bin lên 213 dòng, coverage 48,45→41,26%. Guard dùng lại bảng. | Validation completion 83,75% Original, 71,96% thêm bin, 76,33% thêm bin kèm Guard. Guard trên Original cũng phụ thuộc model. Giữ Guard ngoài so sánh QL thuần. [Stage 8](stage8/README.md) · [Validation Guard](ql_validation_test.md) |
| Busiest thay bin vào bằng cờ có xe và mã edge đông nhất | 15 × 3.600 s. Overdue 176/157 → Busiest 126/108 dòng ở e7/e17. | Validation queue tăng ở cả hai seed; completion seed 12345 giảm 85,61→67,27%. Loại. [Busiest](ql_busiest_state/README.md) |
| Lịch ε có mốc giảm cố định, $t=\min((e-1)/14,1)$, tuyến tính $1-0{,}95t$ hoặc mũ $0{,}05^t$; lịch seed SUMO quay vòng | Smoke 3 × 420 s mỗi lịch. Cùng chu kỳ seed; tuyến tính 30/152 ô đã học, hàm mũ 30/176. Cả hai chọn cặp chưa học 14/14 lần ở validation. | Cả hai có 6 teleport ở validation ngắn. Các mốc 30/60/100 episode và lợi ích riêng của seed quay vòng còn thiếu đối chứng đủ dài. Giữ lịch tuyến tính và seed train lịch sử cho các model khóa. [Ngân sách](ql_training_budget.md) |
| Q ban đầu −1.358,5 từ reward trung vị chia $(1-\gamma)$; nhánh khác chỉ phá hòa ngẫu nhiên khi train | 3 × 420 s. Baseline, Q âm và phá hòa lần lượt cập nhật 33/144, 19/112, 32/184 ô. Q âm giảm lựa chọn chưa học từ 11/11 xuống 0/11. | Completion tương ứng 26,35%, 25,99%, 3,25%; phá hòa có chuỗi giữ một pha 420 s. Giữ Q=0 và cực đại đầu tiên. [Q và phá hòa](ql_action_selection.md) |
| Relative đánh dấu hướng đứng đầu khi hai queue đều ≥5 và chênh ≥5 xe | 3 × 600 s. Bảng Overdue 26 dòng, Relative 30; coverage 21,15→19,58%. | Trên hai seed, completion tăng nhưng queue đều tăng, ở seed 12345 từ 66,33→81,19 xe. Chưa chọn. [Relative](ql_relative_state.md) |
| Reward $-\bar q-\lambda\min(1,\max(0,(A-300)/300))$, với $A$ là tuổi lớn nhất của edge còn queue cuối action | 3 × 900 s mỗi nhánh, λ=0/5/15. Hai λ dương đổi 19 ô so với baseline trong 32 state chung mà vẫn chọn cùng greedy tại cả 32 state. | 12 lượt validation có cùng tuổi chờ cực đại 620 s và cùng chuỗi pha trên từng seed. Giữ λ=0. [Reward](ql_starvation_reward.md) |

Train khóa dùng seed SUMO 23423, exploration seed 7, α=0,1, γ=0,95, 15 episode từ bảng rỗng, 3.600 s/episode và ε tuyến tính từ 1 xuống 0,05. SUMO reset và đồng hồ Overdue bắt đầu lại mỗi episode. Bảng Q và RNG khám phá tiếp tục qua các episode; checkpoint ghi sau từng episode. Cấu hình chọn chính là policy từng qua validation, với Guard tắt. [Train và checkpoint](../src/q_learning.py#L434)

## 4. Validation chọn checkpoint, test lịch sử và bộ test mới

**Khóa trước test mới.** So e7 và e17 ở cùng checkpoint 15 bằng completion validation trung bình trên seed 42/12345, hai lần lặp mỗi seed. Trên Cologne1, Original e7/e17 đạt 69,53/41,76%; Overdue đạt 83,98/75,97%. Trên Ingolstadt1, Original đạt 97,12/88,14%; các trace đối chứng đã khảo sát chưa kích hoạt edge quá hạn. Khóa e7 cho cả ba bảng. “QL cải tiến tốt nhất” trên Cologne1 trùng đúng model Overdue; trên Ingolstadt1, cấu hình giữ lại trùng Original. Mỗi policy chỉ chiếm một hàng và một lượt cho mỗi cặp seed/repeat. [Validation Cologne1 từng lượt](ql_overdue_state/runs.csv) · [Chọn model Ingolstadt1](ql_validation_test.md)

| Map / checkpoint 15 | Dòng × pha | SHA-256 của chính tệp chạy |
|---|---:|---|
| Cologne1 Original | 115 × 8 | `1267e2e6b75f07894cf9f4252097fb719373c0240a0c89fd4229047a78017136` |
| Cologne1 Overdue | 176 × 8 | `6e9953dac311913d77d1dd5abf041a6830b6a150d323ac5566afbc4be93cd33e` |
| Ingolstadt1 Original | 17 × 3 | `24169b3ebe29d41548fc08daf4f96b4f9aa704a32343363534e4d0ae75a7573a` |

Hai checkpoint Cologne1 trong thư mục báo cáo là bản sao nguyên byte từ các thư mục thí nghiệm trong `experiments/`; Ingolstadt1 dùng trực tiếp tệp Stage 7 cùng SHA. [Các checkpoint Cologne1](ql_final_models/) · [Checkpoint Ingolstadt1](stage7/training/ingolstadt1/exploration_7/ingolstadt1_q_table.json)

**Test lịch sử của đúng ba SHA đã khóa.** Mỗi hàng dưới đây là trung bình 6 lượt (3 seed × 2 repeat); lấy cực đại cho cột gap. Cologne1 dùng test Overdue 84201–84203, chọn các dòng QL của e7 theo quyết định validation ở trên. Bảng tổng hợp lịch sử của cả e7 và e17 cho completion 56,89→82,08%, khác mẫu e7 dưới đây. Ingolstadt1 dùng test 52101–52103 của model đã khóa. Các seed này đã được xem trong nghiên cứu trước. [Từng lượt Cologne1](ql_overdue_state/runs.csv) · [Số liệu Ingolstadt1](ql_validation_test.md)

| Cologne1 | Completion | Xe/h | Queue (xe) | Chờ / đi lại / time loss (s/xe) | Teleport/lượt | Gap max (s) |
|---|---:|---:|---:|---|---:|---:|
| FT | 93,73% | 1.888,67 | 74,43 | 146,96 / 190,46 / 167,48 | 0 | 250 |
| MP | 97,58% | 1.966,17 | 28,19 | 64,28 / 100,62 / 77,80 | 10,67 | 770 |
| QL-Original | 69,30% | 1.396,33 | 103,00 | 151,80 / 192,03 / 170,35 | 56,00 | 3.170 |
| QL-Overdue (chọn) | 83,31% | 1.678,67 | 46,33 | 74,27 / 113,25 / 90,04 | 42,83 | 3.130 |

| Ingolstadt1 | Completion | Xe/h | Queue (xe) | Chờ / đi lại / time loss (s/xe) | Teleport/lượt | Gap max (s) |
|---|---:|---:|---:|---|---:|---:|
| FT | 97,96% | 1.681,00 | 9,20 | 31,93 / 64,22 / 43,50 | 0 | 84 |
| MP | 97,34% | 1.670,33 | 5,74 | 34,21 / 66,23 / 45,58 | 1,33 | 388 |
| QL-Original (chọn) | 98,14% | 1.684,00 | 9,39 | 48,91 / 82,08 / 61,26 | 1,33 | 438 |

Completion = số xe đến đích hợp lệ / demand; throughput quy đổi nhóm đó sang xe/giờ. Ba metric thời gian chỉ tính trên xe đã đến, chưa từng teleport và chưa bị xóa. Queue tính theo thời gian trên các làn vào. Teleport đếm sự kiện bắt đầu. Gap tính cả thời gian movement thiếu xanh khi vắng xe; tuổi edge có nhu cầu là chỉ số khác. Khi đọc kết quả cuối, kèm `unfinished`, `undeparted`, collision và số chuyến hợp lệ. Ở Cologne1 e7, Overdue còn trung bình 189,17 xe chưa vào mạng và 107,67 xe đã vào nhưng chưa hoàn thành. [Định nghĩa metric](../src/evaluation_metrics.py#L47)

**Bộ test mới đã chuẩn bị, 0/70 lượt đã chạy.** Seed 161101, 161102, 161103, 161104, 161105 chưa xuất hiện trong các metadata, kế hoạch và lệnh thí nghiệm đã rà soát. Mỗi seed lặp hai lần. Cologne1 dùng 2.015 chuyến, 25.200–28.800 s và 4 controller (40 lượt). Ingolstadt1 dùng 1.716 chuyến, 57.600–61.200 s và 3 controller (30 lượt). Mỗi lượt chạy bằng `run_evaluation()` trên cùng map, demand và luật chuyển pha. QL greedy, ε=0, Guard tắt, λ=0; kiểm tra SHA trước/sau và so bảng trong bộ nhớ. [Kế hoạch chạy](../tools/final_ql_evaluation.py)

Từ gốc repo, lệnh đầu chỉ kiểm SHA, nạp metadata và in lịch. Lệnh thứ hai dành cho đợt chạy đầy đủ đã được cho phép, với một thư mục đích mới.

~~~powershell
.\.venv\Scripts\python.exe -B tools/final_ql_evaluation.py
.\.venv\Scripts\python.exe -B tools/final_ql_evaluation.py --run-to .\experiments\ql-final-test-161101
~~~

Runner ghi `plan.json` trước lượt đầu, giữ XML/CSV/log cho từng lượt và ghi một dòng `runs.csv` sau mỗi lượt thành công. Đối chiếu đủ toàn bộ khóa (map, controller, seed, repeat) trước khi tổng hợp; nếu lỗi, giữ log và ghi lượt thiếu. Tách từng map, lấy trung bình hai repeat trong seed rồi so chênh lệch theo cặp controller cùng seed. Công bố cả năm chênh lệch, trung bình, min–max theo seed, chênh lệch hai repeat và lượt xấu nhất. Hai repeat cùng seed giúp đo độ tái lập, không làm tăng số seed độc lập. Giữ cả hai repeat, cả lượt có collision/teleport và giá trị thời gian thiếu khi số chuyến hợp lệ bằng 0.

## 5. Kiểm chứng và các dòng dùng để thuyết trình

| Bước | Mở hàm và kiểm tra |
|---|---|
| Input → state | [SumoEnv.get_state, dòng 182](../src/sumo_env.py#L182), [EdgeDemand.observe, dòng 76](../src/q_learning.py#L76), [encode_state, dòng 99](../src/q_learning.py#L99). Lần lượt đọc làn, tính tuổi queue theo edge và tạo khóa. |
| State → action | [choose_action, dòng 146](../src/q_learning.py#L146). Chỉ số cột ánh xạ về ID pha gốc; giữ đúng thứ tự khi nạp model. FT chạy vòng theo [fixed_time](../src/ft.py#L18), MP so áp lực vào trừ ra qua [calculate_pressures](../src/mp.py#L30). |
| Action → đèn → reward | [Kiểm xanh xung đột theo ma trận foes](../src/sumo_env.py#L94), [gửi tín hiệu và bước TraCI](../src/sumo_env.py#L214), [chuyển pha và reward](../src/sumo_env.py#L238). Kiểm hình dạng tín hiệu thực ở mỗi giây, đỏ đến khi nút trống. |
| Reward → Q → checkpoint | [Target thường/terminal, dòng 179](../src/q_learning.py#L179), [run_episode, dòng 303](../src/q_learning.py#L303), [save_q_table / load_q_table](../src/q_learning.py#L237). Kiểm map, thứ tự action, encoding, reward version, dữ liệu đầu vào và cấu trúc hàng. |
| Policy khóa → metric | [greedy_steps, dòng 32](../src/evaluate.py#L32) chỉ đọc, kể cả state mới. [run_evaluation, dòng 61](../src/evaluate.py#L61) kiểm clock và bảng trước/sau, gọi chung [read_metrics, dòng 132](../src/evaluation_metrics.py#L132) cho FT, MP, QL. |

**Smoke mới trên SUMO 1.28.0, Python 3.11.9.** Gọi `run_evaluation(..., seed=42, horizon=120)` với cả 7 cặp map/controller trong lịch. Mỗi lượt ghi 120 mẫu một giây, kết thúc đúng horizon, 0 teleport và 0 collision. Cả ba SHA trước/sau khớp bảng khóa. Cologne1 hoàn thành lần lượt 10, 40, 13, 12 trên 65 chuyến (FT, MP, Original, Overdue); Ingolstadt1 là 40, 56, 42 trên 75 (FT, MP, Original). Horizon ngắn chỉ kiểm tra đường thực thi. SUMO cảnh báo chương trình TLS gốc Cologne1 thiếu vàng khi chuyển một số pha; runner đóng pha bằng chuỗi vàng và đỏ của môi trường. [Raw bảy lượt, ZIP trong experiments](../experiments/ql-final-smoke-u6o8k47g.zip)

Lệnh `.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` đạt **219/219** trong 175,95 s. Bộ kiểm thử có phép tính tay reward/Bellman, terminal giữa vàng/đỏ, phát lại trace, từ chối metadata sai, evaluate chỉ đọc, kiểm ma trận xung đột, xanh FT, pressure MP, đối chiếu metric độc lập từ XML/CSV và kiểm đủ các cặp seed/repeat của lịch mới. Thử riêng phần ghi lịch và CSV với đầu ra smoke có sẵn cho đủ 70 khóa, không khởi chạy SUMO trên seed test. Xanh bảo vệ phù hợp ma trận foes tại TLS đã xét. Trong dữ liệu lịch sử vẫn có va chạm ở nút ưu tiên lân cận; kết luận an toàn chỉ giới hạn ở tín hiệu và trace đã kiểm. [Kiểm thử](../tests/) · [Điều tra va chạm Ingolstadt1](stage6_1/README.md)

## 6. Hội tụ, khả năng tổng quát và hướng DQN

Ở Cologne1 e7, Overdue đổi pha greedy trên 29/172 state chung giữa checkpoint 14 và 15; ở Ingolstadt1, Original e7 đổi 8/17. Trên test Overdue lịch sử gộp hai seed khám phá, tỷ lệ quyết định ở state mới là 13,91%, chọn cặp chưa cập nhật là 42,12%; chỉ 2/359 quyết định quá hạn phục vụ ngay edge chờ lâu nhất. Completion ổn định hơn trong mẫu đó (độ lệch chuẩn giữa các lượt 13,77→1,37 điểm phần trăm), nhưng bằng chứng chưa cho thấy policy hội tụ. Mỗi map chỉ có một bộ demand, cùng SUMO seed đôi khi cho kết quả khác nhau. Giá trị của Overdue đối với hướng rẽ cụ thể và khi demand thay đổi còn chưa rõ. [Coverage, starvation và biến thiên](ql_overdue_state/README.md) · [Phát lại SUMO](reproducibility_check.md)

Ưu tiên kế tiếp là chạy bộ test đã khóa. Nếu nghiên cứu tiếp về hội tụ, dùng thí nghiệm ngân sách 15/30/60 episode với validation riêng và bộ test khác sau mọi lần thiết kế lại. Nếu bảng tiếp tục thưa, một nghiên cứu DQN sau đó có thể giữ cùng quan sát, action xanh và môi trường, thay bảng bằng mạng ước lượng vector Q rồi kiểm riêng replay buffer, target network và xử lý terminal. So lại với Q-Learning dạng bảng dưới cùng ngân sách tương tác, seed và metric để đo phần lợi ích thực của xấp xỉ hàm.
