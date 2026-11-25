# Phân tích chất lượng học Q-Learning

## Chẩn đoán giai đoạn 1 · 23/11/2026

**Ưu tiên cho Cologne1 là kiểm chứng xem QL-Overdue đã học đủ các pha có thể phục vụ hướng chờ lâu hay chưa.** Thêm mã hướng quá hạn vào state đã nâng completion và rút ngắn chuỗi giữ pha trong thí nghiệm đã lưu. Tuy nhiên trong 359 quyết định test có hướng quá hạn, chỉ 155 quyết định có ít nhất một pha phục vụ hướng đó từng được cập nhật ở đúng state. Bản chẩn đoán này dùng QL-Original làm đối chứng mặc định và xét QL-Overdue riêng trên Cologne1. [Thí nghiệm Overdue](ql_overdue_state/README.md)

### Từ hàng đợi đến một lần cập nhật

1. `SumoEnv.get_state()` đọc số xe dừng ở từng làn vào, làn ra và ID pha. `build_encoding()` nhóm các làn theo edge, `encode_state()` cộng queue trong nhóm, chia mức 0, 1–4, ≥5 rồi nối ID pha. Overdue thêm mã edge có queue thiếu xanh lâu nhất từ 300 giây, hoặc 0. [Môi trường](../src/sumo_env.py) · [Mã hóa và EdgeDemand](../src/q_learning.py)
2. `run_episode()` gọi `choose_action()`. Khi train Original/Overdue, nhánh khám phá chọn đều trong các pha xanh hợp lệ, nhánh greedy lấy pha đầu tiên đạt Q lớn nhất. Khi giữ pha, `SumoEnv.step()` yêu cầu thêm 30 giây xanh. Khi đổi pha, nó chạy vàng (8 giây ở Cologne1, 6 ở Ingolstadt1), đỏ ít nhất 2 giây, chờ giải tỏa rồi mới xanh. `_run_signal()` gửi tín hiệu qua TraCI và lấy mẫu từng giây. Mốc cuối có thể cắt bất kỳ giai đoạn nào. [Chọn action](../src/q_learning.py) · [Thực thi pha](../src/sumo_env.py)
3. Reward là `r = −queue_seconds / elapsed`, gồm cả vàng và đỏ. Trong train, `update_q_value()` áp dụng `Q[s,a] += α × (target − Q[s,a])`, với `target = r + γ × max Q[s′,·]` hoặc chỉ `r` khi kết thúc. `save_q_table()` lưu từng checkpoint. Trong test dưới đây, `greedy_steps()` dùng ε = 0, Guard tắt, bảng Q cố định. `signal_metrics()` tính queue toàn giờ và khoảng thiếu xanh từng movement. `trip_metrics()` tính completion trên toàn bộ nhu cầu và các thời gian trung bình trên xe đến đích hợp lệ. [Học và lưu](../src/q_learning.py) · [Evaluate](../src/evaluate.py) · [Metric](../src/evaluation_metrics.py)

### Bảng Q đã học đến đâu?

Replay từ `episode_*/actions.csv` của sáu model, 15 episode/model, SUMO seed 23423, α = 0,1, γ = 0,95. State đã gặp được đếm tại thời điểm ra quyết định. Mỗi lần train chọn pha tính một cập nhật, kể cả khi giá trị Q giữ nguyên. Mẫu số coverage là số state đã gặp × số pha. Cột cuối chỉ xét state chung của hai checkpoint. [Replay Bellman, RNG và checkpoint](../tools/analyze_ql.py) · [Cách so greedy](../tools/stage7_summary.py)

| Bản đồ / model | Seed khám phá | State đã gặp | Cặp đã cập nhật / số ô | Số cập nhật theo thứ tự ID pha | Greedy đổi 14→15 |
|---|---:|---:|---:|---|---:|
| Cologne1 Original | 7 | 115 | 467/920 | 187, 211, 211, 164, 156, 142, 168, 158 | 35/113 |
| Cologne1 Original | 17 | 122 | 472/976 | 193, 204, 207, 161, 165, 130, 173, 163 | 31/119 |
| Cologne1 Overdue | 7 | 176 | 587/1.408 | 221, 218, 194, 168, 146, 139, 158, 151 | 29/172 |
| Cologne1 Overdue | 17 | 157 | 536/1.256 | 223, 205, 211, 161, 167, 141, 158, 135 | 34/154 |
| Ingolstadt1 Original | 7 | 17 | 49/51 | 639, 541, 335 | 8/17 |
| Ingolstadt1 Original | 17 | 20 | 50/60 | 658, 538, 317 | 9/20 |

Thứ tự pha là 0–7 ở Cologne1 và 0, 2, 4 ở Ingolstadt1. Với Original Cologne1 seed 7, cả tám pha đều có ít nhất 142 cập nhật, nhưng 234/467 cặp đã thử chỉ nhận **một** cập nhật. Cùng cách đếm ở Overdue seed 7 cho 379/587. [Train theo episode của Cologne1](../experiments/overdue-eval-ukeh31e3/training.csv) · [Train của cả hai bản đồ](../experiments/exploration-204c7770/training.csv)

**Diễn biến khi train.** Với Original Cologne1 seed 7, trung bình tổng reward trên ba nhóm episode 1–5, 6–10, 11–15 lần lượt là −7.867, −7.043, −5.777. Queue tương ứng là 84,40, 75,42, 61,96 xe. Ở seed 17, reward là −8.629, −6.293, −6.091 và queue là 92,36, 68,07, 65,47 xe. Original Ingolstadt1 seed 7 cũng giảm queue từ 13,22 qua 9,28 còn 7,78 xe. Trong các lượt này ε đồng thời giảm từ 1 xuống 0,05. Các trung bình mô tả policy đang khám phá, chưa xác nhận chất lượng greedy hay hội tụ. [Log train Cologne1](../experiments/overdue-eval-ukeh31e3/training.csv) · [Log train Ingolstadt1](../experiments/exploration-204c7770/training.csv)

**Khi evaluate.** State lạ là khóa vắng trong checkpoint. Với khóa đã biết, từng pha vẫn có thể có `N(s,a)=0`. Hàng dành cho khóa lạ được đọc như toàn 0. Q bằng 0 là một **giá trị**, có thể giữ nguyên sau cập nhật nếu target bằng 0 và Q cũ bằng 0. Trong sáu checkpoint cuối được kiểm, số ô vừa có cập nhật vừa còn đúng 0 là 0. Bảng dưới xác định `N` từ log train rồi đối chiếu từng lựa chọn greedy.

| Model test | Quyết định | State lạ | State đã biết, pha chọn có N=0 | Tổng chọn cặp N=0 |
|---|---:|---:|---:|---:|
| Cologne1 Original | 1.228 | 29 (2,36%) | 155 (12,62%) | 184 (14,98%) |
| Cologne1 Overdue | 1.114 | 155 (13,91%) | 314 (28,19%) | 469 (42,10%) |
| Ingolstadt1 Original | 1.214 | 0 | 8 (0,66%) | 8 (0,66%) |

Mỗi dòng gộp quyết định của 12 lượt test (2 model × 3 seed SUMO × 2 lần lặp). Vì số quyết định mỗi lượt khác nhau, số gộp có thể khác trung bình tỷ lệ từng lượt trong báo cáo cũ. Cologne1 dùng seed 84201–84203, Ingolstadt1 dùng 73101–73103. [Kết quả từng lượt Cologne1](ql_overdue_state/runs.csv) · [Kết quả từng lượt Ingolstadt1](ql_untried_first/runs.csv)

### Sáu khả năng và giới hạn của bằng chứng

| Khả năng | Bằng chứng và mức ảnh hưởng có thể kết luận |
|---|---|
| **State thiếu thông tin** | Ảnh hưởng rõ trong một chuỗi kẹt của Original, seed khám phá 17, SUMO 84201, repeat 1. Pha 4 giữ 107 quyết định, 3.190 giây xanh thực. Trong đó 106 quyết định cùng khóa dù có 54 vector queue làn vào khác nhau. Ở khóa này pha 4 đã học 11 lần và có Q lớn nhất duy nhất (−73,07). Trên toàn bộ test cùng bộ seed, Original → Overdue tăng completion trung bình từ 56,89% lên 82,08%, chuỗi cực đại còn 7 quyết định. Đây là bằng chứng cho việc thêm tuổi nhu cầu trong phạm vi đã thử. [Trace chuỗi pha 4](../experiments/overdue-eval-ukeh31e3/test/cologne1/QL-Original/exploration_17/seed_84201/repeat_1/actions.csv) · [So sánh cùng seed](ql_overdue_state/README.md) |
| **Huấn luyện chưa đủ** | Coverage thưa ở Cologne1 và greedy còn đổi ở cả sáu model. Trong 359 quyết định test Overdue có edge quá hạn, 204 lần chưa pha phục vụ nào được cập nhật tại state hiện tại. Chỉ 2/359 lần edge quá hạn lâu nhất nhận xanh ngay trong quyết định đó. Tăng riêng ngân sách train sẽ cải thiện bao nhiêu vẫn chưa biết. [Đối chiếu phục vụ và lượt cập nhật](../experiments/overdue-eval-ukeh31e3/edge_service.csv) |
| **Q khởi tạo bằng 0** | Ở 155 quyết định Original Cologne1 trên state quen có pha chọn chưa học, Q=0 đứng trên ít nhất một ô âm. Trên state lạ, mọi pha hòa và thứ tự ưu tiên pha 0. Cơ chế này xác định lựa chọn tại các điểm đó. Trong chuỗi pha 4 nêu trên, đổi cách phá hòa vẫn cho cùng action tại khóa lặp. Chưa đo riêng tác động lên completion. [Cách đối chiếu Q với N](../tools/analyze_ql.py) |
| **Exploration chưa hiệu quả** | Original Cologne1 seed 7 có 401/466 quyết định ở nhánh ngẫu nhiên trong 5 episode đầu và 87/466 trong 5 episode cuối. Thử `untried-first` trên test 73101–73103 giảm tỷ lệ chọn cặp chưa học từ 14,28% xuống 6,39% (trung bình từng lượt), nhưng completion cũng giảm từ 49,69% xuống 42,36%. Thử được nhiều pha hơn chưa bảo đảm ước lượng Q tốt hơn. Kết quả này chỉ kiểm một chiến lược thay thế. [Thử nghiệm khám phá](ql_untried_first/README.md) |
| **Reward chưa phản ánh đủ mục tiêu** | Reward phạt xe dừng ở làn vào. Lượt kẹt pha 4 có 977 xe chưa vào mạng, chỉ 752/2.015 chuyến hoàn thành hợp lệ. Xe chưa vào mạng và thời gian một hướng liên tục thiếu xanh đều không có số hạng trực tiếp trong reward. Mức ảnh hưởng của việc đổi reward còn là giả thuyết. [Kết quả của lượt kẹt](../experiments/overdue-eval-ukeh31e3/test/cologne1/QL-Original/exploration_17/seed_84201/repeat_1/result.json) |
| **Thời lượng action thay đổi** | Trong train Original Cologne1 seed 7, 162 bước chưa kết thúc dài 30 giây và 1.220 bước dài 40 giây. Mọi bước dùng cùng γ = 0,95 theo quyết định. Tổng reward cộng các trung bình mỗi bước, còn queue toàn giờ dùng trọng số số giây. Đây là khác biệt định lượng giữa hai mục tiêu. Chưa có phép thử tách tác động của discount theo thời gian. [Công thức và ví dụ](stage6_2/README.md) |

**Hướng thực sự có nhu cầu.** Tại giây SUMO 28.790 trên Cologne1, Original seed khám phá 17, SUMO 84201, repeat 2, edge `-32038056#3` có 120 xe dừng và tuổi thiếu xanh quan sát được 3.230 giây. Với Overdue seed khám phá 7, SUMO 84203, repeat 2, edge `27115123#3` có 14 xe dừng sau 3.080 giây. `replay_edge_ages()` đối chiếu queue ở đầu mỗi quyết định với các ký tự xanh ưu tiên `G` thực chạy. Tuổi đặt lại khi queue về 0 tại điểm quan sát hoặc một movement của edge nhận `G`. Khoảng thiếu xanh cực đại của một movement (3.600/3.130 giây ở Original/Overdue) có thể gồm lúc không có nhu cầu. Cả hai phép đo đều chưa xác định thời gian chờ của từng xe hoặc nhu cầu rẽ trên làn chung. [Hàm dựng lại tuổi](../tools/analyze_ql.py) · [Log phục vụ từng lượt](../experiments/overdue-eval-ukeh31e3/edge_service.csv)

### Thứ tự cải tiến đề xuất cho giai đoạn sau

1. **Kiểm chứng ngân sách học ở các state có hướng quá hạn của QL-Overdue.** Giữ encoding, reward, cách khám phá và cùng lịch ε cho đoạn train so sánh, rồi thay riêng số episode. Tại các checkpoint, đo số quyết định quá hạn có pha phục vụ từng được cập nhật, tỷ lệ thực cấp xanh, completion, teleport và queue trên validation cố định. FT và MP đạt 93,73% và 97,58% completion trên bộ test Cologne1 ở trên, cho thấy vẫn còn khoảng cách dù Overdue đã giảm lặp pha. [Đối chứng cùng demand](ql_overdue_state/README.md)
2. **Nếu các pha phục vụ đã được học mà hướng vẫn thường xuyên bị bỏ qua, thử riêng mục tiêu phục vụ.** Đo tuổi của edge có queue và khoảng thiếu xanh từng movement trước khi thử một thay đổi reward. Giữ các repeat và báo cả xe chưa vào mạng. Thử nghiệm Busiest trước đây đã làm bảng nhỏ hơn nhưng hạ completion trên cả ba seed test mới. [Kết quả Busiest](ql_busiest_state/README.md)
3. **Xếp thử nghiệm Q ban đầu, phá hòa và discount theo giây sau hai mục trên.** Mỗi lần chỉ đổi một yếu tố. Tách những quyết định hòa Q khỏi các chuỗi có cực đại duy nhất và giữ nguyên chuyển pha an toàn. Tất cả kết luận hiện tại giới hạn ở demand đã lưu, hai seed khám phá và biến thiên giữa các lượt ngay cả khi cùng seed SUMO. [Kiểm tra độ tái lập](reproducibility_check.md)

**Kiểm chứng offline.** Với Python của `.venv`, đã dùng `replay_training()` cho 90 episode và `evaluation_policy()` cho 36 lượt QL test, kiểm Q trước/sau, terminal, checkpoint, SHA model và tỷ lệ state/cặp chưa học. Các thư mục raw của đợt này nằm cạnh bản clone theo liên kết của từng thí nghiệm. Lệnh dưới tái kiểm tra riêng bằng chứng Stage 5–6 nằm hoàn toàn trong Git. Kết quả `analysis.json` dựng lại khớp nội dung đã lưu.

```powershell
.\.venv\Scripts\python.exe -B -c "import json; from tools import analyze_ql as a; result, _ = a.analyze(a.ROOT); assert json.loads(json.dumps(result)) == a.read_json(a.ROOT/'reports/stage6_2/analysis.json')"
```

---

## Báo cáo lịch sử · Giai đoạn 3 — Phân tích Q-Learning trên Cologne1

Đề xuất kiểm tra trước **QL-Original và QL-Original + Guard 300 giây trên cùng Q-table**. Hai cấu hình chỉ khác cách chọn pha khi một edge đầu vào có nhu cầu đang chờ phục vụ. Dùng lại `greedy_steps()` và `--guard-seconds 300` trong `src/evaluate.py`. Đợt này phân tích log và checkpoint đã lưu để thiết kế phép thử, chưa huấn luyện hoặc đánh giá model mới.

## Kiểm tra từ mã nguồn và dữ liệu thô

| Nội dung | Kết quả |
|---|---|
| State | `build_encoding()` và `encode_state()` cộng hàng đợi theo edge, chia mức `0`, `1–4`, `≥5`, rồi thêm ID pha hiện tại. Cologne1 có 4 edge vào, 4 edge ra và 8 action. Ingolstadt1 có 3 edge vào, 3 edge ra và 3 action. Cách mã hóa bỏ thông tin phân bố giữa các làn cùng edge, nhu cầu rẽ, thời gian chưa phục vụ và thời điểm trong episode. |
| Epsilon và tie-break | `choose_action()` lấy mẫu đều từ action hợp lệ ở nhánh khám phá, dùng RNG riêng với seed SUMO. Epsilon giảm tuyến tính từ 1 đến 0,05 trong các lượt train 15 episode. Evaluate dùng epsilon 0, chọn action đầu tiên trong thứ tự đã lưu nếu hòa Q. State lạ đọc như hàng toàn 0 mà không ghi vào bảng. Một ô Q=0 có thể đã được cập nhật, nên phải dùng số lần update để xác định coverage. |
| Bellman và terminal | `update_q_value()` dùng `Q += α × (target − Q)` với `target = r + γ × max Q(s', ·)` ở bước thường. Khi `done=True`, target bằng `r` và runner không mã hóa state kế tiếp. Replay từ log khớp nhánh RNG, Q trước/sau từng bước và checkpoint đã lưu. |
| Reward và thời gian | `SumoEnv.step()` trả `−queue_seconds / elapsed` từ các mẫu hàng đợi làn vào, tính cả vàng, đỏ và clearance. Trong 15 episode Cologne1 của Original, 172 quyết định chưa kết thúc kéo dài 30 giây khi giữ pha và 1.213 quyết định kéo dài 40 giây khi đổi pha. Cả hai dùng cùng `γ=0,95` mỗi quyết định. Tổng reward episode cộng các trung bình theo quyết định, còn Average Queue Length chia tổng queue-seconds cho 3.600 giây mô phỏng. |
| Kết quả và phục vụ | Reward không đo trực tiếp completion, teleport, xe chưa vào mạng hoặc thời gian movement thiếu xanh. `trip_metrics()` tính các trung bình thời gian trên nhóm xe hoàn thành hợp lệ; `signal_metrics()` tính queue toàn horizon và khoảng không có `G`. Đọc gap, completion và teleport cùng các trung bình thời gian. |

Ví dụ Original episode 15 có 93 quyết định, tổng reward **−5.298,24** và **205.057 xe·s** queue, tương ứng **56,96 xe** dừng trung bình trong 3.600 giây. Đồng hồ mô phỏng và số quyết định là hai đơn vị khác nhau. Công thức Bellman một bước hiện tại đúng với mục tiêu chiết khấu theo quyết định. Nếu sau này đổi mục tiêu sang chiết khấu theo giây, một thử nghiệm riêng cần cộng reward từng giây đã chiết khấu và dùng `γ_giây^elapsed` cho phần bootstrap, với bootstrap bằng 0 ở terminal. [Q-Learning, mục 6.5](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) · [Sutton, Precup và Singh, 1999](https://all.cs.umass.edu/pubs/1999/sutton_ps_AI99.pdf)

## Vì sao Cologne1 có thể lặp pha

Trace test lịch sử seed `31001`, repeat 1, có 104 quyết định với cùng state `(2,2,2,2,0,0,0,0,7)` nhưng 57 vector hàng đợi đầu vào theo edge khác nhau. Q của action `7` cao nhất duy nhất và ô này đã nhận 7 update. Khi giữ pha `7`, phần ID pha giữ nguyên và queue thay đổi vẫn có thể rơi vào cùng bin. Vì vậy đổi tie-break sẽ không đổi action trong chuỗi này. Lượt đó ghi **212/2.015** chuyến hợp lệ, **99** sự kiện teleport và gap lớn nhất **3.520 giây** dù mỗi movement từng nhận `G` ít nhất một lần. Đây là bằng chứng về cơ chế lặp trong một trace lịch sử. Mức đóng góp riêng của state, reward, coverage và biến thiên SUMO chưa được định lượng. [Actions](stage7/test/cologne1/QL-15ep-e7/seed_31001/repeat_1/actions.csv) · [Result](stage7/test/cologne1/QL-15ep-e7/seed_31001/repeat_1/result.json)

Tie-break vẫn có thể ảnh hưởng ở lượt khác. Trên test Stage 8, tỷ lệ quyết định có nhiều action cùng Q cao nhất tính trung bình theo lượt là **11,36%** cho Original và **31,96%** cho NewState. Với reward âm, ô khởi tạo bằng 0 có thể đứng trên các ô đã học mang giá trị âm. [Stage 8](stage8/README.md)

| Model sau 15 episode | State đã gặp | Cặp có update / mẫu số | Đổi greedy từ checkpoint 14 sang 15 |
|---|---:|---:|---:|
| Cologne1 Original | 129 | 500/1.032 · **48,45%** | 26/124 · **20,97%** |
| Cologne1 NewState, thêm bin 15+ | 213 | 703/1.704 · **41,26%** | 45/207 · **21,74%** |
| Ingolstadt1 Original, exploration seed 7 | 17 | 49/51 · **96,08%** | 8/17 · **47,06%** |

Mẫu số coverage dùng số state đã gặp khi train nhân số action hợp lệ. Trong Original Cologne1, 276/500 cặp đã gặp chỉ có một update. Episode 15 vẫn thêm 5 state ở Original và 6 state ở NewState. Tỷ lệ đổi greedy chỉ tính trên state chung của hai checkpoint, và mức thay đổi tuyệt đối lớn nhất của Q lần lượt là **15,49** và **17,45**. Các số liệu này chưa chứng minh hội tụ. [Stage 7 training](stage7/analysis/training.csv) · [Stage 8 training](stage8/training.csv)

## Thay đổi nên thử đầu tiên

1. **Đối chứng phải dùng cùng bảng Q.** Dùng `QL-Original` với bin `0 4`. Lượt có Guard thêm đúng `--guard-seconds 300`. Guard xét thời gian theo đồng hồ SUMO tại điểm quyết định, ưu tiên edge có nhu cầu quá hạn lâu nhất và chọn action có Q cao nhất trong các pha cấp `G` cho edge đó. Dùng cùng `SumoEnv.step()` như đối chứng. Với mỗi bảng Q từ một exploration seed ở Giai đoạn 4, chạy cả hai cấu hình trên bảng đó.
2. **Kiểm tra tác động trên validation trước khi khóa.** Stage 8 đã thử Guard trên cùng bảng *NewState*: completion validation trung bình từ **71,96% lên 76,33%**, gap lớn nhất từ **2.930 xuống 1.470 giây**. Original đạt **83,75%** trên validation lịch sử. Vì thế phép thử mới hỏi Guard có giúp khi gắn vào **Original** hay không. Dùng 300 giây đã định trước trong Stage 8. Đề xuất chốt trước tiêu chí cải thiện là giảm gap lớn nhất trong các lượt lặp của từng seed validation và giữ hoặc tăng completion trung bình. Báo mọi lượt, gồm cả queue, teleport và completion thấp nhất. Khóa cấu hình/checkpoint sau validation, dùng seed test mới và so FT, MP, QL, Guard với cùng cấu hình mô phỏng. [Lựa chọn đã khóa ở Stage 8](stage8/selection.json)
3. **Giữ thay đổi state cho phép thử riêng.** Thử mã hóa lại 1.400 quyết định train đã lưu bằng một trường bổ sung ghi edge quá hạn lâu nhất hoặc giá trị 0 nếu chưa có. Số state tăng từ **129 lên 182** cùng trace, coverage từ **48,45% xuống 39,35%** (573/1.456). Trong 104 quyết định bị gộp ở seed lịch sử, dấu hiệu mới tách 9 quyết định đầu rồi vẫn gộp 95 quyết định còn lại vào cùng khóa. Đây là phép đếm trên trace cố định. Một policy học lại có thể đi qua tập state khác. Chưa có căn cứ để ưu tiên mở rộng state trước phép thử Guard.

Tại quyết định 22 của trace lịch sử, edge `-32038056#3` đã chờ phục vụ quan sát được 310 giây. Các action cấp `G` cho edge này là `1, 2, 3`; bảng Q đã khóa xếp action `1` cao nhất trong ba action. Kiểm tra offline vì thế cho action dự kiến **7 → 1** tại điểm đó. Kết quả mô phỏng sau khi thực thi action khác chưa được đo. Guard theo dõi nhu cầu theo edge và vẫn có thể bỏ sót movement dùng chung làn hoặc chờ thêm trong clearance. Ngưỡng 300 giây là điều kiện xét tại điểm quyết định.

## Kiểm tra và tệp để thuyết trình

- `.\.venv\Scripts\python.exe -m pytest -q` chạy toàn bộ suite với **176 pass, 0 fail**.
- `tools/stage7_summary.py --output <thư_mục_mới>` replay **76 episode, 90 evaluation và 4.720 quyết định greedy**. `tools/stage8_summary.py --output <thư_mục_mới>` replay **30 episode, 36 evaluation**. Hai tập replay có dùng chung model Original và đầu ra được ghi riêng trong thư mục tạm.
- Input và pha hợp lệ ở [`SumoEnv.get_state()` và `step()`](../src/sumo_env.py). Mã hóa, epsilon-greedy và Bellman ở [`encode_state()`, `choose_action()`, `update_q_value()`](../src/q_learning.py). Guard ở [`greedy_steps()`](../src/evaluate.py). Metric ở [`trip_metrics()` và `signal_metrics()`](../src/evaluation_metrics.py).
