# Stage 8 — State, chống bỏ đói và độ ổn định Q-Learning

## Kế hoạch chốt trước thí nghiệm

Thực hiện tuần tự theo yêu cầu Stage 8; không DQN, không sửa FT/MP,
SumoEnv, reward, mạng/route/timing, alpha/gamma hoặc tie-break. Không commit/push.
Git ban đầu sạch trên main; baseline: **157 passed trong 15,33 s**.

1. Đối chiếu raw Stage 7 seed31001 repeat1: 104 quan sát state
   `(2,2,2,2,0,0,0,0,7)` chứa 57 vector queue edge khác nhau. Range incoming:
   34–120, 13–32, 9–14, 12–18 xe. Q7=-66,658930 là cực đại duy nhất;
   action7 chỉ mở movement8/9/18/19. Không quy toàn bộ lỗi cho encoding.
2. Thêm queue_bins=(0,4,14), giữ mặc định (0,4), metadata và load kiểm tra
   encoding tường minh. Test ngưỡng, đọc model cũ, mismatch, replay bin mới.
3. Cologne1 train từ bảng rỗng 15 episode; SUMO23423, exploration7,
   alpha0,1/gamma0,95, epsilon tuyến tính1→0,05. Không đổi thành phần khác.
4. Validation42/12345, **2 repeats mỗi seed** cho Original và NewState.
   Checkpoint5/10 được chọn trước để validation mỗi seed một lần; không chọn
   checkpoint bằng test. So completion/throughput, teleport, queue, movement,
   gaps, repeats, unseen/tie và coverage. Nếu vẫn có chuỗi giữ action >=300s
   hoặc movement gap>=600s, thử Guard riêng sau khi lưu kết quả state-only.
5. Guard nếu cần: ngưỡng **300s**, chỉ áp dụng evaluate; tuổi nhu cầu quan sát
   theo incoming edge ở thời điểm quyết định, reset khi queue=0 hoặc edge
   nhận G thực tế; ưu tiên edge quá hạn lâu nhất và action có G cho edge đó,
   ưu tiên Q cao nhất, hòa lấy action đầu. Không chứng nhận từng movement
   trên lane dùng chung; mọi action qua SumoEnv.step. Không sửa reward/model.
6. Không tăng lên30 episode nếu15ep vẫn bỏ đói rõ và coverage thưa: thêm
   ngân sách không tự chữa giới hạn state/reward. Ghi căn cứ sau kết quả.
   Chỉ mở Ingolstadt1 sau khi Cologne1 đã được phân tích rõ và state mới có
   lợi đủ để biện minh việc mở rộng; không chạy chỉ để tăng số lượt.
7. Khóa model cuối15ep và Guard300 (nếu có), không chọn lại sau test.
   Seeds xác nhận mới **41001,41002,41003**, mỗi seed **2 repeats**, cùng
   Original/NewState/Guard nếu có. Bộ31001–31003 đã biết, chỉ dùng điều tra.
   Xếp theo completion mean, min completion, teleport mean, max gap; báo cáo
   đầy đủ biến thiên, không gọi đây là kiểm định thống kê hoặc thắng mọi metric.
8. Dùng lại replay_training, coverage, checkpoint_changes, metric chung;
   một script offline Stage8 nếu công cụ cũ chưa đủ. Không framework.
   Giữ kết quả/log/checkpoint bắt buộc; không thêm trace mỗi giây đã có signals.
9. LEARNING_GUIDE.md, reports/README.md, summary.csv và kết luận tiếng Việt;
   chạy toàn bộ pytest, kiểm SHA mọi file tracked ban đầu; chỉ file code/test
   có chủ đích được đổi. Không di chuyển/xóa dữ liệu lịch sử.

File dự kiến sửa: q_learning.py, evaluate.py, analyze_ql.py và test tương ứng.
File code mới dự kiến: một script stage8_summary.py và một test nếu cần.
Các giả định/tiến độ và kết quả thực tế sẽ được bổ sung dưới đây.

## Quyết định sau validation, trước test xác nhận

Đã lưu snapshot độc lập `state_only/` sau12 lượt Original/NewState/checkpoint,
trước khi viết/chạy Guard. State mới thất bại ở seed42: cả hai lần1077/2015,
74 teleport, gap2930s, giữ cùng action870s. Do đó thực hiện Guard300 như một
biến thể riêng, không thay reward hay model. Guard chỉ chạy evaluate.

Validation cuối có4 lượt/biến thể: Original83,7469%, NewState71,9603%,
Guard76,3275%; teleport mean42,75/52,50/42,75; gap max2060/2930/1470s.
`selection.json` khóa Original theo tiêu chí đã ghi; final15ep và Guard300
được khóa trước18 lượt xác nhận. Không chọn checkpoint5/10 bằng test.

Không mở Ingolstadt1 và không tăng lên30ep: state-only validation kém hơn
Original, lặp pha vẫn dài, coverage mới41,26% thấp hơn48,45%, checkpoint14→15
vẫn đổi greedy45/207 và max|ΔQ|17,45. Chưa có cơ sở để chỉ tăng ngân sách.

Sau khi khóa, thêm đúng2 diagnostic NewState seed31001 để trả lời trực tiếp
ca action7 đã biết. Không dùng diagnostic để chọn model; không gộp với18
lượt test mới. Matrix cuối:12 validation +4 checkpoint validation +18 test
+2 diagnostic=36. Không chạy diagnostic Guard vì không cần để xác định
vòng lặp action7 của state-only.

## Kết luận

**State4bin giảm một phần mất thông tin nhưng không cải thiện Q-Learning
thuần trong thí nghiệm này. Guard giảm giữ pha và có kết quả test mô tả tốt
hơn Original, nhưng không thắng validation và chưa bảo đảm từng movement.**
Không có bằng chứng Q/policy ổn định hơn hoặc hội tụ. Không triển khai DQN,
không sửa FT/MP, SumoEnv, metric, reward, mạng/route/timing hoặc model cũ.

Đã train mới đúng15 episode Cologne1 và chạy36 evaluation đủ3600s:
12 validation,4 checkpoint validation,18 test mới,2 diagnostic. Không bỏ lượt
xấu, không train30ep, không mở Ingolstadt1. Phân tích còn replay15 episode
Original đã lưu: **30 episode được kiểm chứng, chỉ15 episode train mới**.

## 1. Vì sao ca Stage7 bị kẹt action7?

Nguồn: `../stage7/test/cologne1/QL-15ep-e7/seed_31001/repeat_1/` và model
được khóa trong `../stage7/models/selection.json`.

- `SumoEnv.get_state()` vẫn đọc queue thật. Trong104 lần state
  `(2,2,2,2,0,0,0,0,7)`, có57 vector queue edge khác nhau; range theo thứ tự
  edge là34–120,13–32,9–14,12–18 xe. Queue thay đổi, encoding vẫn đồng nhất.
- `encode_state()` cũ gộp mọi queue>=5 vào2; outgoing đều0 trong các quan
  sát này. `step(7)` giữ `current_phase=7`, nên chiều phase cũng không đổi.
- `choose_action()` evaluate epsilon0, không update: Q theo action0..7 là
  `[-67.756527,-71.571199,-76.039969,-72.283109,-67.788100,-67.634851,
  -76.291277,-66.658930]`. Q7 là cực đại duy nhất, đã update7 lần
  (số update đầy đủ `[6,9,11,8,7,6,9,7]`). Đây không phải tie/unseen/chưa học.
- `step()` cho phép giữ cùng xanh thêm30s, không có max-green hoặc tuổi phục
  vụ trong state. Action7 chỉ mở G ở movement8/9/18/19, thuộc hai edge
  23429231#1 và27115123#3; hai edge -32038056#3 và28198821#3 không được
  action này cấp G. Các movement khác có gap3150–3520s trong lượt đã lưu.
- Reward chỉ đo queue incoming boundary trung bình mỗi decision; không chứa
  trực tiếp completion, xe chưa chèn, teleport hoặc tuổi thiếu G. Gộp lane
  còn mất thông tin rẽ, finite training/coverage và chất lượng Q vẫn là giới
  hạn. Không định lượng được phần lỗi do từng yếu tố; không gọi encoding
  là nguyên nhân duy nhất hoặc kết luận agent cố ý tạo teleport.

## 2. State mới và tương thích model

`queue_bins` là các cận trên inclusive. Mặc định `(0,4)` giữ nguyên; truyền
`(0,4,14)` tạo0/1–4/5–14/>=15. Ngưỡng này là thử nghiệm, không phải tối ưu.
Outgoing giữ nguyên. Reward, alpha0,1, gamma0,95, tie-break first và RNG7
không đổi. Epsilon vẫn tuyến tính1→0,05 qua15 episode với SUMO23423 cố định.

Metadata schema1 đã có trường encoding nên **không cần đổi format**.
`build_encoding()` kiểm ngưỡng tăng dần/nguyên/bắt đầu0; `load_q_table()`
kiểm encoding được yêu cầu và miền bin của từng row. Model mới phải truyền
`--queue-bins 0 4 14` khi load/evaluate; mặc định cũ từ chối model mới, và
load model cũ với bins mới cũng bị từ chối. Không chỉnh metadata model cũ.

Trên104 quan sát bão hòa cũ, bins mới tách thành3 state, nhưng **100/104**
vẫn cùng `(3,3,2,3,0,0,0,0,7)`; ba quan sát thành `(3,3,2,2,...,7)`, một
thành `(3,2,2,3,...,7)`. Vì vậy có giảm aliasing, chưa giữ được queue34 khác
120, tuổi chờ, nhu cầu từng movement hoặc xe ngoài vùng đo.

Diagnostic NewState31001: cả2 lần1572/2015,34 teleport, action7 chỉ6/100
decision thay vì105/117 của Original Stage7 đã lưu; chuỗi dài nhất10
decision/300s, gap870s, đủ20/20 movement. **Vòng action7 cụ thể không tái
xuất hiện trong hai lượt này**, nhưng kết quả dưới đây chứng minh kẹt pha
khác vẫn còn. Không gọi diagnostic là test chưa biết hay bằng chứng tổng quát.

## 3. Validation, checkpoint và state-only thất bại ở đâu?

Các dòng chính là mean4 lượt, hai seed×hai repeat; gap/giữ xanh là max qua
lượt. Checkpoint5/10 chỉ có2 lượt, một lượt mỗi seed, được chốt trước khi chạy.

| Biến thể | N valid / xe/h | Completion | Queue xe | Teleport events | Gap max s | Giữ cùng xanh max s |
|---|---:|---:|---:|---:|---:|---:|
| Original15ep | 1687,50 | 83,75% | 53,69 | 42,75 | 2060 | 510 |
| NewState15ep | 1450,00 | 71,96% | 69,69 | 52,50 | 2930 | 870 |
| NewState-Guard300 | 1538,00 | 76,33% | 65,29 | 42,75 | 1470 | 270 |
| NewState ep05, N=2 | 1470,50 | 72,98% | 121,26 | 40,00 | 3000 | 570 |
| NewState ep10, N=2 | 1758,50 | 87,27% | 69,92 | 17,00 | 790 | 90 |

Episode15 không tốt hơn10 trên validation đã quan sát. Không chọn lại ep10
để test: thiết kế chỉ xét final15ep; số repeat khác nhau và đây không phải
ước lượng phân phối đầy đủ. `state_only/` giữ snapshot trước Guard, gồm raw
summary/runs/training/evidence; các run gốc không bị ghi đè.

Seed42 NewState lặp state `(0,3,2,3,0,0,0,0,1)`57 lần, chọn1 ở cả57 lần;
Q1=-24,130631 là max duy nhất giữa8 action (xem `evidence.json` và
`validation/.../actions.csv`). Có15 vector lane queue khác nhau trong state
này. Một state khác `(3,3,3,0,0,0,0,0,0)` có24 decision với hàng Q toàn0,
chọn0 theo tie-break. Hai cơ chế **argmax đã học** và **unseen/tie** cùng
xuất hiện; thêm bin không tự sửa cả hai.

Action1 chỉ mở edge -32038056#3, đang ở bin0 trong state lặp57 lần này;
ba edge còn lại có queue dương. Như vậy policy có thể giữ xanh cho edge
không có queue dừng quan sát được. Đây là bằng chứng giới hạn thứ hạng Q
đã học và khả năng quan sát/coverage, không phát hiện sai công thức update.

## 4. Guard đo và bảo đảm điều gì?

`evaluate.greedy_steps(..., guard_seconds=300)` theo dõi `demand_since` của
từng incoming edge ở thời điểm quyết định. Queue edge>0 bắt đầu/giữ tuổi;
quan sát queue0 reset. Khi tuổi>=300s, chọn edge quá hạn lâu nhất (hòa theo
thứ tự edge), rồi action hợp lệ có G trên ít nhất một controlled link của
edge đó với Q cao nhất (hòa theo thứ tự action). Có G **thực** trong samples
thì reset; chỉ yêu cầu action mà không có G không được coi là đã phục vụ.

Mọi action gọi cùng `SumoEnv.step()`; Guard không set đèn, không sửa reward,
không học lại hoặc sửa bảng Q. Log thêm greedy_action, guard_edge, guard_age
trong actions.csv có sẵn. `audit_guard_decisions()` replay từ raw queues và
G thực để kiểm cả nhu cầu, tuổi, edge ưu tiên, override bị bỏ sót và action.

Đây là **QL có ràng buộc phục vụ edge**, không gọi Tabular QL thuần. Một
edge có G ở một movement không chứng minh movement đang có xe đã được
phục vụ. Queue chỉ quan sát tại quyết định; chưa đo liên tục từng movement.
Ngưỡng300s cũng không phải cận waiting xe: có độ trễ decision/chuyển pha,
nhiều edge cùng quá hạn và có thể clearance kéo dài. Test vẫn có gap1470s.
Không ép mở hướng không có queue quan sát được.

## 5. Test xác nhận mới: kết quả đầy đủ

Seeds41001/41002/41003, hai repeat/seed, cùng demand2015, horizon3600s,
binary/input/metric/timing và logic an toàn. Mỗi dòng mean6 lượt; không
gộp với validation, diagnostic31001 hoặc các test Stage7.

| Biến thể | N valid / xe/h | Completion | Queue xe | Teleport events | Movement có G | Gap max s | Giữ xanh max s |
|---|---:|---:|---:|---:|---|---:|---:|
| QL-Original | 1542,33 | 76,54% | 75,35 | 50,50 | 20/20 mọi lượt | 2120 | 930 |
| QL-NewState | 1336,67 | 66,34% | 75,62 | 55,67 | 20/20 mọi lượt | 2930 | 1140 |
| QL-NewState-Guard | 1713,33 | 85,03% | 64,94 | 38,33 | 20/20 mọi lượt | 1470 | 270 |

Guard so Original: +171 xe/h, +8,49 điểm phần trăm completion, -12,17
teleport/lượt, -10,41 xe queue trong bộ này; thắng completion5/6 lượt,
riêng41003 repeat1 là1507<1623. NewState thuần giảm205,67 xe/h và10,21 điểm
phần trăm, không chứng minh cải thiện so Stage7. Guard là biến thể có mean
test tốt nhất trong ba biến thể đã chạy, **không phải lựa chọn validation**.
Không chọn lại Guard/ep10 sau khi xem test hoặc gọi đó là đánh giá độc lập
của một policy đã chọn Guard trước. `selection.json` vẫn giữ Original.

| Seed | Original r1/r2 | NewState r1/r2 | Guard r1/r2 |
|---|---|---|---|
| 41001 | 1539/1565 | 1358/1706 | 1770/1770 |
| 41002 | 1452/1452 | 878/1806 | 1805/1722 |
| 41003 | 1623/1623 | 1159/1113 | 1507/1706 |

Range cùng seed lớn nhất: Original26 xe, NewState928 xe (46,05 điểm phần
trăm), Guard199 xe (9,88 điểm phần trăm). Guard validation12345 cũng chênh
306 xe. Không bỏ lượt878 hoặc các repeat xấu. Hai repeat/seed chưa đủ để
kết luận có ý nghĩa thống kê hoặc bảo đảm tổng quát. Stage7 Original mean
43,29% trên31001–31003 không được dùng làm đối chứng trực tiếp với85,03%
trên bộ seed mới. Không chạy FT/MP mới, không tuyên bố vượt baseline.

| Biến thể, test | Wait s/xe | Travel s/xe | TimeLoss s/xe | Unfinished mean | Undeparted mean |
|---|---:|---:|---:|---:|---:|
| Original | 165,56 | 210,57 | 187,19 | 121,67 | 301,50 |
| NewState | 169,30 | 210,84 | 186,13 | 126,83 | 499,67 |
| Guard | 157,29 | 199,55 | 176,61 | 104,67 | 159,33 |

Các trung bình thời gian là mean của mean từng lượt trên cohort arrived,
không vaporized và chưa từng teleport; travel=arrival−actual depart,
waiting/timeLoss theo SUMO, không cộng departDelay. Xe chưa xong/chưa chèn
không nhận0 hoặc vào cohort. Completion=N/requested; throughput=N×3600/horizon.
Queue=queue_seconds/3600 trên incoming lane duy nhất, đủ3600 mẫu1s và mọi
stage. Teleport ở bảng là số sự kiện start, khác số ID teleport (runs.csv có
cả hai). Các lượt test có0 collision; điều này không chứng nhận mọi điều
kiện an toàn hoặc giải quyết nguyên nhân SUMO Stage6.1.

## 6. Coverage, action distribution và mức ổn định

Coverage lấy số cặp **thực update**/(state thực gặp×action), không suy từ Q=0.

| Model train | State | Cặp đã update / có thể | Coverage | Action chưa update |
|---|---:|---|---:|---:|
| Original | 129 | 500/1032 | 48,45% | 51,55% |
| NewState | 213 | 703/1704 | 41,26% | 58,74% |

| Test, mean tỷ lệ mỗi lượt | Unseen | Tie max Q | Action được chọn chưa update |
|---|---:|---:|---:|
| Original | 1,35% | 11,36% | 11,36% |
| NewState | 19,46% | 31,96% | 34,20% |
| Guard | 7,87% | 28,19% | 30,93% |

Guard dùng **cùng bảng** NewState, làm thay đổi vùng state được thăm;
unseen ít hơn không có nghĩa bảng được học thêm. Tie ở Guard là tie của Q
trước ràng buộc; action được chọn chưa update tính theo action thực thi.

Tổng số lựa chọn action0..7 trên6 test, đọc từ actions.csv:

| Original | NewState | Guard |
|---|---|---|
| `[88,108,70,18,26,6,159,117]` | `[156,293,9,16,51,26,53,26]` | `[86,218,21,31,80,49,82,19]` |

Chuỗi dài nhất tương ứng31/38/9 decision. Guard override khác greedy tổng32
lần trên6 test, mean5,33/lượt; không phải Q đã đổi. Từng state gây lặp, Q
của mọi action, số quan sát raw khác nhau và gap **từng movement/từng run**
ở `evidence.json`; distribution và repeat mỗi lượt ở `runs.csv`, raw ở
actions/signals/result.json. Tất cả36 evaluation đều có20/20 movement có G;
một lần G không đủ để kết luận hết bỏ đói.

| Chuyển checkpoint | Original đổi greedy / state chung; max ΔQ | NewState đổi greedy / state chung; max ΔQ |
|---|---|---|
| 10→11 | 24/110;19,02 | 29/177;17,26 |
| 11→12 | 28/117;18,73 | 35/190;19,23 |
| 12→13 | 31/122;19,54 | 30/197;31,95 |
| 13→14 | 27/122;14,06 | 46/202;16,09 |
| 14→15 | 26/124;15,49 | 45/207;17,45 |

Episode15 thêm5 state Original,6 state NewState. Tổng reward từng episode,
state mới, coverage, epsilon, thay đổi Q/policy nằm trong `training.csv`;
replay kiểm toàn bộ RNG, Bellman target, done và mỗi checkpoint, không chỉ
so reward. State giữa hai encoding có nghĩa khác nhau nên không so trực
tiếp các ô Q hoặc gọi số row lớn hơn là tốt hơn.

Tiêu chí đọc độ ổn định: xem đồng thời độ lớn ΔQ trên state chung, tỷ lệ đổi
greedy, state mới, coverage/unupdated và validation5/10/15; không dùng một
đại lượng đơn lẻ. Năm checkpoint cuối vẫn đổi policy nhiều và sinh state,
validation không tăng đều. **Không có bằng chứng ổn định hơn; chưa chứng
minh hội tụ toán học.** Guard là ràng buộc evaluate, không cải thiện hội tụ Q.

## 7. Đơn giản hóa, kiểm thử và dữ liệu phải giữ

- Sửa `q_learning.py`: cấu hình bins, lưu metadata sẵn có, validate/load và
  CLI. Công thức Q/epsilon/tie-break/reward không thay.
- Sửa `evaluate.py`: truyền bins và thêm Guard tùy chọn ngay trong generator
  có sẵn; tiếp tục dùng một runner/metric/SumoEnv cho mọi lượt.
- Sửa `analyze_ql.py`: thay các chỗ hard-code3bin bằng ngưỡng metadata;
  dùng stdlib `bisect_left`, vẫn phân tích offline độc lập runner SUMO.
- Sửa2 test cũ, thêm `test_stage8.py`; thêm đúng **2 file Python mới**:
  `stage8_summary.py`229 dòng và `test_stage8.py`143 dòng. Tổng Python tăng
  **461 dòng** gồm test. Không có class/framework/dependency mới. Luồng học
  dễ tìm hơn nhờ tài liệu, **không tuyên bố repository ít dòng hơn**.
- Đã rà q_learning/analyze_ql/stage7_summary/stage6_1_probe/evaluate và caller.
  Các vòng ghi signals có mục đích/cohort khác nhau; hash file trùng vài dòng
  không đáng tạo module chung hoặc buộc script offline import runner. Không
  xóa/gộp logic đang cần chỉ để ngắn, không sửa stage7_summary/probe. Tái dùng
  `replay_training`, `coverage`, `checkpoint_changes`, `summarize`, `write_csv`
  cho Stage8 thay vì viết lại phân tích Q/checkpoint/metric.
- `LEARNING_GUIDE.md` khoảng3 trang đọc, phân biệt thuật toán/thực nghiệm/
  điều tra. `reports/README.md` chỉ nơi model, summary, raw và cách lưu lâu dài.

Baseline157 passed; kiểm tra cuối **176 passed,0 failed,0 skipped/deselected,
16,11s** (`tests_final.log`, lần cuối). Tất cả test FT/MP/QL/SumoEnv cũ đều
chạy, không chỉ offline. Test mới kiểm bins/save-load/mismatch, ngưỡng sai,
Guard chỉ có nhu cầu/reset và qua step, model frozen, metric SUMO ngắn,
repetition và matrix thiếu/trùng/sai seed, replay Guard phát hiện override
bị bỏ sót/nhu cầu giả. Các test công thức Q/done, epsilon/greedy, action và
chuyển pha an toàn cũ giữ nguyên.

`raw_verification.json`: **36/36** evaluation đạt verifier XML/CSV độc lập
Stage6.1. Script Stage8 kiểm đủ matrix36, model/SHA/encoding, seed path và
metadata, G thực, replay Guard, coverage và checkpoint; từ chối output cũ.
`compatibility.json`: tái tạo8 file Stage6.2 và6 file Stage7 **khớp byte**;
snapshot state_only4 file cũng khớp byte qua `--state-only`. Các kiểm tra
phân tích không khởi chạy lại simulator. Không dùng `python -O` vì audit có assert.

Giữ toàn bộ model/checkpoint/train CSV, mọi evaluation repeat kể cả878 xe,
tripinfo/signals/actions/statistics/collision/warnings/command/metadata,
selection/test manifest và các tệp kiểm chứng. Stage8 khoảng53MB raw và
bảng phân tích; không thêm FCD/trace từng giây ngoài recorder sẵn có.
`state_only/` là snapshot trước Guard có chủ đích, không thay thế raw.
Không di chuyển/xóa dữ liệu đã commit; archive/Git LFS chỉ là đề xuất trong
`reports/README.md`. SHA trước/sau được đối chiếu trong `preservation_after.json`.

## 8. Tái tạo và vị trí thuyết trình

Chạy từ gốc dự án. Các lệnh train và evaluate dưới đây dùng thư mục mới.

```powershell
.\.venv\Scripts\python.exe src/q_learning.py train cologne1 --episodes 15 --seed 23423 --exploration-seed 7 --queue-bins 0 4 14 --output reports/stage8/training_repeat/cologne1
.\.venv\Scripts\python.exe src/evaluate.py --algorithm QL --map cologne1 --seed 42 --queue-bins 0 4 14 --model reports/stage8/training_repeat/cologne1/cologne1_q_table.json --output reports/stage8/validation_repeat/cologne1/QL-NewState/seed_42/repeat_1
.\.venv\Scripts\python.exe src/evaluate.py --algorithm QL --map cologne1 --seed 42 --queue-bins 0 4 14 --guard-seconds 300 --model reports/stage8/training_repeat/cologne1/cologne1_q_table.json --output reports/stage8/validation_repeat/cologne1/QL-NewState-Guard/seed_42/repeat_1
.\.venv\Scripts\python.exe tools/stage8_summary.py --output reports/my_stage8
.\.venv\Scripts\python.exe tools/stage8_summary.py --state-only --output reports/my_state_only
.\.venv\Scripts\python.exe tools/plot_stage8.py --stage8 reports/stage8 --output reports/my_stage8_plots
.\.venv\Scripts\python.exe -m pytest -q
```

Các lệnh tổng hợp và vẽ đọc kết quả lịch sử trong `reports/stage8`. Đặt
`--output` vào thư mục mới để giữ các bảng và ảnh đã lưu.

Lặp evaluation theo matrix ở kế hoạch/selection và `test_manifest.json`,
thay seed/repeat/output tương ứng; Original bỏ queue-bins mới và dùng model
Stage7. `command.txt` từng lượt lưu lệnh SUMO thực; metadata lưu phiên bản,
source/code hashes, begin/end, action/link mapping và model hash. Code hiện
tại được kiểm trong `preservation_after.json`; mô phỏng cùng seed vẫn có
thể khác kết quả đã lưu, phân tích offline phải khớp raw.

| Kiến thức | File/hàm |
|---|---|
| Input queue/phase | `sumo_env.py: SumoEnv.get_state()` |
| State và cận bin | `q_learning.py: build_encoding(), encode_state()` |
| Epsilon-greedy và tie | `q_learning.py: choose_action(), episode_epsilon()` |
| Công thức Q, done | `q_learning.py: update_q_value()` |
| SUMO, vàng/đỏ/clearance và reward | `sumo_env.py: step(), _run_signal()` |
| Train/checkpoint/save/load | `q_learning.py: train(), run_episode(), save_q_table(), load_q_table()` |
| Frozen evaluation và Guard edge | `evaluate.py: greedy_steps(), run_evaluation()` |
| Metric/cohort | `evaluation_metrics.py: trip_metrics(), signal_metrics(), read_metrics()` |
| Lặp pha, replay Guard, đủ lượt | `stage8_summary.py: repetition_evidence(), audit_guard_decisions(), verify_runs()` |
| Update/RNG và Q/policy checkpoint | `analyze_ql.py: replay_training()`; `stage7_summary.py: coverage(), checkpoint_changes()` |

## 9. Những việc chưa được giải quyết

Chưa sửa tính không xác định phía SUMO, chưa đo nhu cầu từng movement hoặc
xe ngoài vùng boundary, chưa bảo đảm max gap/waiting, chưa có model QL thuần
tốt hơn đáng tin cậy, chưa có bằng chứng hội tụ. Guard còn teleport và gap
dài, không được gọi là công bằng tuyệt đối. Chưa có train replicate độc lập
hoặc thống kê đủ mạnh để quy cải thiện cho riêng một nguyên nhân.
Không dùng test để chỉnh model; chưa mở map/nhu cầu khác, không train30ep
hoặc triển khai DQN. Kết quả Guard đáng nghiên cứu tiếp trong một nhiệm vụ
riêng, chưa đủ cơ sở thay lựa chọn validation hoặc khẳng định thành công toàn diện.
