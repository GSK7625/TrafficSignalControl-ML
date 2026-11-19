# Giai đoạn 3 — Phân tích Q-Learning trên Cologne1

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
