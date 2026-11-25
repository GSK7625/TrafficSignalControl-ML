# Giai đoạn 5 — Reward theo queue và tuổi chờ xanh trên Cologne1

**Giữ reward queue gốc (λ=0).** Cả ba model chọn cùng chuỗi 25 pha trên từng seed trong cả hai lần chạy đối chứng ngắn. Tuổi chờ xanh cực đại của edge còn xe dừng là 620 giây ở cả 12 lượt validation. Hai mức phạt dương đã thay đổi các giá trị Q nhưng chưa cải thiện chỉ tiêu phục vụ.

## Reward gốc đo gì?

Trong `SumoEnv._run_signal()`, sau mỗi giây SUMO, `queue` là tổng số xe dừng trên các làn vào do nút điều khiển. Mỗi làn xuất hiện một lần. `queue_seconds = Σ queue × Δt` có đơn vị xe·giây. `SumoEnv.step()` lấy `elapsed = finish − start` từ đồng hồ mô phỏng và trả `−queue_seconds / elapsed`. Cả vàng, tất cả đỏ, thời gian chờ giải tỏa và phần action bị cắt ở cuối episode đều dùng đúng số giây thực tế. [Môi trường và chuyển pha](../src/sumo_env.py)

Reward gốc đo được chi phí của việc giữ pha khi xe dừng trên làn vào tích lại. Đổi pha cũng có thể làm queue tăng trong lúc chuyển tiếp. Cùng một tổng queue trung bình, reward bằng nhau dù một hướng chờ liên tục rất lâu. Xe chưa vào mạng và thời gian thiếu xanh của từng movement không có số hạng riêng. Trace lịch sử đã ghi nhận giữ một pha 3.190 giây xanh với một lựa chọn greedy cực đại duy nhất tại state lặp. State bị gộp và độ phủ của Q cũng có liên quan. Chưa xác định được mức đóng góp riêng của reward vào chuỗi đó. [Phân tích state và Q](q_learning_review.md)

## Một thay đổi trong reward

Giữ state QL-Overdue, action, ε-greedy, Bellman và cơ chế chuyển pha. Tại **cuối** mỗi action, kể cả bước terminal:

1. `EdgeDemand.record_green()` đọc tín hiệu thực đã chạy. Nếu bất kỳ movement nào của một edge đã nhận `G`, đặt lại đồng hồ của edge. Lệnh đổi pha kết thúc khi còn vàng/đỏ chưa phải là một lần cấp xanh.
2. `EdgeDemand.observe()` đọc queue mới theo từng edge. Queue bằng 0 xóa đồng hồ; khi mới thấy queue dương, bắt đầu đếm từ thời điểm quan sát. Gọi `A` là tuổi lớn nhất (giây) trong các edge còn queue; nếu tất cả đều trống, `A=0`. Lấy cực đại một lần, không cộng lại xe hoặc làn theo số movement.
3. `P = min(1, max(0, (A − 300) / 300))`. Reward mới là `r = −mean_queue − λP`. `P` không có đơn vị. `λ` có đơn vị tương đương số xe dừng trung bình, cũng là mức phạt tối đa mỗi action. Giữ cách chiết khấu theo quyết định. [Công thức](../src/ql_reward.py) · [Quan sát trước khi cập nhật Q](../src/q_learning.py)

Ví dụ, 300 xe·giây trong 30 giây cho mean queue 10. Khi một edge có queue đã chờ 450 giây, `P=0,5`; λ=5 cho reward **−12,5**. Một lượt đổi pha dài 40 giây với 400 xe·giây cũng có mean queue 10. Nếu đã cấp xanh cho edge đó và các edge khác trống, `A=0` và reward là **−10**. Khi cả queue trung bình và tuổi đều bằng 0 thì reward bằng 0.

Phép đo chỉ dùng nhu cầu quan sát tại các điểm quyết định. Nó đo tuổi thiếu xanh của **edge**; một movement nhận xanh có thể đặt lại đồng hồ dù xe trên cùng đường còn chờ rẽ theo movement khác. Nó cũng có thể bỏ lỡ queue xuất hiện rồi tan giữa hai quyết định. Các khoảng thiếu xanh khi không có xe vẫn được ghi trong metric tín hiệu; chúng không tự tạo phạt. Không thay đổi các metric FT/MP/QL.

## Chọn trước λ và cách quyết định

Ba nhánh dùng λ **0, 5, 15**. Mốc 300 giây lấy từ state Overdue. Trên 1.395 quyết định train Overdue lịch sử (exploration seed 7), reward trung vị là −67,925. Hai mức phạt tối đa tương đương khoảng 7% và 22% độ lớn trung vị. Phát lại 1.380 điểm cuối không terminal bằng queue đầu bước kế tiếp và xanh thực: 117 lần tuổi vượt 300 giây, 20 lần đạt ít nhất 600 giây, cực đại 1.120 giây. Trace cũ thiếu queue theo làn cuối episode ở 15 bước terminal. [Log train dùng để định thang](../experiments/overdue-eval-ukeh31e3/training/cologne1/exploration_7/)

**Quy tắc validation đã chốt trước khi chạy:** một λ dương cần giảm tuổi chờ quan sát cực đại trên **từng seed** so với λ=0, giữ hoặc tăng completion trung bình, đồng thời queue và teleport trung bình không tăng. Nếu cả hai đạt, ưu tiên tuổi cực đại thấp hơn rồi λ nhỏ hơn khi hòa. Nếu chưa đạt, giữ λ=0. So sánh checkpoint cùng ngân sách và giữ toàn bộ lượt chạy. Tổng reward train không dùng để xếp hạng vì λ thay đổi chính thang reward.

## Validation ngắn với SUMO thật

Mỗi nhánh train từ bảng rỗng với cùng state Overdue, 3 episode × 900 giây, SUMO seed 23423, exploration seed 7, α=0,1, γ=0,95 và lịch ε tuyến tính 1 → 0,525 → 0,05. Mỗi checkpoint 3 chạy greedy, Guard tắt, trên hai seed validation độc lập với train. Bảng ghi số liệu từng lượt trong đợt đầu, mỗi lượt có 546 chuyến yêu cầu.

| Seed | λ | Hoàn thành hợp lệ | Queue TB (xe) | Chờ (s/xe) | Đi lại (s/xe) | Time loss (s/xe) | Throughput (xe/h) | Teleport | Tuổi chờ cực đại (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0 | 205/546 (37,55%) | 94,52 | 18,07 | 48,67 | 23,74 | 820 | 14 | 620 |
| 42 | 5 | 205/546 (37,55%) | 94,63 | 18,34 | 49,56 | 24,61 | 820 | 14 | 620 |
| 42 | 15 | 205/546 (37,55%) | 94,52 | 18,07 | 48,67 | 23,74 | 820 | 14 | 620 |
| 12345 | 0 | 204/546 (37,36%) | 94,96 | 16,61 | 46,88 | 22,04 | 816 | 14 | 620 |
| 12345 | 5 | 204/546 (37,36%) | 94,96 | 16,61 | 46,88 | 22,04 | 816 | 14 | 620 |
| 12345 | 15 | 204/546 (37,36%) | 94,96 | 16,61 | 46,88 | 22,04 | 816 | 14 | 620 |

Ở seed 42, mỗi lượt còn 191 xe đã vào mạng nhưng chưa về đích, 137 xe chưa vào mạng và 13 xe đã đến nhưng từng teleport. Các số tương ứng ở seed 12345 là 190, 138 và 14. Chờ, đi lại và time loss chỉ lấy trung bình trên xe đến hợp lệ. Throughput quy đổi 900 giây sang một giờ; teleport đếm sự kiện. Gap cực đại không có xanh của một movement là 900 giây trong cả sáu lượt, kể cả thời gian không có nhu cầu. Tuổi 620 giây trong bảng chỉ tính ở edge còn queue quan sát được. [Định nghĩa metric](../src/evaluation_metrics.py)

Mỗi lượt train có 8/72 quyết định có phạt chuẩn hóa dương. Replay xác nhận reward của đúng 8 bước thay đổi ở mỗi nhánh λ dương. So từng nhánh với baseline, trên 32 state chung của checkpoint cuối có 19 ô Q khác giá trị, nhưng pha greedy đầu tiên đạt cực đại giống nhau ở cả 32 state. Chênh lệch queue và thời gian nhỏ ở seed 42 đi cùng **cùng chuỗi action**; từ lượt này chưa thể quy chênh lệch ấy cho reward. Quyết định giữ baseline chỉ áp dụng cho mức kiểm chứng ngắn này. [Giới hạn tái lập của Cologne1](reproducibility_check.md)

## Kiểm chứng và tái tạo

Chạy lại từ bảng Q rỗng với cùng cấu hình cho thêm sáu lượt validation. Completion, throughput, số teleport và tuổi chờ cực đại lặp lại trên cả hai seed. Ví dụ ở λ=5, seed 12345, queue trung bình hai lần là 94,96 và 95,01 xe. Đây là hai lần lặp trên cùng demand, không phải hai bộ seed mới. Bảng CSV lưu từng lượt và SHA checkpoint. Bản nén trong `experiments/` chứa log và checkpoint của hai lần, đã kiểm tra đọc lại đủ 12 `result.json` validation. [CSV từng lượt](ql_starvation_validation.csv) · [Log và checkpoint](../experiments/ql-reward-smoke-bel5bde3.zip)

Toàn bộ suite đạt **218/218 test**. Sau khi bổ sung tuổi chờ cực đại vào `result.json`, đã chạy lại riêng **10/10 test** của reward và đối chiếu trường này với tuổi dựng từ trace. Replay bộ lịch sử Stage 6.2 khớp báo cáo đã lưu.

- Checkpoint mới dùng schema 2 và lưu `reward.version`, `lambda` cùng ngưỡng và độ dài đoạn tăng cho biến thể. `model_reward()` chỉ đọc checkpoint schema 1 lịch sử theo reward queue gốc; schema mới thiếu cấu hình hoặc có version mâu thuẫn sẽ báo lỗi. Evaluate dùng cấu hình của model. Có thể yêu cầu khớp chính xác nhánh bằng `--expect-starvation-lambda` trước khi khởi chạy SUMO. [Cấu hình](../src/ql_reward.py) · [Evaluate](../src/evaluate.py)
- `actions.csv` của QL lưu queue-seconds, elapsed, queue làn vào cuối bước, mean queue, tuổi cuối bước, `starvation_penalty` đã chuẩn hóa và reward cuối. `audit_rewards()` tính lại diện tích queue từ từng mẫu 1 giây và dựng lại tuổi từ queue cùng các ký tự `G`; replay còn đối chiếu từng bước Bellman và checkpoint. [Replay độc lập](../tools/analyze_ql.py) · [Test tính tay, metadata và SUMO](../tests/test_ql_reward.py)
- Chạy riêng phép so sánh ngắn bằng `.\.venv\Scripts\python.exe -B -m pytest -q tests/test_ql_reward.py`. Để chạy chính thức khi được phép, dùng lệnh dưới từ gốc repository. Mỗi nhánh train mới 15 episode, SUMO seed 23423, exploration seed 7, ε tuyến tính cùng mốc 15; checkpoint 15 được validation trên seed 42 và 12345, hai repeat/seed. Đọc số liệu từng lượt bên cạnh bảng gộp vì cùng seed SUMO vẫn có thể biến thiên trên Cologne1. [Bộ tổng hợp](../tools/ql_progress.py) · [Độ tái lập](reproducibility_check.md)

~~~powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$batch = Join-Path '.\experiments' ('ql-reward-' + [guid]::NewGuid().ToString('N'))
foreach ($weight in 0,5,15) {
    $folder = "$batch/lambda_$weight"
    & $python src/q_learning.py train cologne1 --overdue-state --episodes 15 --decay-episodes 15 --seed 23423 --exploration-seed 7 --starvation-lambda $weight --output "$folder/train"
    if ($LASTEXITCODE) { throw "Train failed: $folder" }
    $runs = @()
    foreach ($seed in 42,12345) {
        foreach ($repeat in 1,2) {
            $out = "$folder/validation/seed_$seed/repeat_$repeat"
            & $python src/evaluate.py --map cologne1 --algorithm QL --model "$folder/train/episode_015/q_table.json" --expect-starvation-lambda $weight --seed $seed --output $out
            if ($LASTEXITCODE) { throw "Validation failed: $out" }
            $runs += $out
        }
    }
    $csv = & $python tools/ql_progress.py "$folder/train" --validation @runs
    if ($LASTEXITCODE) { throw "Audit failed: $folder" }
    $csv | Set-Content -Encoding utf8 "$folder/progress.csv"
}
~~~
