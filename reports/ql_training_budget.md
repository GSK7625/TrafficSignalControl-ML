# Giai đoạn 2 — Ngân sách huấn luyện Q-Learning trên Cologne1

**Câu hỏi còn mở là liệu 30 hoặc 60 episode có giúp QL-Overdue phục vụ tốt hơn 15 episode hay không.** Đã chuẩn bị ba phép so sánh trên cùng state Overdue, cùng tám pha xanh, reward, Bellman, α = 0,1, γ = 0,95 theo quyết định và khám phá `uniform`. Chỉ smoke test dùng mô phỏng mới trong giai đoạn này. [Chẩn đoán giai đoạn 1](q_learning_review.md)

## Cách đọc code khi thuyết trình

| Bước | Điều xảy ra |
|---|---|
| 1. Bắt đầu episode | `train()` giữ một bảng Q và một `random.Random(exploration_seed)` xuyên suốt lượt train. Mỗi episode tạo `SumoEnv` với seed đã định. `run_episode()` gọi `reset()`, khởi động tiến trình SUMO mới từ thời điểm bắt đầu của bản đồ và đóng nó sau episode. Tuổi nhu cầu của Overdue cũng bắt đầu lại. [Train](../src/q_learning.py) · [Reset](../src/sumo_env.py) |
| 2. Chọn ε và seed | `episode_epsilon()` nhận chỉ số episode và độ dài giảm ε cố định. `episode_sumo_seed()` lấy phần tử `(episode − 1) % số_seed`; danh sách một phần tử cho seed cố định. Hai hàm dùng được khi replay offline. [Lịch huấn luyện](../src/training_schedule.py) |
| 3. Học và lưu | `choose_action()` dùng RNG khám phá, `SumoEnv.step()` thực thi pha. `update_q_value()` cập nhật đúng ô đã chọn, bước terminal dùng riêng reward. Sau từng episode, `train()` lưu `episode_NNN/q_table.json` và ghi ε, seed thực, tổng reward, queue-seconds, kích thước bảng vào `training.csv`. JSON lưu cả lịch ε, thứ tự seed train, seed khám phá, phiên bản SUMO, encoding và hash dữ liệu. [Vòng lặp train](../src/q_learning.py) |
| 4. Đánh giá checkpoint | Trước đây `train()` cũng lưu checkpoint từng episode và một bản cuối; thí nghiệm Overdue lịch sử khóa episode 15. Với thí nghiệm mới, `run_evaluation()` đọc từng checkpoint ứng viên, chạy greedy, Guard tắt và kiểm bảng Q cùng SHA trước/sau. So sánh các ứng viên trên validation rồi ghi đường dẫn và SHA của model được chọn. [Đánh giá](../src/evaluate.py) · [Thiết kế lịch sử](ql_overdue_state/README.md) |

## Thiết kế so sánh

| Nhánh | ε | Seed SUMO lúc train | So sánh tại cùng checkpoint |
|---|---|---|---|
| `linear-fixed` | Tuyến tính | 23423 | Đối chứng mới, gồm checkpoint 15 với lịch của model lịch sử |
| `exp-fixed` | Hàm mũ | 23423 | Với `linear-fixed` để đo ảnh hưởng của cách giảm ε; checkpoint 15 là đối chứng cùng ngân sách cho lịch mới |
| `linear-cycle` | Tuyến tính | 23423, 23424, 23425 lặp theo thứ tự | Với `linear-fixed` để đo ảnh hưởng của lịch seed |

**Ngân sách.** Mỗi nhánh train một lần đến 60 và lấy các checkpoint 15, 30, 60 của chính lượt đó. Vì thế các ứng viên trong một nhánh chia sẻ đúng 15 episode đầu, kể cả khi SUMO có biến thiên giữa hai lần chạy. Nếu chỉ cần một ngân sách, dùng cùng lệnh với `--episodes 15` hoặc `30`. Để mở rộng đến 100, xem điều kiện bên dưới. [Độ tái lập trên Cologne1](reproducibility_check.md)

**Lịch ε.** Lịch cũ đạt sàn tại cuối ngân sách dự kiến. Lịch mới mặc định cố định mốc này tại episode 15. Đặt `--decay-episodes N` để tái hiện tuyến tính của một ngân sách lịch sử N. Với `t = min((episode−1)/14, 1)`, hai công thức ở đây là `1 − 0,95t` và `0,05^t`. Các giá trị sau làm tròn bốn chữ số; `training.csv` ghi giá trị thực ở **mọi** episode.

| Episode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16–100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tuyến tính | 1 | ,9321 | ,8643 | ,7964 | ,7286 | ,6607 | ,5929 | ,5250 | ,4571 | ,3893 | ,3214 | ,2536 | ,1857 | ,1179 | ,0500 | ,0500 |
| Hàm mũ | 1 | ,8074 | ,6518 | ,5263 | ,4249 | ,3430 | ,2770 | ,2236 | ,1805 | ,1458 | ,1177 | ,0950 | ,0767 | ,0619 | ,0500 | ,0500 |

**Vai trò seed.** Cố định exploration seed **7** cho cả ba nhánh. Seed SUMO train lấy theo bảng trên. Validation dùng **42, 12345**, mỗi seed hai repeat cho từng model. Dành riêng **781101–781103** cho test sau khi khóa lựa chọn. Các lượt vẫn dùng cùng demand Cologne1. Thay seed có thể đổi chuyển động xe; cùng seed vẫn có thể cho hai quỹ đạo khác nhau. Kết quả của thiết kế này chỉ đại diện cho một seed khám phá.

## Lệnh train và validation

Chạy từ gốc repository trong PowerShell khi bắt đầu thí nghiệm dài. Mỗi lần khởi tạo tạo một thư mục kết quả mới bên cạnh bản clone. Các lệnh train/evaluate từ chối ghi đè. Trong ví dụ, mỗi lượt chính thức dùng đủ 3.600 giây.

~~~powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$runRoot = Join-Path '.\experiments' ("ql-budget-" + [guid]::NewGuid().ToString("N"))
$arms = @{
    'linear-fixed' = @('--epsilon-schedule','linear','--seed',23423)
    'exp-fixed'    = @('--epsilon-schedule','exponential','--seed',23423)
    'linear-cycle' = @('--epsilon-schedule','linear','--sumo-seeds',23423,23424,23425)
}
function Measure-Budget($arm, [int]$budget, [int[]]$checkpoints) {
    $folder = Join-Path $runRoot "$arm-$budget"
    $options = $arms[$arm]
    & $python src/q_learning.py train cologne1 --episodes $budget --output "$folder/train" --overdue-state --exploration-seed 7 --decay-episodes 15 @options | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Train failed: $folder" }

    $validation = @()
    foreach ($episode in $checkpoints) {
        $checkpoint = 'episode_{0:D3}' -f $episode
        $model = "$folder/train/$checkpoint/q_table.json"
        foreach ($seed in 42,12345) {
            foreach ($repeat in 1,2) {
                $out = "$folder/validation/$checkpoint/seed_$seed/repeat_$repeat"
                & $python src/evaluate.py --algorithm QL --map cologne1 --seed $seed --model $model --output $out | Out-Host
                if ($LASTEXITCODE -ne 0) { throw "Validation failed: $out" }
                $validation += $out
            }
        }
    }
    $csv = & $python tools/ql_progress.py "$folder/train" --validation @validation
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint audit failed: $folder" }
    $csv | Set-Content -Encoding utf8 "$folder/progress.csv"
    foreach ($row in ($csv | ConvertFrom-Csv)) {
        if ($row.validation_runs -eq '4') {
            [pscustomobject]@{
                arm = $arm; budget = $budget; episode = [int]$row.episode
                completion = [double]$row.completion; queue = [double]$row.queue
                teleports = [double]$row.teleports; gap = [int]$row.max_gap
                model = "$folder/train/episode_$('{0:D3}' -f [int]$row.episode)/q_table.json"
            }
        }
    }
}
$candidates = @(foreach ($arm in 'linear-fixed','exp-fixed','linear-cycle') {
    Measure-Budget $arm 60 @(15,30,60)
})
if ($candidates.Count -ne 9) { throw 'Missing checkpoint validation' }
$candidates | Format-Table arm, episode, completion, queue, teleports, gap
~~~

**Quy tắc dự kiến cho 100 episode.** Chỉ mở rộng nhánh mà completion validation trung bình tăng cả từ 15 đến 30 và từ 30 đến 60, đồng thời ở lần tăng cuối số teleport trung bình và gap cực đại không tăng. Lệnh dưới dành riêng cho bước mở rộng có điều kiện. Vì train bắt đầu từ bảng rỗng, đánh giá lại checkpoint 60 trong lượt 100 để có đối chứng trên cùng quỹ đạo. Giữ tất cả repeat, kể cả khi hai checkpoint 60 cho kết quả khác nhau.

~~~powershell
foreach ($arm in 'linear-fixed','exp-fixed','linear-cycle') {
    $at15, $at30, $at60 = @($candidates | Where-Object { $_.arm -eq $arm -and $_.budget -eq 60 } | Sort-Object episode)
    if ($at30.completion -gt $at15.completion -and $at60.completion -gt $at30.completion -and
        $at60.teleports -le $at30.teleports -and $at60.gap -le $at30.gap) {
        $extension = @(Measure-Budget $arm 100 @(60,100))
        if ($extension.Count -ne 2) { throw 'Missing extension validation' }
        $extension | Format-Table arm, episode, completion, queue, teleports, gap
        $candidates += $extension | Where-Object episode -eq 100
    }
}
~~~

**Khóa checkpoint.** Mỗi ứng viên có cùng bốn lượt validation. Xếp theo completion trung bình cao, rồi teleport trung bình thấp, gap cực đại thấp, queue trung bình thấp, ngân sách học nhỏ và tên nhánh để xử lý hòa. Checkpoint 60 chạy lại trong lượt mở rộng dùng riêng để đối chiếu 60→100. Kiểm tra độ chênh từng repeat trong `result.json` trước khi diễn giải chênh lệch nhỏ. Lưu bảng lựa chọn trước khi mở test.

~~~powershell
$ranking = @($candidates | Sort-Object @{Expression='completion';Descending=$true},teleports,gap,queue,episode,arm)
$ranking | Export-Csv "$runRoot/validation-ranking.csv" -NoTypeInformation -Encoding utf8
$winner = $ranking[0]
$winner | Select-Object *, @{Name='sha256';Expression={(Get-FileHash -Algorithm SHA256 $_.model).Hash}} |
    ConvertTo-Json | Set-Content -Encoding utf8 "$runRoot/selection.json"
~~~

## Đọc kết quả học

- Trong `progress.csv`, `total_reward` cộng reward theo quyết định. `average_queue_length` bằng queue-seconds chia cho giây mô phỏng, gồm vàng và đỏ. Hai cột mô tả quá trình train. Kết quả xếp hạng dùng các lượt greedy độc lập.
- `q_states` là số dòng checkpoint. `coverage = updated_pairs / (state đã gặp × số pha)`. `train_unseen_action_rate` là tỷ lệ quyết định chọn cặp chưa từng cập nhật **trước** quyết định đó. Khi validation, `validation_unseen_action_rate` đối chiếu mỗi lựa chọn với số lần cập nhật **đến đúng checkpoint**. `validation_unseen_state_rate` chỉ đếm khóa vắng trong model. Hai tỷ lệ validation gộp theo số quyết định; các trung bình giao thông cho mỗi lượt trọng số bằng nhau.
- `greedy_changes / common_states` so pha greedy trên các state chung của hai checkpoint liên tiếp. Ô tỷ lệ đầu tiên để trống. Đọc cùng coverage, số state mới, chỉ số giao thông và vùng state đến được khi validation. Một tỷ lệ đổi greedy nhỏ trong một episode chưa đủ chứng minh hội tụ.
- Với Overdue, `overdue_serving_phase_tried / overdue_decisions` cho biết khi có edge quá hạn, bao nhiêu lần ít nhất một pha phục vụ từng được học tại state đó. `overdue_served` đếm số lần edge nhận xanh trong chính quyết định ấy. Mẫu số bằng 0 nghĩa là lượt validation chưa gặp tình huống đó. [Công cụ tạo bảng](../tools/ql_progress.py) · [Đếm coverage và so checkpoint](../tools/stage7_summary.py)

## Kết quả đã kiểm tra

**Mốc lịch sử.** Đọc lại model Overdue 15 episode, exploration seed 7, SUMO train 23423. Checkpoint có **587/1.408 cặp**, đổi greedy ở **29/172** state chung. Trên bốn lượt validation lịch sử seed 42/12345, completion trung bình **83,98%**, queue **46,23 xe**, **41,75** teleport/lượt. Có 221 quyết định quá hạn, 97 lần đã từng cập nhật một pha phục vụ và 0 lần cấp xanh ngay. Đây là mốc của riêng seed khám phá 7. [Các lượt lịch sử](ql_overdue_state/runs.csv) · [Checkpoint và SHA đã lưu](../experiments/overdue-eval-ukeh31e3/models.json)

**Smoke với SUMO thật.** Mỗi lịch train 3 episode × 420 giây Cologne1, cùng Overdue, seed khám phá 7, seed SUMO 23423 → 23424 → 23423. Replay khớp lựa chọn, cập nhật Bellman và cả ba checkpoint mỗi lịch. Validation riêng dùng seed 42, 420 giây. SHA hai model đều giữ nguyên qua evaluate.

| Lịch | Số lần vào nhánh khám phá / số quyết định ở episode 1, 2, 3 | Coverage sau episode 3 | Greedy đổi 2→3 | Cặp chưa học khi validation |
|---|---|---:|---:|---:|
| Tuyến tính | 11/11, 11/11, 11/11 | 30/152 | 0/15 | 14/14 |
| Hàm mũ | 11/11, 11/11, 8/11 | 30/176 | 2/15 | 14/14 |

Hai lượt validation ngắn đều ghi 0 collision và 6 sự kiện teleport. Lệnh `.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` đạt **190/190 test**. Smoke kiểm tra đường chạy và khả năng đọc checkpoint. Sáu episode ngắn và một seed validation mỗi lịch chưa đủ để chọn lịch tốt hơn hoặc kết luận 15 episode đã đủ. Các ngân sách dài và test giữ riêng vẫn chờ thực nghiệm.
