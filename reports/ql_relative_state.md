# Giai đoạn 4 — Phân biệt hai hướng cùng có hàng đợi dài trên Cologne1

**Giữ QL-Overdue làm đối chứng.** Biến thể QL-Relative phân biệt được hướng nào dẫn trước rõ rệt khi nhiều hàng đợi cùng vượt bin cuối. Trên hai trace train lịch sử, số dòng tăng 45–47% và coverage giảm. Trong smoke, Relative hoàn thành nhiều chuyến hơn ở cả hai seed validation, nhưng queue cũng cao hơn ở cả hai. Chưa đủ bằng chứng để chọn Relative.

## State hiện tại mất thông tin ở đâu?

`SumoEnv.get_state()` đọc `getLastStepHaltingNumber` ở bước SUMO vừa kết thúc trên 8 làn vào, 8 làn ra và lấy ID pha hiện tại. `build_encoding()` ánh xạ mỗi làn vào đúng một edge theo XML. `encode_state()` cộng theo 4 edge vào, 4 edge ra, rồi chia từng tổng thành bin 0, 1–4, ≥5. Trong một bin, 6 và 60 xe có cùng mã. Việc cộng làn còn làm mất phân bố xe giữa các làn trên cùng edge. [Môi trường](../src/sumo_env.py) · [Mã hóa](../src/q_learning.py)

Overdue nối mã 0 hoặc ID 1–4 của edge có queue chờ xanh lâu nhất từ 300 giây. `EdgeDemand.observe()` bắt đầu đếm ở quyết định quan sát thấy queue dương, đặt lại khi thấy queue bằng 0. `record_green()` đặt lại khi một movement của edge **thực nhận** xanh ưu tiên trong bước đã qua. Mỗi episode tạo bộ đếm mới. Đặc trưng này từng nâng completion test từ 56,89% lên 82,08% trên bộ seed của thí nghiệm Overdue. Busiest giữ mã quá hạn nhưng thay bin queue vào bằng cờ có xe và hướng đông nhất. Trên bộ test riêng của Busiest, bảng nhỏ hơn nhưng completion giảm từ 78,85% ở Overdue xuống 66,43%. [Thí nghiệm Overdue](ql_overdue_state/README.md) · [Thí nghiệm Busiest](ql_busiest_state/README.md)

**Chọn chênh lệch queue để thử trước.** Trên Cologne1, mỗi làn vào nối với 2 hoặc 3 làn ra. Một số đếm xe dừng trên làn chưa cho biết bao nhiêu xe đang chờ từng hướng rẽ. Gán toàn bộ queue cho mỗi movement sẽ đếm trùng. Thời gian thiếu phục vụ đã có trong Overdue. Relative dùng lại đúng một ảnh chụp queue hiện tại và giữ nguyên đồng hồ này.

## Một thay đổi trong khóa Q

Với bin mặc định, nếu hai hướng đứng đầu **đều có ít nhất 5 xe** và hướng dẫn đầu hơn hướng thứ hai **ít nhất 5 xe**, đổi riêng bin của hướng dẫn đầu từ 2 thành 3. Các trường hợp khác giữ nguyên bin Overdue, kể cả khi hòa. Mốc 5 là số xe đầu tiên thuộc bin cuối; đây là lựa chọn thiết kế để các chênh lệch 1–4 xe không tách khóa. Với bộ bin khác, ngưỡng tương ứng bằng cận cuối cộng 1 và được lưu trong metadata.

Ví dụ thực lấy từ trace Overdue seed khám phá 7. Ba quan sát dưới đây có cùng bin đầu ra, pha 0 và mã quá hạn 0. Thứ tự edge theo `incoming_groups` trong model.

| Queue vào (xe) | Overdue | Relative |
|---|---|---|
| (20, 26, 14, 0) | (2, 2, 2, 0) | (2, 3, 2, 0) |
| (83, 32, 11, 0) | (2, 2, 2, 0) | (3, 2, 2, 0) |
| (9, 12, 13, 0) | (2, 2, 2, 0) | (2, 2, 2, 0) |

Nếu chỉ một hướng đông, các bin cũ đã chỉ ra hướng đó. Nếu hai hướng dẫn đầu cùng có 9 xe, cả hai giữ mã 2. Relative vẫn gộp nhiều mức độ ùn tắc có cùng hướng dẫn đầu. Cờ train `--relative-state` tạo bảng rỗng và ghi `state_version=overdue-relative-v1`, `lead_gap=5`, thứ tự làn, bin, pha và ngưỡng quá hạn. `load_q_table()` kiểm schema cùng miền giá trị rồi đọc nguyên khóa đã lưu. Tám action và thứ tự cột Q dùng chung cho hai nhánh. [Triển khai và kiểm tra model](../src/q_learning.py)

## Kích thước và mức gộp trên cùng dữ liệu

Các cận trên dưới đây dùng bin mặc định, 4 edge vào, 4 edge ra, 8 pha, 5 mã quá hạn nếu có. Tín hiệu và nhu cầu thực tế chỉ đi qua một phần các tổ hợp. Q-table cấp phát 8 ô cho mỗi dòng đã tạo.

| Biểu diễn | Số tổ hợp phần vào | Cận trên số state |
|---|---:|---:|
| Original | 3⁴ = 81 | 52.488 |
| Overdue | 81 | 262.440 |
| Busiest | 1 + 4 × 2³ = 33 | 106.920 |
| Relative | 81 + 4 × (3³ − 2³) = 157 | 508.680 |

Ở Busiest, hoặc mọi hướng đều trống, hoặc chọn một hướng đông nhất rồi ghi cờ hiện diện của ba hướng còn lại. Ở Relative, mỗi khả năng mới cần một hướng đánh dấu và ít nhất một hướng khác còn ở bin 2. Nhân số tổ hợp phần vào với 81 tổ hợp đầu ra, 8 pha và số mã quá hạn tương ứng cho cận trên.

Đã phát lại hai trace **train Overdue**, 15 episode mỗi seed, lần lượt 1.395 và 1.401 quyết định. Giữ nguyên quan sát, pha và action đã thực hiện, chỉ mã hóa lại. Cột coverage bằng số cặp phân biệt chia cho số dòng × 8. Cột cuối đếm số quan sát phân biệt theo **tổng queue từng edge, pha và mã quá hạn nếu có** bị gộp nhiều nhất trong một khóa. [Trace từng episode](../experiments/overdue-eval-ukeh31e3/training/cologne1/) · [Đối chiếu độc lập trong observation_analysis()](../tools/analyze_ql.py)

| State | Số dòng e7 / e17 | Cặp đã chọn e7 / e17 | Coverage e7 / e17 | Số quan sát gộp cực đại e7 / e17 |
|---|---:|---:|---:|---:|
| Original | 124 / 111 | 484 / 445 | 48,79% / 50,11% | 129 / 139 |
| Overdue | 176 / 157 | 587 / 536 | 41,69% / 42,68% | 110 / 121 |
| Busiest | 122 / 112 | 425 / 390 | 43,55% / 43,53% | 106 / 101 |
| Relative | 259 / 228 | 745 / 697 | 35,96% / 38,21% | 74 / 85 |

Relative tách 57 khóa Overdue ở seed 7 và 47 ở seed 17. Các số liệu mã hóa lại đo mức phân biệt trên cùng trace. Chỉ hàng Overdue là bảng đã học trên chính quỹ đạo đó. Ví dụ khi Busiest thực sự được train lại trong thí nghiệm cũ, nó đi qua 126/108 state; bảng dưới đây đo riêng hai policy mới được train cho smoke.

## So sánh hai model train lại

Mỗi model train từ đầu 3 episode × 600 giây trên Cologne1, SUMO seed 23423, exploration seed 7, Q ban đầu 0, khám phá đều, phá hòa lấy pha đầu tiên. Cùng ε tuyến tính 1 → 0,525 → 0,05, α=0,1, γ=0,95, reward và Bellman hiện hành. Checkpoint 3 dùng greedy, Guard tắt. Mỗi seed validation có một lượt 600 giây và 416 chuyến yêu cầu.

| Model | Dòng / ô Q | Cặp đã cập nhật / coverage |
|---|---:|---:|
| Overdue | 26 / 208 | 44 / 21,15% |
| Relative | 30 / 240 | 47 / 19,58% |

| Seed | Model | Hoàn thành hợp lệ | Queue TB (xe) | Teleport | Gap movement cực đại (s) | Tuổi cực đại của edge có queue (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Overdue | 135/416 | 64,65 | 5 | 600 | 540 |
| 42 | Relative | 150/416 | 72,07 | 5 | 600 | 530 |
| 12345 | Overdue | 132/416 | 66,33 | 5 | 600 | 540 |
| 12345 | Relative | 135/416 | 81,19 | 6 | 600 | 540 |

Gap đếm mọi giây một movement thiếu `G`, kể cả lúc trống. Tuổi edge được dựng lại từ queue quan sát tại quyết định và `G` thực ở các bước trước. Trong cả bốn lượt, edge chờ lâu nhất nhận xanh ngay **0/8** lần có edge quá hạn. Khung 600 giây cắt các khoảng thiếu phục vụ đang tiếp diễn; chưa xác định được chúng kéo dài bao lâu. [Định nghĩa metric](../src/evaluation_metrics.py) · [Replay tuổi và tín hiệu](../tools/analyze_ql.py)

## Kiểm chứng và thiết kế validation tiếp theo

- Đã nạp checkpoint Cologne1 lịch sử Original 3 bin (57 dòng), 4 bin (213), Overdue (176) và Busiest (126) theo metadata gốc. Replay Stage 6.2 khớp kết quả đã lưu. Test kiểm tra không xe, một/nhiều hướng đông, biên 4/5, chênh 4/5, hòa, thứ tự làn/pha, tuổi quá hạn và reset. Smoke đối chiếu khóa từ queue thô, Bellman, số cặp, checkpoint và SHA qua evaluate. [Kiểm thử](../tests/test_overdue_state.py)
- Lệnh `.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider` đạt **208/208 test**. Để chạy riêng hai nhánh smoke, thêm `tests/test_overdue_state.py -k relative_state_comparison_smoke`. Xem `encode_state()` cho quy tắc chọn bin, `EdgeDemand.observe()/record_green()` cho tuổi, rồi `validation_metrics()` cho queue, teleport và phục vụ. [Mã Q-Learning](../src/q_learning.py) · [Tổng hợp](../tools/ql_progress.py)
- Cho validation đủ giờ, cố định checkpoint 15, exploration seed 7 và 17, SUMO train 23423, ε tuyến tính trong 15 episode. Mỗi model dùng seed validation 42 và 12345, hai repeat/seed. Chỉ khác cờ state giữa hai nhánh. Chỉ cân nhắc chọn Relative nếu completion trung bình tăng ở cả hai seed khám phá, queue và teleport trung bình không tăng, gap thiếu xanh cực đại không tăng, và số dòng thực lưu không vượt gấp đôi Overdue ở từng seed khám phá. Đọc thêm tuổi edge, số lần phục vụ khi quá hạn và tỷ lệ chọn cặp chưa học. Giữ mọi repeat vì Cologne1 có biến thiên ngay cả cùng seed. [Quy trình và lệnh train/validation](ql_training_budget.md)

Lệnh cho một cặp model cùng seed khám phá khi được phép train dài. Đổi `$exploration` thành 17 trong thư mục kết quả mới để chạy nửa còn lại của đối chứng.

~~~powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$batch = Join-Path '.\experiments' ('ql-relative-' + [guid]::NewGuid().ToString('N'))
$exploration = 7
foreach ($flag in '--overdue-state','--relative-state') {
    $folder = "$batch/$($flag.TrimStart('-'))"
    & $python src/q_learning.py train cologne1 $flag --episodes 15 --decay-episodes 15 --seed 23423 --exploration-seed $exploration --output "$folder/train"
    if ($LASTEXITCODE) { throw "Train failed: $folder" }
    $runs = @()
    foreach ($seed in 42,12345) {
        foreach ($repeat in 1,2) {
            $out = "$folder/validation/seed_$seed/repeat_$repeat"
            & $python src/evaluate.py --algorithm QL --map cologne1 --seed $seed --model "$folder/train/episode_015/q_table.json" --output $out
            if ($LASTEXITCODE) { throw "Validation failed: $out" }
            $runs += $out
        }
    }
    $csv = & $python tools/ql_progress.py "$folder/train" --validation @runs
    if ($LASTEXITCODE) { throw "Audit failed: $folder" }
    $csv | Set-Content -Encoding utf8 "$folder/progress.csv"
}
~~~

Thí nghiệm dừng ở Cologne1. Overdue hiện được thiết kế cho bản đồ này; cần bằng chứng rằng nhiều hướng thường xuyên cùng bão hòa và có nhu cầu quá hạn trên Ingolstadt1 trước khi mở rộng.
