# Giai đoạn 3 — Q ban đầu và cách chọn pha khi hòa

**Quyết định hiện tại: giữ Q ban đầu bằng 0 và chọn pha đầu tiên khi hòa.** Hai biến thể dưới đây là tùy chọn để thử độc lập. Trong smoke Cologne1, Q âm giảm số lần chọn cặp chưa học nhưng queue và teleport đều tăng. Phá hòa ngẫu nhiên khi train vẫn cho một lượt evaluate giữ nguyên pha suốt 420 giây. Chưa có kết quả validation đủ một giờ để thay mặc định.

## Agent chọn pha nào khi bảng Q học chưa đủ?

| Tình huống | Hành vi của Q-Learning thuần |
|---|---|
| State chưa gặp khi evaluate | `get_q_values(create=False)` trả một hàng tạm, mọi cột bằng `initial_q`. Cả ba nhánh chọn pha đầu tiên trong thứ tự action đã lưu. Bảng trên đĩa và trong bộ nhớ giữ nguyên. |
| State đã gặp, một số pha chưa cập nhật | Mỗi pha chưa thử còn giá trị khởi tạo. Ví dụ thứ tự pha `[0,2,4]`, hàng `[-4,0,-2]` sẽ chọn pha 2. Nếu nhật ký cho biết pha 2 chưa thử, chính giá trị 0 cao hơn hai Q âm dẫn đến lựa chọn đó. Đổi cách phá hòa không tác động vì chỉ một pha đạt cực đại. |
| Nhiều pha cùng Q lớn nhất | Mặc định `choose_action()` lấy cột đầu tiên đạt max. Nhánh thử `--train-tie-seed 19` bốc đều trong các pha cùng max **chỉ khi train đi vào nhánh greedy**. RNG phá hòa và RNG khám phá độc lập, mỗi RNG giữ trạng thái qua các episode. Khi chỉ có một cực đại hoặc khi ε chọn khám phá, RNG phá hòa không được dùng. Evaluate luôn chọn cực đại đầu tiên. |
| Một pha đã học có Q lớn nhất duy nhất | Cả hai cách phá hòa đều chọn pha đó. Giai đoạn 1 ghi nhận một chuỗi giữ pha 4 với 106 quyết định cùng state, tại đó pha 4 đã nhận 11 cập nhật và có Q lớn nhất duy nhất. |

`update_q_value()` chỉ khởi tạo khi thiếu hàng và chỉ ghi cột của pha đã chọn. Khởi tạo âm áp dụng cả cho state kế tiếp vừa gặp trong bootstrap. Bước thường dùng `target = r + γ max Q(s′,·)`, bước cuối dùng riêng `r`. Một ô đã học vẫn có thể bằng giá trị ban đầu. Coverage và tỷ lệ chọn cặp chưa học phải đối chiếu số cập nhật `N(s,a)` từ train. [Chọn pha và Bellman](../src/q_learning.py) · [Evaluate và kiểm bảng cố định](../src/evaluate.py) · [Trace giữ pha](q_learning_review.md)

**Các phương án khác.** Chỉ cho greedy chọn pha đã thử sẽ loại các ước lượng ít bằng chứng, nhưng cũng có thể khóa policy vào pha đầu tiên từng được thử và vẫn cần quy tắc cho state hoàn toàn mới. Ép thử action mới từng làm giảm completion trong thử nghiệm trước. Fallback theo áp lực sẽ tạo bộ điều khiển lai QL+MP. Cả ba nhánh ở đây đều dùng cùng tập action của Q-Learning thuần. [Thử nghiệm untried-first](ql_untried_first/README.md)

## Vì sao thử Q ban đầu −1.358,5 trên Cologne1?

Đọc cột `reward` của 15 tệp `episode_*/actions.csv` thuộc **train** QL-Overdue lịch sử, exploration seed 7, SUMO seed 23423. Trên **1.395 quyết định**, reward nằm trong [−171,525; 0], trung vị **−67,925**, trung bình −76,4915. Với γ = 0,95 theo quyết định, nếu reward trung vị lặp lại thì tổng chiết khấu là `−67,925 / (1 − 0,95) = −1.358,5`. Đây là giả định để chọn **một** mức thử theo thang giá trị Q. Episode thực hữu hạn, reward thay đổi theo state và pha. Bản đồ hoặc γ khác cần tính lại mức thử. [Log train dùng để tính trung vị](../experiments/overdue-eval-ukeh31e3/training/cologne1/exploration_7/)

Có thể thấy tác động ngay ở bước thường đầu tiên nếu cả hai hàng Q đều mới. Với mức ban đầu `c`, thay đổi của ô đã chọn là `α × [r − (1−γ)c]`. Khi `c=0`, mọi reward âm đều đẩy pha đã thử xuống dưới pha chưa thử. Khi `c=−1.358,5`, một reward tốt hơn −67,925 đẩy pha đã thử lên trên giá trị ban đầu; reward thấp hơn vẫn đẩy nó xuống. Giá trị âm sâu vì thế cũng có thể làm policy bám sớm vào một pha và giảm coverage.

**Căn cứ thử phá hòa riêng.** Trong cùng lượt train lịch sử có **245/662** quyết định greedy hòa cực đại. Bốn lượt validation lưu sẵn của đúng model có **220/371** quyết định hòa; 89/371 ở state mới. Vì phép thử chỉ đổi phá hòa lúc train, mọi khác biệt khi evaluate xuất phát từ bảng học được hoặc biến thiên SUMO. [Nhánh khám phá và số lần hòa khi train](../experiments/overdue-eval-ukeh31e3/training.csv) · [Các lượt validation](ql_overdue_state/runs.csv)

## Kết quả smoke trên SUMO thật

Ba nhánh cùng train 3 episode × 420 giây trên Cologne1, state Overdue, 8 action, α=0,1, γ=0,95, exploration seed 7, SUMO seed 23423, lịch ε tuyến tính 1 → 0,525 → 0,05. Mỗi checkpoint 3 được evaluate một lần trên seed 42, cùng 420 giây và 277 chuyến yêu cầu. Guard tắt. Replay đối chiếu Q trước/sau, RNG, nhánh terminal và từng checkpoint. SHA model khớp trước/sau evaluate.

| Nhánh | Completion hợp lệ | Queue TB (xe) | Teleport | Coverage cuối train | Chọn cặp chưa cập nhật khi evaluate | Chuỗi cùng pha dài nhất (quyết định / giây xanh) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, Q=0 | 73/277 (26,35%) | 60,03 | 2 | 33/144 | 11/11 | 2 / 60 |
| Chỉ đổi Q ban đầu | 72/277 (25,99%) | 65,52 | 3 | 19/112 | 0/11 | 2 / 60 |
| Chỉ đổi phá hòa khi train | 9/277 (3,25%) | 75,12 | 6 | 32/184 | 14/14 | 14 / 420 |

Mẫu số coverage là số state **thực gặp ở từng lượt train** nhân 8. Ở nhánh Q âm, chọn cặp đã học trong cả 11 quyết định vẫn đi kèm ít chuyến hoàn thành hơn baseline. Một seed và khung giờ rút gọn chỉ đủ kiểm chứng cơ chế; cùng seed SUMO trên Cologne1 cũng có thể cho các quỹ đạo khác nhau. [Smoke và các phép tính tay](../tests/test_q_learning.py) · [Độ tái lập](reproducibility_check.md)

## Validation chính thức đã thiết kế

So từng biến thể với baseline ở **checkpoint 15**. Cả ba cùng dùng lịch tuyến tính giảm trong 15 episode, SUMO train 23423, khám phá `uniform` và cùng seed khám phá 7, 17. Mỗi model dùng validation SUMO seed 42 và 12345, mỗi seed hai repeat, mỗi lượt 3.600 giây. Như vậy mỗi nhánh có tám lượt validation. Xem kết quả từng lượt và từng seed khám phá trước khi gộp; test giữ riêng.

**Quy tắc đề xuất trước validation:** chỉ cân nhắc giữ biến thể nếu completion trung bình tăng ở cả hai seed khám phá, đồng thời queue và số teleport trung bình trên tám lượt không tăng, chuỗi cùng pha cực đại tính bằng giây xanh không dài hơn baseline. Đọc thêm gap thiếu xanh, coverage, tỷ lệ state lạ, tỷ lệ chọn cặp chưa cập nhật và tỷ lệ hòa để giải thích. Nếu các repeat dao động đến mức chưa phân biệt được hai nhánh, tiếp tục dùng baseline. [Cách đối chiếu từng checkpoint](../tools/ql_progress.py)

Lệnh chạy từ gốc repository khi được phép chạy thí nghiệm dài. Mỗi đợt tạo thư mục riêng trong `experiments/`. Tệp `progress.csv` có một dòng cho mỗi episode và gắn validation với đúng SHA checkpoint.

~~~powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$batch = Join-Path '.\experiments' ("ql-actions-" + [guid]::NewGuid().ToString("N"))
$arms = @{
    'baseline' = @()
    'negative' = @('--initial-q','-1358.5')
    'train-ties' = @('--train-tie-seed','19')
}
foreach ($exploration in 7,17) {
    foreach ($arm in 'baseline','negative','train-ties') {
        $folder = "$batch/$arm/e$exploration"
        $options = $arms[$arm]
        & $python src/q_learning.py train cologne1 --output "$folder/train" --episodes 15 --decay-episodes 15 --overdue-state --seed 23423 --exploration-seed $exploration @options
        if ($LASTEXITCODE) { throw "Train failed: $folder" }
        $model = "$folder/train/episode_015/q_table.json"
        $runs = @()
        foreach ($seed in 42,12345) {
            foreach ($repeat in 1,2) {
                $out = "$folder/validation/seed_$seed/repeat_$repeat"
                & $python src/evaluate.py --algorithm QL --map cologne1 --seed $seed --model $model --output $out
                if ($LASTEXITCODE) { throw "Validation failed: $out" }
                $runs += $out
            }
        }
        $csv = & $python tools/ql_progress.py "$folder/train" --validation @runs
        if ($LASTEXITCODE) { throw "Audit failed: $folder" }
        $csv | Set-Content -Encoding utf8 "$folder/progress.csv"
    }
}
~~~

Đã chạy `.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider`: **196/196 test đạt**. Để chạy riêng ba nhánh smoke, thêm `tests/test_q_learning.py -k action_selection_smoke`. Khi thuyết trình, đi từ `get_q_values()` đến `choose_action()`, `update_q_value()`, rồi `greedy_steps()`. Mở `replay_training()` để xem cách đếm cập nhật thực. Trong `ql_progress.py`, `progress()` tính coverage từ train; `validation_metrics()` đối chiếu lựa chọn greedy với các cặp đã cập nhật và lấy chuỗi xanh dài nhất từ `actions.csv`. [Q-Learning](../src/q_learning.py) · [Runner](../src/evaluate.py) · [Replay](../tools/analyze_ql.py) · [Thống kê](../tools/ql_progress.py)
