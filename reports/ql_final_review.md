# Giai đoạn 6 — Kết luận hai thí nghiệm Tabular Q-Learning

**Giữ QL-Overdue làm biến thể cho Cologne1 và giữ QL-Original làm đối chứng.** Khi thêm hướng có hàng đợi chờ xanh lâu nhất vào state, tỷ lệ hoàn thành trên test Cologne1 tăng 25,19 điểm phần trăm, chuỗi giữ cùng pha dài nhất giảm từ 107 xuống 7 quyết định. **Giữ cách khám phá mặc định trên cả hai bản đồ.** Ưu tiên pha chưa từng cập nhật làm giảm tỷ lệ hoàn thành ở cả hai bản đồ và trượt tiêu chí validation.

## Thử gì và quyết định thế nào?

| Thí nghiệm | Một thay đổi so với Original | Căn cứ quyết định trên validation |
|---|---|---|
| 1. Ưu tiên pha chưa thử | Trong nhánh khám phá, lấy mẫu đều trong các pha có số lần cập nhật `N(s,a) = 0` nếu còn. Khi đã thử hết, lấy mẫu đều trong mọi pha. | Completion giảm từ 55,83% xuống 42,36% trên Cologne1 và từ 92,63% xuống 88,51% trên Ingolstadt1. Biến thể trượt điều kiện giữ ở cả hai bản đồ. |
| 2. Ghi hướng chờ xanh vào state | Dùng khám phá gốc. Từ khóa gồm bin queue và ID pha, nối thêm mã 1–4 cho edge vào có queue thiếu xanh ưu tiên lâu nhất từ 300 giây, hoặc 0 nếu chưa có edge quá hạn. | Trên Cologne1, khoảng thiếu xanh cực đại giảm ở cả seed 42 (3.440 → 3.280 s) và 12345 (3.600 → 3.130 s), đồng thời completion trung bình tăng từ 55,65% lên 79,98%. Đạt cả hai điều kiện. |

Trước và sau hai thay đổi, reward là `r = −queue_seconds / elapsed`, cập nhật là `Q(s,a) ← Q(s,a) + α[target − Q(s,a)]`. Bước thường dùng `target = r + γ max Q(s′,·)`, bước cuối dùng `target = r`. Mỗi model học 15 episode với α = 0,1, γ = 0,95 theo quyết định, ε giảm từ 1 xuống 0,05. Khi evaluate, agent dùng greedy và Guard tắt. Cả hai model ứng với seed khám phá 7 và 17 đều vào test ở checkpoint 15. [Thí nghiệm 1](ql_untried_first/README.md) · [Thí nghiệm 2](ql_overdue_state/README.md)

## So sánh trên test giữ riêng

Mũi tên chỉ **Original → biến thể trong cùng thí nghiệm**. Mỗi dòng QL gồm 12 lượt cho mỗi cấu hình, với ba seed SUMO, hai model và hai lần lặp. Mỗi baseline có sáu lượt. Tất cả lượt chạy đủ 3.600 giây với cùng demand và cấu hình an toàn trong từng bản đồ. Thí nghiệm 1 dùng seed 73101–73103, thí nghiệm 2 dùng 84201–84203.

| Thí nghiệm / bản đồ | Completion QL | Completion FT / MP | Queue TB (xe) | Teleport TB/lượt | Chuỗi cùng pha dài nhất (quyết định) | Khoảng thiếu xanh dài nhất (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1 / Cologne1 | 49,69% → 42,36% | 93,72% / 97,50% | 115,22 → 117,29 | 70,00 → 77,50 | 107 → 107 | 3.600 → 3.600 |
| 1 / Ingolstadt1 | 94,12% → 89,53% | 98,27% / 97,65% | 8,35 → 6,70 | 5,00 → 9,67 | 7 → 7 | 1.220 → 1.634 |
| 2 / Cologne1 | 56,89% → 82,08% | 93,73% / 97,58% | 108,65 → 66,20 | 65,50 → 47,58 | 107 → 7 | 3.600 → 3.130 |

Completion tính số chuyến đến đích hợp lệ trên toàn bộ nhu cầu. Các trung bình thời gian chỉ tính xe hoàn thành trong thời hạn, chưa bị loại và chưa từng teleport. Với QL-Overdue, throughput tăng từ 1.146,33 lên 1.654,00 xe/h, nhưng trên nhóm về đích, chờ tăng từ 91,81 lên 112,81 s/xe, thời gian đi lại từ 124,07 lên 155,25 s/xe và time loss từ 103,50 lên 132,17 s/xe. Mỗi policy đưa một tập xe khác nhau tới đích. [Thống kê đầy đủ thí nghiệm 1](ql_untried_first/summary.csv) · [Thống kê đầy đủ thí nghiệm 2](ql_overdue_state/summary.csv)

## Lặp pha, thiếu xanh và độ ổn định

- **Lặp pha có thể xảy ra với action đã học.** Trong một lượt thí nghiệm 1 trên Cologne1, cùng khóa state xuất hiện 106 lần trong chuỗi 107 quyết định giữ pha 7. Ở khóa này, pha 7 từng được cập nhật 7 lần và có Q lớn nhất duy nhất. Ưu tiên thử action mới lúc train vẫn cho phép lặp pha khi chạy greedy. [Trace và phân tích thí nghiệm 1](ql_untried_first/README.md)
- **Đổi pha thường xuyên vẫn có thể bỏ qua một hướng rẽ.** Trên test thí nghiệm 2, QL-Overdue cấp xanh ngay cho edge chờ lâu nhất ở 2/359 quyết định có edge quá hạn. State theo dõi cả đường vào, còn khoảng thiếu xanh đo riêng từng movement (một hướng di chuyển qua nút), kể cả khi không có xe chờ. Reward đo queue trong mạng, chưa tính trực tiếp xe chưa vào mạng hoặc thời gian thiếu xanh của từng movement. Hai thí nghiệm chưa đo tác động riêng của việc đổi reward. [Đối chiếu tín hiệu và nhu cầu](ql_overdue_state/README.md)
- **Kết quả ổn định hơn theo completion, nhưng policy vẫn thay đổi.** Trên 12 lượt test Cologne1 của thí nghiệm 2, completion Original nằm trong 37,32–71,91% (độ lệch chuẩn 13,77 điểm phần trăm), Overdue trong 80,25–83,87% (1,37 điểm). Tuy vậy, giữa các checkpoint 14 và 15, Overdue đổi pha greedy ở 29/172 state chung với seed khám phá 7 và 34/154 với seed 17. Tỷ lệ chọn cặp state–action chưa từng cập nhật khi test tăng từ 16,24% lên 42,12%. Chưa đủ bằng chứng hội tụ. [Coverage, checkpoint và repeat](ql_overdue_state/README.md)

Cologne1 có biến thiên ngay cả khi giữ nguyên model và seed SUMO. Hai lượt lặp của Original, seed khám phá 7 và SUMO seed 84201, lệch nhau 108 chuyến hợp lệ. Kết luận giữ QL-Overdue áp dụng cho demand Cologne1 đã thử. Chưa có phép so sánh QL-Overdue trên Ingolstadt1. [Kết quả từng lượt thí nghiệm 2](ql_overdue_state/runs.csv)

## Kiểm chứng và vị trí thuyết trình

Trong Giai đoạn 6, chạy `python -B -m pytest -q -p no:cacheprovider` đạt **184/184 test**, gồm cả tích hợp SUMO. Từ hai tệp kết quả từng lượt, tính lại số mẫu, trung bình, min, max và độ lệch chuẩn mẫu của cả 324 dòng thống kê. Tất cả khớp trên 48 lượt validation và 108 lượt test đã lưu. [Dữ liệu thí nghiệm 1](ql_untried_first/runs.csv) · [Dữ liệu thí nghiệm 2](ql_overdue_state/runs.csv)

| Câu hỏi khi trình bày | Hàm cần mở |
|---|---|
| Agent nhìn thấy gì? | `SumoEnv.get_state()` đọc số xe dừng trên các làn vào, làn ra và pha hiện tại. [sumo_env.py](../src/sumo_env.py#L182) |
| Tạo khóa bảng Q ra sao? | `encode_state()` cộng queue theo edge, chia bin rồi ghép với pha. Với Overdue, `EdgeDemand.observe()` xác định thêm mã hướng quá hạn. [Mã hóa](../src/q_learning.py#L89) · [Xác định hướng](../src/q_learning.py#L66) |
| Ai chọn pha xanh? | `choose_action()` thực hiện ε-greedy, nhánh greedy lấy pha đầu tiên nếu hòa Q. [q_learning.py](../src/q_learning.py#L129) |
| SUMO nhận lệnh ở đâu? | `SumoEnv.step()` tổ chức chuyển pha an toàn, `_run_signal()` gửi tín hiệu qua TraCI và lấy mẫu từng giây. [Điều khiển](../src/sumo_env.py#L238) · [Lệnh TraCI](../src/sumo_env.py#L214) |
| Reward từ đâu? | `SumoEnv.step()` lấy tổng xe-giây chờ trong bước, chia cho số giây thực trôi qua rồi đổi dấu. [Dòng tính reward](../src/sumo_env.py#L268) |
| Q thay đổi ở đâu? | `update_q_value()` tính target, xử lý terminal rồi cập nhật đúng ô của state và pha đã chọn. [q_learning.py](../src/q_learning.py#L147) |
| Lưu gì để dùng lại? | `save_q_table()` ghi bảng và metadata để lúc tải khớp đúng cấu hình. [q_learning.py](../src/q_learning.py#L211) |
| Đánh giá bằng cách nào? | `run_evaluation()` chạy từng controller, `greedy_steps()` tra bảng Q đã khóa. `trip_metrics()` tính chỉ số chuyến xe, `signal_metrics()` tính queue và khoảng thiếu xanh. [Runner](../src/evaluate.py#L60) · [Greedy](../src/evaluate.py#L32) · [Chuyến xe](../src/evaluation_metrics.py#L47) · [Tín hiệu](../src/evaluation_metrics.py#L95) |
