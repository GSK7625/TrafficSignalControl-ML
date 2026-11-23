# Bản đồ báo cáo và dữ liệu thực nghiệm

Bắt đầu từ [README ở gốc](../README.md) và [hướng dẫn đọc mã](../LEARNING_GUIDE.md).

[Tổng kết hai thí nghiệm Q-Learning](ql_final_review.md) ghi quyết định giữ biến thể nào,
kết quả trên từng bộ seed test và các hàm cần mở khi thuyết trình.

[Thí nghiệm ưu tiên pha chưa cập nhật](ql_untried_first/README.md) so sánh FT,
MP, QL-Original và QL dùng `untried-first` trên hai bản đồ. Báo cáo kèm kết
quả từng lượt và thống kê theo controller. Stage 7 và Stage 8 lưu bối cảnh các model đối chứng.

[Thí nghiệm state có hướng chờ xanh lâu nhất](ql_overdue_state/README.md) so sánh
FT, MP, QL-Original và QL-Overdue trên Cologne1, kèm kết quả từng lượt, coverage
và đối chiếu khoảng thiếu xanh theo movement.

[Thí nghiệm thu nhỏ bảng Q bằng hướng đông nhất](ql_busiest_state/README.md)
so sánh QL-Busiest với QL-Overdue sau khi huấn luyện lại, kèm kết quả của
FT, MP và QL-Original trên cùng ba seed test mới.

[Kiểm tra độ tái lập](reproducibility_check.md) ghi lại kết quả phát lại mới
với cùng lịch đèn, đối số khởi chạy và cổng TraCI.

[Phân tích Q-Learning](q_learning_review.md) đối chiếu log và checkpoint,
đồng thời đề xuất thử Guard và QL thuần trên cùng bảng Q.

[Huấn luyện và đánh giá Q-Learning](ql_validation_test.md) ghi kết quả
train, validation và test giữ riêng của FT, MP, QL và Guard trên hai bản đồ.

| Stage | Nội dung | Nên mở trước |
|---|---|---|
| 1 | Khảo sát hai map, TLS, movement và dữ liệu gốc | `stage1/README.md` |
| 2–3 | Môi trường chung, pha an toàn, clock/clearance và kiểm chứng | README từng stage |
| 4 | Fixed-Time và Max-Pressure | `stage4/README.md` |
| 5 | Tabular Q-Learning ban đầu, model3ep | `stage5/README.md`, `stage5/train/` |
| 6 | Đánh giá FT/MP/QL trên bộ seed chung | `stage6/README.md`, bảng tổng hợp và `runs/` |
| 6.1 | Đối chiếu metric raw, tái lập SUMO và điều tra safety | `stage6_1/README.md`, `verification.json` |
| 6.2 | Replay Q, state aliasing, coverage và tín hiệu thiếu G | `stage6_2/README.md`, `analysis.json` |
| 7 | Model15ep, validation/khóa model/test độc lập tại thời điểm đó | `stage7/README.md`, `analysis/summary.csv` |
| 8 | State4bin; kiểm vòng lặp, Guard nếu cần, ổn định và hướng dẫn học | `stage8/README.md`, `summary.csv` |

## Kết quả quan trọng và bằng chứng

- **Quan trọng:** README kết luận, model cuối + metadata, selection/manifest,
  summary CSV, các biểu đồ chính. Stage7 model được tham chiếu trong
  `stage7/models/selection.json`, nằm tại `stage7/training/<map>/exploration_7/`.
  Stage8 model mới ở `stage8/training/cologne1/cologne1_q_table.json`.
- **Bằng chứng:** training.csv, từng `episode_*/q_table.json`, actions.csv,
  signals.csv, tripinfo.xml, statistics.xml, collisions.xml, warnings.log,
  command.txt và metadata/result mỗi lượt. Stage7: `training/`, `validation/`,
  `test/`, `analysis/`; Stage8 dùng các thư mục tương tự, thêm
  `checkpoint_validation/` và bảng/evidence offline ở gốc Stage8.
- **Công cụ:** script phân tích và Stage 6.1 probe/replay trong `tools/`, bộ test trong `tests/`.
  Chúng phục vụ kiểm chứng, không phải các lớp cần học để hiểu Q-Learning.

Không xóa dữ liệu gốc trong `data/`, model cũ, các repeat xấu, checkpoint,
manifest/SHA, raw CSV/XML/log hoặc báo cáo lịch sử. Không chỉ giữ mean rồi
xóa raw: cần raw để kiểm teleport/cohort, thời gian xanh và replay công thức Q.
Train không có tripinfo thì arrival thô không được gọi là completion hợp lệ.

## Tái tạo

Dùng `.venv` và bản SUMO được ghi trong metadata; lệnh chạy thật nằm trong
manifest, README và `command.txt` của từng lượt. Luôn chọn **output mới**.
Ví dụ phân tích offline, không khởi chạy SUMO:

```powershell
.\.venv\Scripts\python.exe tools/analyze_ql.py --output reports/my_stage6_2
.\.venv\Scripts\python.exe tools/stage7_summary.py --output reports/my_stage7
.\.venv\Scripts\python.exe tools/stage8_summary.py --output reports/my_stage8
```

Stage7 script đọc bộ raw Stage7 cố định; Stage8 script đọc bộ raw Stage8
cố định và từ chối ghi đè bốn tệp xuất. Đọc README từng stage trước khi train/
evaluate lại: các seed31001–31003 đã dùng thiết kế Stage8, không còn là test
hoàn toàn chưa biết. Không cộng điểm hai map hoặc kết quả train/evaluate.

## Giai đoạn 5 — Rút gọn mã và rà soát lưu trữ

Phạm vi là một người chạy mô phỏng hoặc đọc kết quả trên máy cá nhân.

1. **Định dạng trace tín hiệu.** FT, MP, QL train, evaluate và probe từng
   khai báo riêng cùng 10 cột và cách mã hóa mảng trong `signals.csv`.
   Giờ cả năm dùng `SIGNAL_FIELDS` và `write_signal_sample()` trong
   [mô-đun metric chung](../src/evaluation_metrics.py). `actions.csv` vẫn
   ghi các trường riêng của từng runner, như pressure hoặc Q trước/sau.
2. **Cấu hình pytest.** `pytest.ini` đã cung cấp các đường dẫn import mà
   `conftest.py` thiết lập lần nữa. Pytest thu đủ 177 test với
   `--collect-only --noconftest` trước khi bỏ tệp lặp.
3. **Runner chẩn đoán và thuật toán.** `src/ft.py`, `src/mp.py` và
   `src/q_learning.py evaluate` có định dạng chẩn đoán mà các báo cáo và
   test lịch sử đang đọc. Dùng `src/evaluate.py` khi cần metric so sánh
   chung, và mở `fixed_time()`, `calculate_pressures()`,
   `update_q_value()` hay `greedy_steps()` khi thuyết trình thuật toán.
   `tools/` phục vụ phân tích, phát lại và vẽ từ các lượt đã lưu.

Luồng trong `SumoEnv.step()` nối an toàn pha với mẫu theo giây và reward.
`run_episode()` ghi Q trước/sau để replay từng cập nhật. Hai hàm giữ những
kiểm tra gắn với luồng đó. Bộ replay Guard ở `tools/stage8_summary.py`
tái dựng quyết định từ trace độc lập với phần chọn action trong evaluate.

Đối chiếu trước và sau chỉnh sửa trên Ingolstadt1 bằng `src/ft.py`,
`src/mp.py`, `src/q_learning.py train` một episode, `src/evaluate.py` với
Guard và `tools/stage6_1_probe.py` mode C. Cả 18 tệp được chọn để đối chiếu,
gồm CSV, JSON kết quả và Q-table, khớp từng byte. Metadata và log khởi chạy
lưu đường dẫn và thông tin phiên bản theo từng lượt. `python -m pytest -q`
pass 177 test. Ba công cụ phân tích offline ở trên tái tạo `analysis.json`
của Stage 6.2 và hai bảng `summary.csv` của Stage 7 và Stage 8 khớp nội dung.

**Đề xuất lưu trữ.** Giữ các báo cáo Stage 1–8 ở vị trí hiện tại vì một số
test và công cụ replay dùng trực tiếp đường dẫn đến XML, CSV và checkpoint.
Lưu các lượt mới ngoài Git như [Giai đoạn 4](ql_validation_test.md), kèm
manifest đầu vào, phiên bản SUMO, checkpoint và SHA của các tệp chứng cứ.
Khi cần lưu dài hạn, tạo một bản nén theo đợt, kiểm SHA và giải nén thử
vào thư mục riêng, rồi chạy công cụ tổng hợp để đối chiếu bảng. Việc nén,
di chuyển hoặc xóa dữ liệu lịch sử chưa được thực hiện.
