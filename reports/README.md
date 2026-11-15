# Bản đồ báo cáo và dữ liệu thực nghiệm

Đọc `../LEARNING_GUIDE.md` để hiểu code, rồi **stage8/README.md** để biết
kết quả mới nhất. Đọc Stage7 trước nếu cần bối cảnh model đối chứng;
Stage6.1 giải thích vì sao cùng SUMO seed chưa bảo đảm cùng quỹ đạo.

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
- **Công cụ:** script phân tích ở gốc repository, test, Stage6.1 probe/replay.
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
.\.venv\Scripts\python.exe analyze_ql.py --output reports/my_stage6_2
.\.venv\Scripts\python.exe stage7_summary.py --output reports/my_stage7
.\.venv\Scripts\python.exe stage8_summary.py --output reports/my_stage8
```

Stage7 script đọc bộ raw Stage7 cố định; Stage8 script đọc bộ raw Stage8
cố định và từ chối ghi đè bốn tệp xuất. Đọc README từng stage trước khi train/
evaluate lại: các seed31001–31003 đã dùng thiết kế Stage8, không còn là test
hoàn toàn chưa biết. Không cộng điểm hai map hoặc kết quả train/evaluate.

## Lưu trữ lâu dài — đề xuất, chưa thực hiện

Giữ model/metadata/summary/README/manifest dễ truy cập. Có thể đóng gói raw
theo stage bằng archive nén và lưu ở kho ngoài Git có backup, SHA-256 cùng
danh sách file; kiểm giải nén và tái tạo bảng trước khi xem xét thay đổi cách
lưu. Git LFS là lựa chọn khác cho binary/trace lớn, cần thống nhất quota,
remote và cách lấy dữ liệu trước. CSV/XML nén thường hiệu quả nhưng mất khả
năng diff trực tiếp. Không tự migrate lịch sử Git, di chuyển hoặc xóa file đã
commit; mọi thay đổi lưu trữ cần một nhiệm vụ riêng.
