# Giai đoạn 2 — Độ tái lập của SUMO

**Ngày 19/11/2026, commit `9239374`.** Cologne1 tái hiện sai khác quỹ đạo ở cùng bước như Stage 6.1 ngay cả khi các lượt có cùng toàn bộ đối số SUMO, cổng TraCI, đường dẫn đầu ra và lịch đèn. Xe đầu tiên lệch là `142128_415_0`, trong bước `[25853, 25854)`. Chưa xác định được biến nội bộ hoặc phép tính C++ đầu tiên gây lệch.

## Phép thử mới

Chạy 16 tiến trình SUMO nối tiếp. Mỗi tiến trình chạy 660 giây mô phỏng, seed `42`, bước 1 giây. Hai nhóm phát lại dùng 50 lệnh đặt đèn khôi phục từ `foes_0/samples.json`, gồm lệnh reset và các lần giữ đỏ để clear nút. Mỗi nhóm dùng một đường dẫn đầu ra cố định và cổng `54240`. Sau khi SUMO đóng, lưu thư mục từng lượt rồi mới dùng lại đường dẫn cho lượt kế tiếp. Phát lại vẫn gửi setter đèn và `simulationStep()` qua TraCI; các nhóm này lấy quỹ đạo từ FCD do SUMO ghi.

| Nhóm | Lượt | Kết quả |
|---|---:|---|
| Cologne1 phát lại, FCD | 8 | 7 lượt ở **10,836552277279 m/s**, 1 lượt ở **10,052142789202 m/s** |
| Cologne1 phát lại, FCD và lưu state RNG | 4 | 2 lượt mỗi tốc độ. Cả 4 cây XML state đã parse, gồm `rngState`, có cùng SHA-256. |
| Cologne1 FT mode C, quan sát và tripinfo | 2 | Trace và samples trùng theo hash, cùng **47.801 xe·s** queue và **244/425** chuyến hợp lệ |
| Ingolstadt1 FT mode C, quan sát và tripinfo | 2 | Trace và samples trùng theo hash, cùng **6.256 xe·s** queue và **243/276** chuyến hợp lệ |

Cả 11 cặp phát lại khác quỹ đạo trong cùng nhóm đều lệch trước tiên ở xe trên, tại FCD `25853`. FCD gắn mốc đầu bước, tương ứng TraCI đọc trạng thái sau bước tại `25854`. Xe ở cùng làn nội bộ `:cluster_357187_359543_13_0`, với vị trí dọc làn khoảng **1,830 m** hoặc **1,046 m**. Hai lần FT mode C trùng nhau trên mỗi bản đồ chỉ mô tả bốn lượt quan sát mới. [Bằng chứng đối chiếu Stage 6.1](stage6_1/README.md)

## Các yếu tố đã đối chiếu

| Yếu tố | Bằng chứng và phạm vi |
|---|---|
| Binary và đầu vào | SUMO 1.28.0 trên Windows, SHA-256 binary và SHA các tệp mạng, route và config Cologne1 khớp Stage 6.1. Manifest lưu SHA đầu vào cả hai bản đồ. Hai config trỏ vào cùng tệp route gốc cho mọi lần lặp tương ứng. |
| Seed và luồng mô phỏng | `--seed 42` rõ ràng, `random=false`, `threads=1`, `num-clients=1`, `device.rerouting.threads=0` theo argv và template của binary. Seed khám phá Q-Learning không tham gia các lượt FT/phát lại. |
| TraCI và thứ tự chạy | Python 3.11.9 nạp `traci` và `sumolib` thật từ thư mục `tools` của bản cài SUMO qua `PYTHONPATH`. Mỗi lượt phát lại có một tiến trình mới, đóng trước lượt kế tiếp. Trong từng nhóm phát lại, đối chiếu argv thực từ `process.args` và toàn bộ lệnh đèn đều khớp. Khác biệt cổng hoặc tên đường dẫn không cần xảy ra để quan sát thấy hai quỹ đạo. |
| Quan sát và output | Stage 6.1 đã lưu cả hai nhánh qua nhiều chế độ tripinfo và observer. Thử nghiệm mới quan sát được cả hai nhánh từ FCD ngay cả khi bỏ các getter queue, observer và logic quyết định trong Python. Cây XML state giống nhau chỉ xác nhận các trường SUMO đã xuất tại mốc đó. |
| Yếu tố còn mở | Trạng thái nội bộ chưa xuất, thứ tự xử lý xung đột trong SUMO, DLL đang nạp và bố trí bộ nhớ khi khởi chạy chưa được đối chiếu. Mới dùng một build và một máy. Cảnh báo `Missing yellow phase` của chương trình `0` xuất hiện cả ở nhóm phát lại hai nhánh và đã quan sát được khi reset mà chưa tiến mô phỏng. |

Tài liệu SUMO nêu kỳ vọng tái lập khi dùng cùng phiên bản, đối số và đầu vào. Mã v1.28.0 duyệt một tập cặp con trỏ trong `MSLink::recheckSetRequestInformation()`, nên thứ tự khởi tạo xung đột là một giả thuyết cụ thể. Stage 6.1 đã quan sát cùng thứ tự foe trả qua TraCI mà vẫn ra hai nhánh, vì vậy cần kiểm các giá trị C++ mà phép tính vận tốc thực sự dùng. [Tài liệu SUMO về tái lập](https://sumo.dlr.de/docs/Simulation/Randomness.html#reproducibility) · [Mã SUMO v1.28.0](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/MSLink.cpp#L547-L557)

## Kiểm tra các lượt lịch sử

Đối chiếu `command.txt`, `metadata.json` và `result.json` với các dòng test trong CSV. Trong mỗi cặp, SHA dữ liệu, SHA mã chạy mô phỏng, model, cấu hình môi trường, phiên bản SUMO, giới hạn thời gian và seed ghi nhận đều khớp. `command.txt` của runner đánh giá được ghi trước khi TraCI nối thêm cổng, nên các tệp này xác nhận cấu hình mô phỏng nhưng thiếu cổng thực của lượt lịch sử. Ba cặp Stage 8 có khác SHA ở `test_stage8.py`, trong đó một cặp còn khác `stage8_summary.py`.

| Bộ test | Cặp khác số chuyến hợp lệ | Chênh lớn nhất trong một cặp |
|---|---:|---:|
| Stage 7 Cologne1 | 6/12 | **811** chuyến, QL-15ep-e7 |
| Stage 7 Ingolstadt1 | 0/12 | **0** chuyến |
| Stage 8 Cologne1 | 6/9 | **928** chuyến, QL-NewState |

So sánh theo từng seed và lần lặp khi đánh giá các bộ điều khiển. Với Cologne1, báo cả kết quả từng lượt, khoảng biến thiên trong cùng seed, queue và teleport. Dùng cùng nhu cầu xe và cùng bộ seed cho mọi bộ điều khiển. Các lượt cùng seed vẫn có thể đi theo những quỹ đạo khác nhau. [Stage 7](stage7/analysis/runs.csv) · [Stage 8](stage8/runs.csv)

## Cách chạy lại và phép thử tiếp theo

Thư mục `TrafficSignalControl-stage2-96746490` trong Documents của tài khoản chứa `manifest.json`, `summary.json`, toàn bộ FCD/state/XML/CSV/log và lệnh thực từng lượt. Bản chạy này dùng `run_replays.py` ngoài repository, gọi lại `replay_signals()`, `run_probe()` và `first_difference()` trong [tools/stage6_1_probe.py](../tools/stage6_1_probe.py). Chạy lại từ gốc dự án với bản sao script trong một thư mục mới. Nếu đường dẫn checkout hoặc binary khác, tạo một probe Cologne1 seed 42 dài 660 giây theo hướng dẫn Stage 6.1 rồi chỉnh biến `source` trong bản sao script để trỏ tới probe mới. Đối chiếu lịch đèn của nguồn mới trước khi so kết quả với bảng trên.

```powershell
$documents = [Environment]::GetFolderPath('MyDocuments')
$saved = Join-Path $documents 'TrafficSignalControl-stage2-96746490'
$newRun = Join-Path $documents ("TrafficSignalControl-stage2-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $newRun -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath (Join-Path $saved 'run_replays.py') -Destination $newRun
.\.venv\Scripts\python.exe (Join-Path $newRun 'run_replays.py')
.\.venv\Scripts\python.exe -m pytest -q
```

Bộ test hiện có trả **176 pass, 0 fail trong 15,40 giây**. Dữ liệu gốc và bằng chứng Stage 6.1–8 được giữ riêng khỏi đầu ra mới.

Phép thử tìm nguyên nhân tiếp theo dùng build chẩn đoán cùng tag `v1_28_0` và giữ lịch đèn cố định. Ghi cặp ID link và hình học xung đột tại lúc khởi tạo, rồi ghi xe dẫn đầu, xe xung đột, giá trị vận tốc ứng viên và lần lấy mẫu RNG cho xe `142128_415_0` trong bước `[25853, 25854)`. Đối chiếu hai trace nội bộ nếu thu được ít nhất một lượt của mỗi nhánh. Chưa chạy build này.
