# Giai đoạn 4 — Huấn luyện, validation và test Q-Learning

## Thiết kế chốt trước khi chạy

**Giả thuyết:** Guard 300 giây giảm khoảng thời gian movement thiếu xanh khi gắn vào QL-Original. So sánh QL thuần với QL + Guard trên cùng bảng Q và cùng bộ seed để tách tác động của Guard.

| Tập | Thiết kế | Số lượt |
|---|---|---:|
| Training | Hai bản đồ, exploration seed `7` và `17`, mỗi model học mới 15 episode từ bảng rỗng. SUMO seed `23423` trong mọi episode. | 60 episode |
| Validation | Hai bản đồ, hai model mỗi bản đồ, seed SUMO `42` và `12345`, hai lần lặp mỗi seed. Chạy từng model với QL thuần và Guard. | 32 evaluation |
| Test | Hai bản đồ, ba seed SUMO giữ riêng `52101`, `52102`, `52103`, hai lần lặp mỗi seed. So FT, MP, QL và Guard. Hai biến thể QL dùng cùng model đã khóa cho mỗi bản đồ. | 48 evaluation |

Giữ bin `0 4`, `α=0,1`, `γ=0,95` theo quyết định và epsilon giảm tuyến tính từ `1` đến `0,05`. Giữ đầu vào, thời gian mô phỏng 3.600 giây, metric và cơ chế chuyển pha chung. Chạy tuần tự. Trong từng nhóm seed, đảo thứ tự QL và Guard hoặc xoay thứ tự bốn controller theo lần lặp đã định trước.

Chỉ chọn checkpoint cuối episode 15. Trên mỗi bản đồ, xếp hai model theo QL thuần trên validation, lần lượt ưu tiên completion trung bình cao, completion thấp nhất cao, teleport trung bình thấp, gap lớn nhất thấp và exploration seed nhỏ nếu mọi tiêu chí hòa. Khóa đường dẫn và SHA của model trước khi chạy test. Guard đạt tiêu chí định trước trên validation nếu gap cực đại giảm ở từng seed và completion trung bình gộp không giảm. Test chạy đủ cả hai biến thể dù Guard đạt hay trượt tiêu chí này.

So sánh theo từng bản đồ. Báo trung bình, min–max và chênh giữa các lần lặp cùng seed. Waiting Time, Travel Time và Time Loss dùng nhóm hoàn thành hợp lệ. Completion, Throughput, Queue và Teleport dùng cùng định nghĩa từ `src/evaluation_metrics.py`. Train chỉ dùng để xem coverage, reward và checkpoint.

## Training và độ ổn định của policy

Đã train bốn model mới và replay toàn bộ 60 episode từ log, gồm RNG khám phá, reward, cập nhật terminal và từng checkpoint. SHA model cuối trùng checkpoint 15 của chính lượt train.

| Bản đồ | Seed khám phá | State đã gặp | Cặp đã update / mẫu số | Cặp chỉ update một lần | Đổi greedy 14 → 15 | Max ΔQ 14 → 15 |
|---|---:|---:|---:|---:|---:|---:|
| Cologne1 | 7 | 115 | 467/920 · 50,76% | 234 | 35/113 | 17,19 |
| Cologne1 | 17 | 122 | 472/976 · 48,36% | 271 | 31/119 | 18,87 |
| Ingolstadt1 | 7 | 17 | 49/51 · 96,08% | 8 | 8/17 | 5,42 |
| Ingolstadt1 | 17 | 20 | 50/60 · 83,33% | 8 | 9/20 | 4,01 |

Coverage tính trên state đã gặp khi train nhân với số action hợp lệ. Tỷ lệ đổi greedy và max ΔQ chỉ xét các state xuất hiện trong cả hai checkpoint. Ở hai model cuối khác seed khám phá, greedy khác trên **75/93** state chung của Cologne1 và **12/17** state chung của Ingolstadt1. Cologne1 vẫn thêm 2 và 3 state ở episode 15; cả bốn model vẫn đổi greedy giữa hai checkpoint cuối. Kết quả chưa cho thấy hội tụ. Mỗi seed khám phá chỉ có một lượt train mới, và SUMO cùng seed vẫn có thể biến thiên trên Cologne1.

## Sửa phép đối chiếu xe bị loại trong validation

Lượt Cologne1 QL, exploration seed 7, SUMO seed 12345, repeat 1 chạy đủ 3.600 giây và đóng SUMO. Hậu xử lý dừng vì `unfinished=212` trong tripinfo còn SUMO ghi `running=211`. Xe `123353_406_0` bị teleport qua edge đích và bị loại, với `vaporized="teleport"`. Xe đó chiếm đúng một đơn vị chênh lệch. [Cách SUMO 1.28.0 ghi TripInfo](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/devices/MSDevice_Tripinfo.cpp#L356-L385)

`src/evaluation_metrics.py` giữ định nghĩa chuyến hợp lệ và `unfinished`. Trường `removed_without_arrival` ghi số xe đã bị loại mà chưa đến đích; phép kiểm giờ so `unfinished − removed_without_arrival` với `running`. Test hồi quy tái hiện lỗi trước khi sửa, sau đó cả hai trường hợp trong fixture đều pass. Lượt validation bị lỗi được tính lại từ chính XML và CSV đã lưu, cho **1.365/2.015** chuyến hợp lệ. Không chạy lại lượt này. Manifest lưu SHA của hai phiên bản xử lý metric và các tệp thô dùng để phục hồi.

## Validation và khóa model

Mỗi cấu hình trong bảng có 4 lượt validation. Gap là giá trị lớn nhất trong 2 lần lặp của từng seed.

| Bản đồ | Seed khám phá | Completion QL → Guard | Gap seed 42 (s) | Gap seed 12345 (s) | Guard đạt tiêu chí |
|---|---:|---:|---:|---:|---|
| Cologne1 | **7** | 69,90% → 73,40% | 1.820 → 2.380 | 2.500 → 1.740 | Không |
| Cologne1 | 17 | 41,76% → 86,41% | 3.440 → 670 | 3.600 → 1.770 | Có |
| Ingolstadt1 | **7** | 97,12% → 97,12% | 438 → 438 | 530 → 530 | Không |
| Ingolstadt1 | 17 | 88,14% → 88,14% | 758 → 758 | 1.114 → 1.114 | Không |

Checkpoint 15 của exploration seed **7** được chọn cho cả hai bản đồ theo QL thuần. Trên Cologne1, Guard đạt tiêu chí khi gắn vào bảng e17 nhưng trượt trên bảng e7 đã chọn. So sánh test dùng e7 cho cả QL và Guard theo thiết kế. `selection.json` được khóa trước lệnh test đầu tiên, với SHA-256 `688418be16973639a5a1f25324bcc227f8261c3d6e1935e6344a89049d9a9ab8`.

## Kết quả test giữ riêng

Mỗi dòng là trung bình không trọng số của 6 lượt đầy đủ, gồm 3 seed × 2 lần lặp. Nhu cầu mỗi lượt là 2.015 chuyến trên Cologne1 và 1.716 trên Ingolstadt1.

| Bản đồ | Controller | Wait (s/xe) | Travel (s/xe) | Queue (xe) | Time Loss (s/xe) | Throughput (xe/h) | Completion | Teleport (sự kiện/lượt) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cologne1 | FT | 144,05 | 187,48 | 73,58 | 164,45 | 1.894,33 | 94,01% | 0,00 |
| Cologne1 | MP | 64,40 | 100,94 | 28,75 | 78,07 | 1.966,00 | 97,57% | 11,00 |
| Cologne1 | QL | 151,40 | 192,39 | 98,89 | 170,29 | 1.412,00 | 70,07% | 57,17 |
| Cologne1 | Guard | 141,59 | 184,46 | 83,35 | 161,61 | 1.554,17 | 77,13% | 42,33 |
| Ingolstadt1 | FT | 31,93 | 64,22 | 9,20 | 43,50 | 1.681,00 | 97,96% | 0,00 |
| Ingolstadt1 | MP | 34,21 | 66,23 | 5,74 | 45,58 | 1.670,33 | 97,34% | 1,33 |
| Ingolstadt1 | QL | 48,91 | 82,08 | 9,39 | 61,26 | 1.684,00 | 98,14% | 1,33 |
| Ingolstadt1 | Guard | 48,91 | 82,08 | 9,39 | 61,26 | 1.684,00 | 98,14% | 1,33 |

Wait và Time Loss lấy từ SUMO; Travel bằng arrival trừ actual depart. Cả ba dùng nhóm xe đến đích hợp lệ chưa từng teleport. Queue là số xe dừng trung bình theo giây trên các làn vào được đo, kể cả vàng, đỏ và clearance. Throughput sử dụng số chuyến hợp lệ quy đổi ra xe/giờ; Completion chia số chuyến hợp lệ cho toàn bộ yêu cầu. Số sự kiện teleport có thể lớn hơn số xe từng teleport, ví dụ QL Cologne1 trung bình 57,17 sự kiện trên 56,50 xe. Không có va chạm được ghi ở 48 lượt test.

- **Cologne1:** Guard tăng completion **7,06 điểm phần trăm**, đồng thời giảm gap cực đại từ **3.010 xuống 1.690 giây** so với QL thuần. Khi tính riêng theo từng seed, Guard tăng completion trung bình và giảm gap cực đại ở cả ba seed. Guard đổi action greedy 26 lần trong 6 lượt. FT và MP vẫn có completion cao hơn cả hai biến thể QL. FT ghi 0 teleport và gap 250 giây ở mọi lượt; MP có queue và Time Loss thấp nhất trong bộ test này.
- **Ingolstadt1:** Guard không kích hoạt trong 6 lượt và tất cả metric trùng với QL. QL hoàn thành trung bình 1.684 chuyến/giờ so với 1.681 của FT, còn Waiting Time là 48,91 so với 31,93 s/xe và gap cực đại là 438 so với 84 giây.

### Biến thiên giữa seed và lần lặp

| Bản đồ | Controller | Chuyến hợp lệ ở 52101 | Chuyến hợp lệ ở 52102 | Chuyến hợp lệ ở 52103 | Chênh cùng seed lớn nhất | Khoảng gap (s) |
|---|---|---:|---:|---:|---:|---:|
| Cologne1 | FT | 1.905 / 1.889 | 1.885 / 1.897 | 1.895 / 1.895 | 16 | 250–250 |
| Cologne1 | MP | 1.969 / 1.969 | 1.963 / 1.963 | 1.966 / 1.966 | 0 | 690–1.130 |
| Cologne1 | QL | 1.343 / 1.558 | 1.394 / 1.390 | 1.394 / 1.393 | 215 | 1.490–3.010 |
| Cologne1 | Guard | 1.607 / 1.607 | 1.531 / 1.531 | 1.576 / 1.473 | 103 | 1.460–1.690 |
| Ingolstadt1 | FT | 1.681 / 1.681 | 1.681 / 1.681 | 1.681 / 1.681 | 0 | 84–84 |
| Ingolstadt1 | MP | 1.654 / 1.654 | 1.662 / 1.662 | 1.695 / 1.695 | 0 | 312–388 |
| Ingolstadt1 | QL | 1.656 / 1.656 | 1.697 / 1.697 | 1.699 / 1.699 | 0 | 364–438 |
| Ingolstadt1 | Guard | 1.656 / 1.656 | 1.697 / 1.697 | 1.699 / 1.699 | 0 | 364–438 |

Đổi seed SUMO thực sự đổi giá trị `speedFactor` trong tripinfo FT. Với lần lặp đầu ở seed 52101 và 52102, thuộc tính này khác ở 1.898/1.958 ID xe chung trên Cologne1 và 1.637/1.707 trên Ingolstadt1. Hai lượt FT cùng seed 52101 có `speedFactor` khớp trên các ID chung ở cả hai bản đồ. Seed ở đây tạo biến thiên trên cùng nhu cầu xe gốc. Ba seed và hai lần lặp mỗi seed chỉ cho phép diễn giải mô tả, và model đã chọn vẫn trượt tiêu chí Guard trên validation.

## Kiểm tra và dữ liệu đã lưu

Các tệp ở thư mục `experiments/stage4-2382c8e3` ngoài Git. `protocol.md` là thiết kế đã lưu trước khi train. `plan.json`, `selection.json`, `metric_repair.json` và `commands.jsonl` lưu thiết lập, SHA, quyết định khóa và thời điểm gọi từng lệnh.

- Lệnh train dùng `src/q_learning.py train`; lệnh validation/test dùng `src/evaluate.py`. Mỗi lượt giữ `actions.csv`, `signals.csv`, metadata và log SUMO; các lượt evaluate giữ thêm tripinfo đầy đủ.
- `audit_experiment.py` replay **60 episode** với công cụ Stage 7, kiểm quyết định QL/Guard với công cụ Stage 8 và đối chiếu metric từ XML/CSV cho **80 lượt evaluate**. Audit xác nhận FT dùng 30 giây cho mọi lượt xanh đầy đủ, SHA model QL giữ nguyên trong evaluate và test dùng đúng SHA đã khóa.
- `analysis/runs.csv` chứa từng lượt, `summary.csv` chứa mean/min/max/SD, `train_replay.csv` chứa coverage và checkpoint. `seed_repeats.csv` và `guard_vs_ql_by_seed.csv` lưu các so sánh theo seed. `test_comparison.png` vẽ các điểm test cùng trung bình.
- Bộ lệnh kiểm tra `python -m pytest -q` cho **177 pass, 0 fail**. `git diff --check` đạt.

Dừng sau Giai đoạn 4. Các kết quả giữ nguyên bố trí thí nghiệm đã khóa; chưa có phép thử trên nhu cầu xe khác hoặc nhiều hơn hai lượt train độc lập mỗi bản đồ.
