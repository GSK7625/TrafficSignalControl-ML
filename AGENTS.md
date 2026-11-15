# AGENTS.md — TrafficSignalControl-ML

## Mục tiêu

Đây là dự án **học tập bằng Python + SUMO/TraCI**, không phải sản phẩm thương mại.

- Thuật toán chính: **Q-Learning** và **DQN**.
- Thuật toán đối chứng: **Fixed-Time (FT)** và **Max-Pressure (MP)**.
- Phạm vi trước mắt: `data/cologne1` và `data/ingolstadt1`, mỗi bản đồ một bộ điều khiển. Chưa làm điều khiển nhiều nút trên `cologne3` nếu chưa được giao.
- Người mới học phải **đọc được, giải thích được và chỉ vào code để thuyết trình**.

**Ưu tiên: đúng lý thuyết và an toàn pha đèn > dễ hiểu > ít code > tốc độ.**

## 1. Phong cách làm việc: đơn giản như Ponytail

Tham khảo: https://github.com/DietrichGebert/ponytail

- **Hiểu code cũ trước, sửa ít nhất có thể nhưng phải hoàn chỉnh.**
- Tận dụng code đang có, thư viện chuẩn và thư viện đã cài; không tự tạo framework.
- Ưu tiên hàm Python thông thường, `if`, `for` và tên biến rõ nghĩa.
- Không tạo base class, factory, interface, lớp trung gian hoặc nhiều tầng thư mục chỉ để “đúng kiến trúc”.
- Chỉ dùng class nếu thực sự cần, ví dụ mạng neural kế thừa `torch.nn.Module`.
- **Ngắn không bằng dễ hiểu:** không viết một dòng phức tạp thay cho vài dòng dễ giải thích.
- Chỉ viết chú thích khi cần giải thích **vì sao** hoặc liên hệ với công thức; không chú thích từng dòng hiển nhiên.
- Mỗi hàm có một mục đích rõ ràng. Không tách hàm vô nghĩa, cũng không dồn mọi thứ vào một hàm quá dài.
- Không làm tính năng “để sau”, không thêm tùy chọn/cấu hình/phụ thuộc không cần thiết.

**Đơn giản hóa cách viết, không được lược bỏ phần bắt buộc của thuật toán.**

## 2. Quy trình cho coding agent

**Trước khi sửa:**
1. Đọc `AGENTS.md`, kiểm tra Git và các tệp liên quan. Không đoán cấu trúc hay kết quả chạy.
2. Nói ngắn gọn: mục tiêu, tệp cần sửa, giả định quan trọng và cách kiểm tra.
3. Chỉ làm **giai đoạn/nhiệm vụ được giao**. Nếu còn thiếu thông tin có thể làm sai thuật toán hoặc tín hiệu đèn, hỏi trước; với chi tiết nhỏ, chọn cách đơn giản và nêu giả định.

**Khi sửa:**
- Ưu tiên thay đổi nhỏ; không refactor rộng hoặc đổi tên hàng loạt ngoài nhiệm vụ.
- Không chỉnh sửa dữ liệu gốc trong `data/` để cải thiện kết quả; nếu cần biến thể, tạo bản riêng và nói rõ.
- Không tự xóa dữ liệu, cài thêm thư viện nặng, chạy train dài, commit hoặc push.
- Không dùng `try/except` để che lỗi; sửa đúng nguyên nhân.

**Sau khi sửa:**
- Chạy kiểm tra nhỏ cho phần logic vừa viết; nếu không chạy được phải nói rõ lý do.
- Báo cáo ngắn: **đã sửa gì → chạy bằng lệnh nào → kết quả thực tế → phần chưa kiểm tra**.
- Chỉ rõ **tên tệp/hàm** tương ứng với kiến thức cần thuyết trình.
- Không khẳng định “đã chạy đúng” chỉ vì code trông hợp lý.

## 3. Thiết kế tối thiểu

Chỉ tạo tệp khi thực sự cần. Hướng tới:

- Một phần **môi trường SUMO dùng chung**: khởi chạy/reset, lấy state, áp dụng pha an toàn, tiến mô phỏng, tính reward/metric.
- Các phần thuật toán riêng, dễ tìm: `ft`, `mp`, `q_learning`, `dqn`.
- Lệnh chạy train/evaluate đơn giản; kết quả CSV và đồ thị Matplotlib khi cần.

Không cần web app, database, API, Docker, framework RL hoặc hệ thống logging lớn. Không bắt buộc đổi cấu trúc thư mục đang hoạt động tốt.

## 4. Các quy tắc không được làm sai

### SUMO và tín hiệu đèn

- Khảo sát `*.net.xml`, `*.rou.xml`, `*.sumocfg`, TLS ID, controlled links, pha và thời gian mô phỏng **trên từng bản đồ** trước khi viết bộ điều khiển.
- Chỉ coi **pha xanh hợp lệ** là action. Pha vàng và tất cả đỏ là **chuyển tiếp an toàn**, không phải action tùy chọn để agent học.
- Không chuyển trực tiếp giữa hai pha xanh xung đột. Thiết kế chuyển tiếp phù hợp cho từng bản đồ; không mặc định `cologne1` giống `ingolstadt1`.
- Xác định rõ thời điểm quyết định, thời gian xanh, thời gian vàng, cách đo queue và reward. Tránh đếm trùng làn/xe.
- Các thuật toán dùng cùng cơ chế thực thi pha và giới hạn an toàn.

### Fixed-Time (FT)

- Điều khiển các pha xanh theo thứ tự hợp lệ, **mỗi pha xanh đúng 30 giây**.
- Thời gian vàng/tất cả đỏ tính riêng, không lấy vào 30 giây xanh.
- Kiểm tra thời gian pha bằng đồng hồ mô phỏng SUMO, không chỉ bằng cấu hình.

### Max-Pressure (MP)

- Tính pressure từ **chênh lệch hàng đợi đầu vào và đầu ra** của các chuyển động, rồi cộng theo pha (với trọng số nếu cần, và phải giải thích).
- Không thay MP bằng việc chọn làn đông nhất. Không mặc định pressure chỉ bằng tổng xe chờ đầu vào.
- Nếu thiếu dữ liệu hoặc mapping đầu vào/đầu ra, nêu giới hạn và chốt cách tính trước khi code.

### Q-Learning

- State phải **rời rạc**, action là pha xanh hợp lệ, reward có ý nghĩa đo rõ ràng.
- Có **epsilon-greedy** và cập nhật Q-table theo:
  `Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])`
- Nếu `done=True`, target chỉ là `r`, không cộng giá trị trạng thái sau.
- Train được cập nhật Q; evaluate không cập nhật, mặc định chọn action tham lam. Có lưu/tải Q-table.

### DQN

- Dùng PyTorch trực tiếp với **MLP nhỏ**; không sử dụng framework RL lớn.
- Phải có: **online network, target network, replay buffer, epsilon-greedy, minibatch update**.
- Target: `r + gamma * (1 - done) * max_a' Q_target(s', a')`.
- Không truyền gradient qua target; cập nhật target theo lịch rõ ràng. Xử lý đúng `done`, shape và action hợp lệ.
- Có lưu/tải model; evaluate không khám phá hoặc cập nhật trọng số theo mặc định.

**Đối với Q-Learning và DQN:** ưu tiên dùng cùng định nghĩa state thông tin gốc, action, reward và khoảng điều khiển; Q-Learning có thể rời rạc hóa state, DQN có thể chuẩn hóa đầu vào. Mọi khác biệt phải được ghi rõ.

## 5. Đánh giá công bằng, không chạy theo điểm đẹp

- FT, MP, Q-Learning và DQN phải dùng **cùng bản đồ, nhu cầu xe, thời gian mô phỏng, giới hạn an toàn và cách đo metric**.
- Huấn luyện RL riêng; **không so kết quả trong lúc train** với kết quả evaluate của baseline.
- Ghi rõ định nghĩa/đơn vị của: **Average Waiting Time, Travel Time, Queue Length, Time Loss, Throughput**, đặc biệt cách xử lý xe chưa kết thúc chuyến.
- Lưu kết quả thô; nếu chạy nhiều seed, kiểm tra seed có thật sự tạo biến thiên trong mô phỏng hay chỉ ảnh hưởng quá trình học.
- Không sửa dữ liệu, bỏ lần chạy kém hoặc khẳng định DQN/QL phải thắng FT/MP.

## 6. Kiểm tra tối thiểu nhưng đúng chỗ

Không tạo bộ test phức tạp. Ưu tiên các phép kiểm tra nhỏ, chạy được:

- Pha xanh không xung đột; chuyển pha an toàn.
- FT đúng 30 giây xanh; MP đúng một ví dụ pressure tính tay.
- Q-Learning đúng một phép cập nhật Q, gồm trường hợp `done`.
- DQN có shape đúng, loss hữu hạn, target không có gradient, lưu/tải được.
- Evaluate sinh được kết quả và không lẫn metric giữa các thuật toán.

Dùng `assert` hoặc `pytest` nếu có sẵn. Chỉ viết thêm test khi logic dễ sai hoặc cần kiểm chứng.

## 7. Quy tắc dễ thuyết trình

Sau mỗi thuật toán, chỉ ra trong code:
1. **Input** lấy ở hàm nào?
2. **Action/pha** được chọn ở đâu?
3. **Công thức chính** nằm ở đâu?
4. **SUMO nhận lệnh** ở đâu?
5. **Reward và kết quả** được tính ở đâu?

Nếu không thể giải thích trực tiếp từ code, hãy **viết lại cho dễ đọc**, không xóa logic cần thiết.

**Kết luận:** viết phần ít nhất đủ để **đúng, chạy được, kiểm tra được và trình bày được**.
