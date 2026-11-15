# Kế hoạch giai đoạn 3

Phạm vi theo yêu cầu: sửa tối thiểu môi trường giai đoạn 2, sau đó FT 30 s;
không sửa dữ liệu, cài gói, train, commit/push hoặc triển khai thuật toán khác.
Làm trực tiếp trong checkout hiện tại để dùng `.venv` và giữ kết quả tại đây.

1. Đọc báo cáo/log, đối chiếu network/route/config và movement. Lưu hash đầu vào.
2. Test trước: runtime chỉ giữ G, không có foes cùng mở, mọi link có G ở ít nhất
   một action. Đổi g/s thành r; giữ ID và thứ tự action gốc. Giữ clearance đuôi xe.
3. Vàng cố định riêng từng map, kiểm tra theo tốc độ/giảm tốc; đỏ >=2 s và chờ
   hết xe. Chạy lại smoke mọi cặp action đủ 3600 s trước khi viết FT chính thức.
   Nếu còn vấn đề an toàn quan trọng không giải quyết được thì dừng, báo cáo.
4. Test trước rồi thêm vòng lặp FT trong ft.py: thứ tự cố định, không dùng queue
   hoặc reward để chọn action; chỉ gọi env.step. Ghi signals/actions và log SUMO.
5. Chạy 3600 s/map, seed 23423 và thêm 2 seed nếu thời gian cho phép. Kiểm chứng
   bằng timestamp/tín hiệu SUMO, đầu cuối episode, collision/teleport/cảnh báo.
6. Viết README với kết quả thực, giới hạn, định nghĩa và vị trí hàm thuyết trình;
   chạy toàn bộ pytest, kiểm tra diff/hash. Không so chất lượng thuật toán.

Các trường hợp cần kiểm tra: cắt ở vàng/đỏ/xanh, nút bị chặn, xe bus dài hơn
edge ra, action sai, giữ cùng action, không mất movement khi đóng g/s.

Đã chạy baseline: `python -m pytest -q`: 13 passed. Git ban đầu sạch, main.
