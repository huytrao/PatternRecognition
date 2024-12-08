# 1. Tổng quan:
- Chủ đề: Phân loại trạng thái chú ý tinh thần bằng dữ liệu EEG
- Nguồn dữ liệu: https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/data
  - Dữ liệu EEG được thu thập trong 25 giờ từ 5 người tham gia, mỗi người điều khiển tàu mô phỏng bằng “Microsoft Train Simulator” trong 35-55 phút trên tuyến đường đơn giản.
  - Trạng thái tinh thần:
    - Tập trung: Giám sát tàu một cách thụ động, duy trì tập trung mà không cần can thiệp tích cực.
    - Mất tập trung: Không chú ý, tỉnh táo nhưng tách rời, khó phát hiện bằng biểu hiện bên ngoài.
    - Buồn ngủ: Dễ nhận thấy qua EEG (dải alpha tăng) hoặc các dấu hiệu như chớp mắt, nhịp tim.
  - Quy trình thí nghiệm:
    - 10 phút đầu: Điều khiển tập trung, theo dõi sát màn hình.
    - 10 phút tiếp: Ngừng điều khiển, không chú ý nhưng vẫn tỉnh táo.
    - 10 phút cuối: Thư giãn, nhắm mắt, ngủ gật nếu muốn.
  - Cài đặt thí nghiệm:
    - Sử dụng đầu máy “Acela Express” và đoạn đường “Amtrak-Philadelphia” dài 40 phút trong chương trình mô phỏng.
    - Đoạn đường phẳng, yêu cầu ít đầu vào điều khiển, trừ 5 phút đầu và cuối cần mức tham gia cao hơn.
    - Người tham gia duy trì tốc độ 40 mph, điều chỉnh ga và phanh qua bàn phím.
  - Quy trình thực hiện:
    - Mỗi người tham gia thực hiện 7 thí nghiệm (1 lần/ngày), 2 thí nghiệm đầu để làm quen, 5 thí nghiệm sau thu thập dữ liệu.
    - Thí nghiệm diễn ra từ 7-9 giờ tối để dễ vào trạng thái buồn ngủ trong giai đoạn cuối.
    - Người tham gia được giám sát và ghi hình để đảm bảo tuân thủ cấu trúc thí nghiệm, không có gián đoạn như di chuyển hay nói chuyện.
  <div style="text-align: center;">
    <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/1.png?raw=true" width="70"/>
</div>
- Mục tiêu:
