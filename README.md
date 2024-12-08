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
  - Sử dụng tai nghe Epoc EEG được chỉnh sửa, trang bị 12 kênh với:
    - Tốc độ lấy mẫu: 128 Hz.
    - Độ phân giải: 0,51 μV.
    - Băng thông: 0,2–43 Hz.
    - Điện cực đặt tại các vị trí: F3, F4, Fz, C3, C4, Cz, Pz (hệ thống 10–20).
    - 4 đạo trình (T3, T4, T5, T6) dùng làm dòng điện và tham chiếu, không thu thập dữ liệu.
    - Sử dụng script Matlab tùy chỉnh dựa trên eeglogger.m.
    - Trở kháng điện cực được kiểm tra trước và sau thí nghiệm, nếu không đạt yêu cầu, thí nghiệm được lặp lại.
    <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/1.png?raw=true" width="500"/>
    </div>
  - Dữ liệu:
    - Dữ liệu thô nằm trong o.data, mảng kích thước {số mẫu}x25.
    - Mỗi cột o.data(:,i) tương ứng với một kênh dữ liệu.
    - Tần số lấy mẫu: 128 Hz.
    - Danh sách kênh:
      - 1-'ED_COUNTER'
      - 2-'ED_INTERPOLATED'
      - 3-'ED_RAW_CQ'
      - 4-'ED_AF3'
      - 5-'ED_F7'
      - 6-'ED_F3'
      - 7-'ED_FC5'
      - 8-'ED_T7'
      - 9-'ED_P7'
      - 10-'ED_O1'
      - 11-'ED_O2'
      - 12-'ED_P8'
      - 13-'ED_T8'
      - 14-'ED_FC6'
      - 15-'ED_F4'
      - 16-'ED_F8'
      - 17-'ED_AF4'
      - 18-'ED_GYROX'
      - 19-'ED_GYROY'
      - 20-'ED_TIMESTAMP'
      - 21-'ED_ES_TIMESTAMP'
      - 22-'ED_FUNC_ID'
      - 23-'ED_FUNC_VALUE'
      - 24-'ED_MARKER'
      - 25'ED_SYNC_SIGNAL'
    The EEG data is in the channels 4:17.
    <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/2.png" width="500"/>
    </div>
- Mục tiêu:
  - Tích lũy thêm domain knowledge về tín hiệu não
  - Biết cách xử lý tín hiệu sóng phức tạp
  - Phân loại các trạng thái tinh thần bằng các mô hình máy học
# 2. Tổ chức file:
- Data: dữ liệu khá lớn nên đã được upload lên drive dưới đây:
  https://drive.google.com/drive/u/0/folders/1iaqDKBWp38GZGi8MHfXA52AzVIMiwGMt?fbclid=IwY2xjawHCBRRleHRuA2FlbQIxMAABHf4JmYq8Iyuaq36LuuE37YgthesIUllLeF4xDqT8WYDW-ILyXNZlyZIllg_aem_6pE42FPlIIzYKKE7j-D9iA
  - CNN data
  - SVM data and KNN data
  - XGboots data
- File code chính:
  - ***1. Tổng Quan Về Dữ Liệu EEG.ipynb***: Phân tích và và đánh giá tổng quan dữ liệu EEG thô
  - ***2.ICA dữ liệu và hiểu dữ liệu thông qua hình vẽ và mô tả EEG.ipynb***: Thực hiện ICA và trực quan hóa dữ liệu
  - ***3. feature_extraction_for_SVM_and_KNN_model.ipynb***: Thực hiện lấy dữ liệu đầu vào cho mô hình SVM và KNN
  - ***4. SVM_model baseline model.ipynb***: Thực hiện mô hình SVM
  - ***5. Feature Extraction Cnn.ipynb***: Lấy dữ liệu đầu vào cho mô hình CNN
  - ***6. cnn_model.ipynb***: Thực hiện mô hình CNN
  - ***7. XGBOOST_feature extraction and trainning model.ipynb***: Dữ liệu đầu vào cho XGBOOTS
  - ***8. Compare_model.ipynb***: So sánh giữa các model với nhau
- File báo cáo và đánh giá nhóm:
  - ***PRML2024_Midterm_Group05_report.pdf***: Báo cáo
  - ***11-Bảng-Đánh-Giá-Quá-Trình-Làm-Việc-Của-Nhóm-5..pdf***: Đánh giá
# 3. Quy trình:

  ### 3.1: Lấy dữ liệu cho SVM và KNN:
<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/3.png" width="500"/>
    </div>   
    
    - Chi tiết coi trong file notebook 3. feature_extraction_for_SVM_and_KNN_model.ipynb
  ### 3.2: Lấy dữ liệu cho CNN:
<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/4.png" width="500"/>
    </div>
    -Chi tiết coi trong file notebook 5. Feature Extraction Cnn.ipynb
  ### 3.3: Lấy dữ liệu cho XGBOOTS:
    - Chi tiết coi trong file 7. XGBOOST_feature extraction and trainning model.ipynb
# 4. Triển khai model và kết quả:
  ### 4.1: Model SVM:
  
  <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/5.png" width="500"/>
  </div>

  ### 4.2: Model KNN:
  
  <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/6.png" width="500"/>
  </div>

  ### 4.3: Model CNN:
  
  <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/7.png" width="500"/>
  </div>

  ### 4.4: XGBOOTS:
  
  <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/8.png" width="500"/>
  </div>

# 5. So sánh giữa các model:

<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="https://github.com/VietDucFCB/PatternRecognition/blob/main/image/9.png" width="500"/>
</div>

# 6. Kết luận:
- Nội dung nghiên cứu:
  - Phát triển EEG BCI thụ động để theo dõi và phân loại ba trạng thái tinh thần: chú ý thụ động, thảnh thơi, và buồn ngủ.
  - Sử dụng mô hình SVM để đạt độ chính xác cao trong phân biệt trạng thái tinh thần, với kết quả tốt nhất là 86,78% và trung bình là 85.78%.
- Ứng dụng tiềm năng:
  - An toàn người lái xe: Sử dụng để phát hiện trạng thái không tập trung hoặc buồn ngủ nhằm cảnh báo kịp thời.
  - Ứng dụng lâm sàng:
    - Đánh giá hoặc theo dõi trạng thái tinh thần của bệnh nhân.
    - Mở rộng các phương pháp như chỉ số lưỡng cực (BIS) để theo dõi độ sâu gây mê, dựa trên việc phân tích tín hiệu EEG.
  - Khái quát hóa: Nghiên cứu cung cấp cơ sở để phát triển các hệ thống phát hiện trạng thái tinh thần khác trong nhiều bối cảnh khác nhau, từ bảo mật đến y học.
 - Ý nghĩa đặc biệt:
  - Phương pháp phân tích các tham số từ tín hiệu EEG giúp cung cấp hiểu biết mới về cách biểu thị các trạng thái tinh thần.
  - Hướng đi mới trong việc ứng dụng EEG BCI vào các bài toán thực tiễn với tiềm năng lớn trong cải thiện an toàn và hiệu quả giám sát.

# 7. Tài liệu tham khảo:
- https://www.researchgate.net/publication/333499959_Distinguishing_mental_attention_states_of_humans_via_an_EEG-based_passive_BCI_using_Machine_Learning_Methods

