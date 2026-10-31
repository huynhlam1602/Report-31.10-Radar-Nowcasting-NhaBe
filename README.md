# Radar Image Classification for Rainfall Nowcasting
## Phân Loại Trạng Thái Mưa và Cảnh Báo Hạn Cực Ngắn Từ Dữ Liệu Radar Thời Tiết Nhà Bè

**Báo cáo Đồ án cơ sở ngành Khoa học Dữ liệu**  
*Khoa Khoa học & Kỹ thuật Máy tính, Trường Đại học Bách Khoa – ĐHQG-HCM*

* **Sinh viên thực hiện:** Nguyễn Huỳnh Lâm (MSSV: 2411841)
* **Giảng viên hướng dẫn:** 
  * GS. Phan Thành An (Trường ĐH Bách Khoa – ĐHQG-HCM)
  * ThS. Nguyễn Văn Gia Thịnh (Trường ĐH Bách Khoa – ĐHQG-HCM)
  * ThS. Lê Đình Quyết (Đài Khí tượng Thủy văn Khu vực Nam Bộ)
* **Bài báo tham chiếu:** Le Hong Trang, Nguyen Hoang Anh Thu, Nguyen Minh Dan, Nguyen Hoang Hoang, Pham Tan An, Phan Thanh Vu (2025). *Radar Image Classification for Rainfall Nowcasting*, SSRN Preprint, doi: 10.2139/ssrn.5139080.

---

## 1. Tóm Tắt Đề Tài

Dự báo mưa hạn cực ngắn (Rainfall Nowcasting, khung thời gian 0–3 giờ) đóng vai trò then chốt trong công tác cảnh báo ngập lụt đô thị và điều tiết giao thông tại khu vực TP. Hồ Chí Minh và Nam Bộ. Các phương pháp hồi quy chuỗi ảnh radar (Image-to-Image regression) thường đối mặt với hiện tượng ảnh dự báo bị mờ nhòe (blurry) và chi phí tính toán lớn.

Nghiên cứu này tiếp cận bài toán theo hướng phân loại ảnh radar (Image Classification):
* **Đầu vào (Input):** Một ảnh radar phản hồi (Reflectivity, dBZ) ở góc quét Sweep 0 tại thời điểm hiện tại $T_0$.
* **Đầu ra (Output):** Trạng thái thời tiết tổng thể sau 2 giờ ($T_0 + 2h$) được phân thành 5 cấp độ khí tượng:
  1. **Clear** ($< 30\text{ dBZ}$): Quang đãng / Không mưa.
  2. **Light rain** ($30 \le \bar{Z} < 40\text{ dBZ}$): Mưa nhỏ lác đác.
  3. **Moderate rain** ($40 \le \bar{Z} < 47.5\text{ dBZ}$): Mưa vừa diện rộng.
  4. **Heavy rain** ($47.5 \le \bar{Z} < 55\text{ dBZ}$): Mưa to (nguy cơ ngập úng đô thị).
  5. **Very heavy rain** ($\bar{Z} \ge 55\text{ dBZ}$): Dông bão / Mưa rất to nguy hiểm.

---

## 2. Phương Pháp Luận & Giải Pháp Kỹ Thuật

### 2.1. Tập Dữ Liệu Thực Nghiệm 3 Tháng Mùa Mưa
Khác với các nghiên cứu thử nghiệm giới hạn trong 1 tháng dễ gây học vẹt (overfitting), nghiên cứu mở rộng toàn diện trên 3 tháng cao điểm mùa mưa Nam Bộ (Tháng 06, 07 và 08 năm 2025) từ gần 30.000 file RAW trạm Radar Nhà Bè, thu được **26.405 lượt quét radar hợp lệ**.

### 2.2. Công Thức Trọng Số Mũ Nghịch Đảo Tần Suất
Để tránh việc hơn 95% diện tích bầu trời quang đãng triệt tiêu tín hiệu của các vùng mây dông nguy hiểm (chiếm 3–5% diện tích quét), nghiên cứu áp dụng công thức hàm mũ nghịch đảo tần suất:

$$w_i = 10^{100 \cdot (1 - p_i)}$$

$$\bar{Z} = \frac{\sum_i N_i \cdot w_i \cdot Z_i}{\sum_i N_i \cdot w_i}$$

Trong đó:
* $p_i$: Tần suất xuất hiện của khoảng phản hồi thứ $i$ (mỗi khoảng cách nhau 5 dBZ).
* $N_i$: Số lượng điểm ảnh (pixel count) trong khoảng $i$.
* $Z_i$: Giá trị dBZ trung bình của khoảng thứ $i$.

### 2.3. Thuật Toán Time-Shifting & Stratified Split 80:10:10
* Chu kỳ quét thực tế dao động 3–7–3 phút và có các khoảng gián đoạn kỹ thuật. Thuật toán ghép dòng thời gian thực (`pd.merge_asof`) được áp dụng với dung sai $\le 5$ phút để xác định chính xác trạng thái sau 2 giờ ($T+2h$).
* Tỷ lệ mất cân bằng giữa Mưa nhỏ (13.319 mẫu) và Bão lớn (90 mẫu) lên tới 1:148. Kỹ thuật phân tầng `StratifiedShuffleSplit` (80% Train : 10% Validation : 10% Test) được sử dụng nhằm bảo toàn tỷ lệ 90 mẫu bão lớn đồng đều ở cả 3 tập (Train: 72, Val: 9, Test: 9).

### 2.4. Kiến Trúc Mạng ConvNeXt-Base & Focal Loss
* **Mô hình học sâu:** ConvNeXt-Base kết hợp ưu thế kernel lớn ($7 \times 7$), Inverted Bottleneck, GELU, LayerNorm và khởi tạo từ trọng số tiền huấn luyện ImageNet-22K (14 triệu ảnh).
* **Hàm mất mát Focal Loss kết hợp Label Smoothing ($\epsilon = 0.1$):**

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t) \quad \text{với } \gamma = 2.0$$

Hệ số $(1 - p_t)^2$ tự động triệt tiêu gradient của các mẫu dễ đoán (mưa nhỏ), buộc mạng nơ-ron dồn toàn bộ trọng tâm tối ưu vào các ca dông bão hiếm gặp.

---

## 3. Kết Quả Thực Nghiệm Trên Tập Kiểm Thử Độc Lập

Mô hình được đánh giá độc lập trên **2.634 mẫu kiểm thử (Test Set)**:
* **Độ chính xác tổng thể (Test Accuracy):** **76.23%** (Tập Val: 76.91%, độ lệch $< 0.7\%$, hoàn toàn không xảy ra overfitting).
* **Độ nhạy bắt Mưa to (Heavy Rain Recall):** **83.88%** (Precision: 72.36%, F1-Score: 77.69%).
* **Weighted F1-Score:** **75.76%** (Macro F1: 50.23% trên cả 5 lớp).

### Bảng Đánh Giá Chi Tiết Từng Lớp:
| Cấp độ thời tiết | Precision | Recall | F1-Score | Số lượng (Support) |
| :--- | :---: | :---: | :---: | :---: |
| Clear | 0.7692 | 0.1389 | 0.2353 | 72 |
| Light rain | 0.8660 | 0.8098 | 0.8370 | 1.325 |
| Moderate rain | 0.6314 | 0.6965 | 0.6624 | 738 |
| **Heavy rain (Mưa to)** | **0.7236** | **0.8388** | **0.7769** | **490** |
| Very heavy rain | 0.0000 | 0.0000 | 0.0000 | 9 |
| **Accuracy Tổng thể** | \multicolumn{3}{c}{\textbf{0.7623}} | **2.634** |
| **Macro Average** | 0.5981 | 0.4968 | **0.5023** | 2.634 |
| **Weighted Average** | **0.7682** | **0.7623** | **0.7576** | **2.634** |

### Phân Tích Ma Trận Nhầm Lẫn (Confusion Matrix):
* **100% mẫu bão lớn (9/9 mẫu Very heavy rain)** được mô hình dự báo rơi vào lớp liền kề là Heavy rain (Mưa to). Trong nghiệp vụ phòng chống thiên tai, cảnh báo mưa to vẫn đảm bảo kích hoạt đầy đủ hệ thống bơm tiêu thoát nước và cảnh báo người dân.
* **Tính liên tục vật lý:** Toàn bộ sai số phân loại chỉ xuất hiện ở các ô sát đường chéo chính giữa hai cấp mưa liền kề (Mưa vừa $\leftrightarrow$ Mưa to, Mưa nhỏ $\leftrightarrow$ Quang đãng), không có sai số nhảy vọt cực đoan.

---

## 4. Cấu Trúc Mã Nguồn

```text
Radar_Nowcasting_Repo/
├── 01_Preprocessing_and_Labeling/          # Trích xuất dBZ & Gán nhãn 3 tháng
│   ├── extract_and_label_thang6.py         # Tiền xử lý Tháng 06/2025
│   ├── extract_and_label_thang7.py         # Tiền xử lý Tháng 07/2025
│   └── extract_and_label_thang8.py         # Tiền xử lý Tháng 08/2025
├── 02_Dataset_Preparation/                 # Chuẩn hóa & Phân tầng
│   ├── standardize_and_stratified_split.py # Time-shifting merge_asof & Stratified Split
│   └── export_radar_images_224.py          # Trích xuất ảnh PPI 224x224 đa tiến trình
├── 03_Model_Pipeline/                      # Pipeline PyTorch huấn luyện & kiểm thử
│   ├── radar_dataset.py                    # Dataset & chuỗi Data Augmentation
│   ├── focal_loss.py                       # Hàm Focal Loss with Label Smoothing
│   ├── train_convnext.py                   # Vòng lặp huấn luyện FP16 AMP + AdamW
│   └── evaluate_test_set.py                # Đánh giá độc lập Test Set & vẽ Heatmap
├── assets/                                 # Hình ảnh minh họa & biểu đồ
│   ├── radar_clean.png
│   ├── radar_224x224.png
│   ├── confusion_matrix_2h.png
│   ├── formula_weight.png
│   └── formula_focal_loss.png
├── requirements.txt                        # Danh sách thư viện phụ thuộc
├── .gitignore
└── README.md                               # Báo cáo tổng quan dự án
```

---

## 5. Hướng Dẫn Thực Thi

### Cài đặt môi trường
```bash
git clone https://github.com/<username>/Radar-Nowcasting-NhaBe.git
cd Radar-Nowcasting-NhaBe
pip install -r requirements.txt
```

### Tiền xử lý & Chia tập dữ liệu
```bash
python 02_Dataset_Preparation/standardize_and_stratified_split.py
python 02_Dataset_Preparation/export_radar_images_224.py
```

### Huấn luyện mô hình ConvNeXt-Base
```bash
python 03_Model_Pipeline/train_convnext.py
```

### Đánh giá kiểm thử độc lập
```bash
python 03_Model_Pipeline/evaluate_test_set.py
```

---

## 6. Tài Liệu Tham Khảo

1. L. H. Trang, N. H. A. Thu, N. M. Dan, N. H. Hoang, P. T. An, and P. T. Vu, "Radar image classification for rainfall nowcasting," *SSRN Preprint*, 2025. doi: 10.2139/ssrn.5139080.
2. Z. Liu, H. Mao, C. Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A convnet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.
3. T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.
