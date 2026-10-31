# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:21:21 2026

@author: Admin
"""

import os
import pyart
import numpy as np
import pandas as pd
import warnings

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# 1. CẤU HÌNH ĐƯỜNG DẪN 
folder_path = r'D:\Rainfall Nowcasting Data\2025 Pro-Raw T08\2025 Pro-Raw T08'  
results = []

# 2. QUÉT XUYÊN THƯ MỤC CON BẰNG OS.WALK
file_data_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.startswith('NHB'):
            # Lưu lại cả thư mục chứa file và tên file
            file_data_list.append((root, file))

print(f"Bắt đầu quét {len(file_data_list)} file từ tất cả các ngày. Vui lòng đợi...")

# 3. VÒNG LẶP TÍNH TOÁN
for count, (root, file_name) in enumerate(file_data_list, 1):
    file_path = os.path.join(root, file_name)
    
    try:
        radar = pyart.io.read(file_path)
        sweep_slice = radar.get_slice(0)
        matrix = radar.fields['reflectivity']['data'][sweep_slice]
        
        # --- BƯỚC LỌC CHUẨN KỸ THUẬT PY-ART ---
        if np.ma.is_masked(matrix):
            valid_dbz = matrix[~matrix.mask].data if hasattr(matrix[~matrix.mask], 'data') else matrix[~matrix.mask]
        else:
            valid_dbz = matrix.flatten()
            
        valid_dbz = valid_dbz[~np.isnan(valid_dbz)]
        
        # --- CHỐT CHẶN AN TOÀN CHO NGÀY TRỜI QUANG (CLEAR) ---
        if len(valid_dbz) == 0:
            final_dbz = 0.0
        else:
            bins = [-np.inf] + list(range(0, 65, 5)) + [np.inf]
            counts, _ = np.histogram(valid_dbz, bins=bins)
            total_points = len(valid_dbz)
            ratios = counts / total_points
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for i in range(len(counts)):
                if counts[i] > 0:
                    dbz_in_bin = valid_dbz[(valid_dbz >= bins[i]) & (valid_dbz < bins[i+1])]
                    bin_mean = np.mean(dbz_in_bin)
                    ratio = ratios[i] 
                    weight = 10 ** (100 * (1 - ratio))
                    weighted_sum += counts[i] * weight * bin_mean
                    total_weight += counts[i] * weight
            
            final_dbz = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Gán nhãn
        if final_dbz < 30: label = "Clear"
        elif 30 <= final_dbz < 40: label = "Light rain"
        elif 40 <= final_dbz < 47.5: label = "Moderate rain"
        elif 47.5 <= final_dbz < 55: label = "Heavy rain"
        else: label = "Very heavy rain"
            
        results.append({
            'File_Name': file_name,
            'Average_dBZ': round(final_dbz, 2),
            'Label': label
        })
        
        if count % 500 == 0:
            print(f"> Đã xử lý {count}/{len(file_data_list)} file...")
            
    except Exception:
        pass

# 4. XỬ LÝ TRỤC THỜI GIAN VÀ NOWCASTING
print("\nĐang nối liền trục thời gian và gán nhãn tương lai...")
df = pd.DataFrame(results)

df['Timestamp_Str'] = df['File_Name'].str.extract(r'NHB(\d{12})')
df['Datetime'] = pd.to_datetime(df['Timestamp_Str'], format='%y%m%d%H%M%S')
df = df.sort_values(by='Datetime').reset_index(drop=True)

shift_steps = 12 
df['Label_1_Hour_Later'] = df['Label'].shift(-shift_steps)
df['Label_1_Hour_Later'] = df['Label_1_Hour_Later'].fillna('No_Future_Data')

final_df = df[['File_Name', 'Datetime', 'Average_dBZ', 'Label', 'Label_1_Hour_Later']]

output_file = os.path.join(folder_path, 'Ketqua_Nowcasting_Thang_8.csv')
final_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n🎉 HOÀN THÀNH XUẤT SẮC! Đã lưu kết quả tại:\n{output_file}")