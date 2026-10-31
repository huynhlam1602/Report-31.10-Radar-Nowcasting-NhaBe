# -*- coding: utf-8 -*-
"""
Đề tài: Phân loại trạng thái mưa và cảnh báo hạn cực ngắn từ radar thời tiết Nhà Bè
Tác giả: Nguyễn Huỳnh Lâm (MSSV: 2411841) - ĐHBK ĐHQG-HCM

Mô tả:
1. Đọc dữ liệu đã tính phản hồi có trọng số hàm mũ của 3 tháng (T6, T7, T8/2025).
2. Thuật toán Time-Shifting chính xác tuyệt đối với pd.merge_asof (dung sai <= 5 phút).
3. Tạo nhãn đa mốc thời gian: Label_0H, Label_1H, Label_2H, Label_3H.
4. Phân chia tập dữ liệu Stratified Split 80% Train : 10% Val : 10% Test.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    csv_dir = 'Data_CSVs'
    files = {
        'T06': os.path.join(csv_dir, 'Ketqua_Nowcasting_Thang_6.csv'),
        'T07': os.path.join(csv_dir, 'Ketqua_Nowcasting_Thang_7.csv'),
        'T08': os.path.join(csv_dir, 'Ketqua_Nowcasting_Thang_8.csv')
    }

    # 1. Thu thập và định dạng thời gian
    df_all = []
    for m, path in files.items():
        if os.path.exists(path):
            print(f"Đang nạp dữ liệu: {path}")
            df_temp = pd.read_csv(path)
            if 'Datetime' not in df_temp.columns:
                df_temp['Datetime'] = pd.to_datetime(
                    df_temp['File_Name'].str.extract(r'NHB(\d{12})')[0], 
                    format='%y%m%d%H%M%S'
                )
            else:
                df_temp['Datetime'] = pd.to_datetime(df_temp['Datetime'])
            df_all.append(df_temp[['File_Name', 'Datetime', 'Average_dBZ', 'Label']])

    if not df_all:
        print("Không tìm thấy các file CSV gốc trong thư mục Data_CSVs. Vui lòng kiểm tra lại đường dẫn.")
        return

    df_full = pd.concat(df_all, ignore_index=True).sort_values('Datetime').reset_index(drop=True)
    print(f"Tổng số lượt quét radar thu thập được: {len(df_full)}")

    # 2. Thuật toán Time-Shifting thời gian thực với pd.merge_asof
    def create_horizon_label(df, hours_offset):
        target_df = df[['Datetime', 'Label']].copy()
        target_df['Lookup_Time'] = target_df['Datetime'] - pd.Timedelta(hours=hours_offset)
        target_df = target_df.sort_values('Lookup_Time')
        
        merged = pd.merge_asof(
            df,
            target_df[['Lookup_Time', 'Label']],
            left_on='Datetime',
            right_on='Lookup_Time',
            direction='nearest',
            tolerance=pd.Timedelta('5min')  # Dung sai thời gian tối đa 5 phút
        )
        return merged['Label_y']

    print("Đang tạo nhãn đa mốc dự báo (0H, 1H, 2H, 3H)...")
    df_full['Label_0H'] = df_full['Label']
    df_full['Label_1H'] = create_horizon_label(df_full, 1)
    df_full['Label_2H'] = create_horizon_label(df_full, 2)
    df_full['Label_3H'] = create_horizon_label(df_full, 3)

    # 3. Lọc bỏ các mẫu khuyết thiếu tương lai ở cuối tháng
    df_clean = df_full.dropna(subset=['Label_1H', 'Label_2H', 'Label_3H']).reset_index(drop=True)
    print(f"Số mẫu hợp lệ đầy đủ 4 mốc thời gian: {len(df_clean)}")

    # 4. Phân tầng dữ liệu Stratified Split 80 : 10 : 10 theo nhãn 2H
    print("Đang phân tầng Stratified Split 80:10:10 theo Label_2H...")
    train_df, temp_df = train_test_split(
        df_clean, test_size=0.2, random_state=42, stratify=df_clean['Label_2H']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['Label_2H']
    )

    print(f"- Tập Train: {len(train_df)} mẫu")
    print(f"- Tập Val:   {len(val_df)} mẫu")
    print(f"- Tập Test:  {len(test_df)} mẫu")

    # 5. Xuất kết quả
    train_df.to_csv('Train_Set.csv', index=False)
    val_df.to_csv('Val_Set.csv', index=False)
    test_df.to_csv('Test_Set.csv', index=False)
    print("Đã lưu thành công: Train_Set.csv, Val_Set.csv, Test_Set.csv")

if __name__ == '__main__':
    main()
