# -*- coding: utf-8 -*-
"""
Script trích xuất toàn bộ ảnh Radar PPI 224x224 chuẩn bài báo SSRN-5139080
Sử dụng đa tiến trình (Multiprocessing) để tận dụng tối đa CPU của máy tính.
"""

import os
import glob
import time
import zipfile
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pandas as pd
import numpy as np

# Tắt cảnh báo backend matplotlib và Py-ART
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')  # Chạy nền không mở cửa sổ GUI, tối ưu tốc độ và giải phóng RAM
import matplotlib.pyplot as plt
import pyart
from PIL import Image

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = r"d:\Rainfall Nowcasting Data"

# Thư mục chứa 3 file CSV tập dữ liệu chuẩn đã sinh từ Phase 0
CSV_DIR = os.path.join(BASE_DIR, r"Đồ án cơ sở_AnGra\Code\Data_CSVs")

# Các thư mục chứa file RAW của 3 tháng
RAW_SEARCH_DIRS = [
    os.path.join(BASE_DIR, "2025 Pro-Raw T06"),
    os.path.join(BASE_DIR, "2025 Pro-Raw T07"),
    os.path.join(BASE_DIR, "2025 Pro-Raw T08"),
]

# Thư mục đích lưu ảnh PNG 224x224
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "radar_images_224")
OUTPUT_ZIP_FILE = os.path.join(BASE_DIR, "radar_images_224.zip")

# ==============================================================================
# HÀM XỬ LÝ TRÍCH XUẤT 1 FILE RAW -> ẢNH 224x224
# ==============================================================================
def process_single_file(args):
    file_name, raw_path, out_path = args
    if os.path.exists(out_path):
        return True  # Đã có ảnh từ trước thì bỏ qua (resume)
    
    try:
        radar = pyart.io.read(raw_path)
        
        # Vẽ ảnh PPI góc quét sweep 0 theo chuẩn bài báo
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_subplot(111)
        display = pyart.graph.RadarDisplay(radar)
        
        display.plot('reflectivity', 0, 
                     vmin=-20, vmax=70, 
                     colorbar_flag=False, 
                     title_flag=False, 
                     ax=ax)
        
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        # Lưu ảnh tạm và đóng fig để giải phóng RAM triệt để
        temp_img_path = out_path + ".tmp.png"
        fig.savefig(temp_img_path, format='png', pad_inches=0, transparent=False)
        plt.close(fig)
        
        # Đảm bảo đúng chuẩn 224x224 pixel
        with Image.open(temp_img_path) as img:
            img_resized = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            img_resized.save(out_path, format='PNG', optimize=True)
            
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
        return True
    except Exception as e:
        return False

# ==============================================================================
# HÀM CHÍNH
# ==============================================================================
def main():
    print("=" * 70)
    print("🌧️ BẮT ĐẦU TRÍCH XUẤT ẢNH RADAR PPI 224x224 (PHASE 1 - OFFLINE PREP)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    # 1. Đọc danh sách file cần trích xuất từ 3 tập train/val/test
    print("\n1. Đang nạp danh sách file từ Train, Val, Test Set...")
    train_df = pd.read_csv(os.path.join(CSV_DIR, "Train_Set.csv"))
    val_df = pd.read_csv(os.path.join(CSV_DIR, "Val_Set.csv"))
    test_df = pd.read_csv(os.path.join(CSV_DIR, "Test_Set.csv"))
    
    all_needed_files = set(train_df['File_Name']).union(set(val_df['File_Name'])).union(set(test_df['File_Name']))
    print(f"-> Tổng số file cần trích xuất ảnh: {len(all_needed_files):,} file")
    
    # 2. Quét tìm đường dẫn thực tế của các file RAW trên ổ cứng
    print("\n2. Đang tìm kiếm vị trí các file RAW trong 3 thư mục tháng 6, 7, 8...")
    raw_map = {}
    for search_dir in RAW_SEARCH_DIRS:
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f in all_needed_files:
                    raw_map[f] = os.path.join(root, f)
                    
    print(f"-> Đã xác định được vị trí của: {len(raw_map):,} / {len(all_needed_files):,} file")
    
    # Chuẩn bị danh sách tham số để chạy song song
    tasks = []
    for fname, raw_path in raw_map.items():
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(OUTPUT_IMG_DIR, out_name)
        tasks.append((fname, raw_path, out_path))
        
    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    print(f"\n3. Bắt đầu trích xuất ảnh đa luồng (Sử dụng {num_cpus} nhân CPU)...")
    print("   (Tiến trình đang chạy ngầm, cậu có thể mở thư mục xem ảnh sinh ra liên tục)")
    
    start_time = time.time()
    success_count = 0
    total_tasks = len(tasks)
    
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        for i, res in enumerate(executor.map(process_single_file, tasks, chunksize=50), 1):
            if res:
                success_count += 1
            if i % 1000 == 0 or i == total_tasks:
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed > 0 else 0
                remain = (total_tasks - i) / speed if speed > 0 else 0
                print(f"   [Tiến độ: {i:,}/{total_tasks:,} ({i/total_tasks*100:.1f}%)] "
                      f"- Tốc độ: {speed:.1f} ảnh/s - Còn lại: ~{remain/60:.1f} phút")
                
    total_time = time.time() - start_time
    print(f"\n✅ Hoàn tất trích xuất {success_count:,} ảnh trong {total_time/60:.1f} phút!")
    print(f"-> Thư mục lưu ảnh: {OUTPUT_IMG_DIR}")
    
    # 3. Nén thành file zip duy nhất để upload lên Google Drive
    print("\n4. Đang nén toàn bộ ảnh thành 1 file ZIP duy nhất (radar_images_224.zip)...")
    img_files = glob.glob(os.path.join(OUTPUT_IMG_DIR, "*.png"))
    
    with zipfile.ZipFile(OUTPUT_ZIP_FILE, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for idx, img_path in enumerate(img_files, 1):
            arcname = os.path.basename(img_path)
            zipf.write(img_path, arcname=arcname)
            if idx % 5000 == 0:
                print(f"   Đã nén {idx:,}/{len(img_files):,} ảnh...")
                
    zip_size_mb = os.path.getsize(OUTPUT_ZIP_FILE) / (1024 * 1024)
    print(f"\n🎉 TẤT CẢ ĐÃ HOÀN TẤT!")
    print(f"-> File ZIP nén sẵn sàng để upload lên Drive: {OUTPUT_ZIP_FILE}")
    print(f"-> Dung lượng file ZIP: ~{zip_size_mb:.1f} MB (Rất nhẹ, tải lên Drive chỉ mất ~1-2 phút!)")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
