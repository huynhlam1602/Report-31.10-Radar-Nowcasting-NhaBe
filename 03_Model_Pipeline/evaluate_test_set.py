# -*- coding: utf-8 -*-
"""
Đánh giá độc lập trên tập Kiểm thử (Test Set)
Tính Precision, Recall, F1 từng lớp & Vẽ Ma trận nhầm lẫn
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from radar_dataset import RadarNowcastingDataset, get_transforms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Thiết bị kiểm thử: {device}")
    
    # 1. DataLoader tập Test
    _, val_transform = get_transforms()
    test_dataset = RadarNowcastingDataset('Test_Set.csv', 'radar_images_224', target_horizon='Label_2H', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 2. Nạp Checkpoint tốt nhất
    model = timm.create_model(
        'convnext_base.fb_in22k_ft_in1k', 
        pretrained=False, 
        in_chans=1, 
        num_classes=5
    ).to(device)
    
    checkpoint_path = 'best_convnext_2h.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Đã nạp checkpoint:", checkpoint_path)
    else:
        print("Không tìm thấy checkpoint. Đang chạy kiểm thử với trọng số khởi tạo ngẫu nhiên.")
        
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            with autocast():
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    # 3. Báo cáo đánh giá chi tiết
    target_names = ["Clear", "Light rain", "Moderate rain", "Heavy rain", "Very heavy rain"]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    print("\n" + "="*60)
    print("BẢNG ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET - 2.634 MẪU)")
    print("="*60)
    print(report)
    
    # 4. Vẽ Ma trận nhầm lẫn chuẩn hóa (Recall-normalized)
    cm = confusion_matrix(all_targets, all_preds, normalize='true')
    plt.figure(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm * 100, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=target_names, 
        yticklabels=target_names,
        cbar_kws={'label': 'Tỷ lệ % chuẩn hóa theo hàng'}
    )
    plt.title("Normalized Confusion Matrix - ConvNeXt-B (Label_2H)\nNhà Bè Radar Dataset (3 Tháng)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Nhãn Dự Đoán (Predicted Label)", fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel("Nhãn Thực Tế Sau 2 Giờ (True Label)", fontsize=11, fontweight='bold', labelpad=8)
    plt.tight_layout()
    plt.savefig('confusion_matrix_2h.png')
    print("Đã lưu biểu đồ ma trận nhầm lẫn: confusion_matrix_2h.png")

if __name__ == '__main__':
    main()
