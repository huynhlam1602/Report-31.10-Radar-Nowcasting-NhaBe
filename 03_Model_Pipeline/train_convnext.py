# -*- coding: utf-8 -*-
"""
Huấn luyện mạng ConvNeXt-Base (ImageNet-22K) với Focal Loss
Hỗ trợ Mixed Precision FP16, AdamW, Cosine Annealing LR Scheduler
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
from sklearn.metrics import accuracy_score, f1_score
from radar_dataset import RadarNowcastingDataset, get_transforms
from focal_loss import FocalLossWithSmoothing

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return val_loss, val_acc, val_macro_f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Thiết bị huấn luyện: {device}")
    
    # 1. Chuẩn bị DataLoader
    train_transform, val_transform = get_transforms()
    train_dataset = RadarNowcastingDataset('Train_Set.csv', 'radar_images_224', target_horizon='Label_2H', transform=train_transform)
    val_dataset = RadarNowcastingDataset('Val_Set.csv', 'radar_images_224', target_horizon='Label_2H', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Khởi tạo ConvNeXt-Base ImageNet-22K
    print("Khởi tạo ConvNeXt-Base (ImageNet-22K)...")
    model = timm.create_model(
        'convnext_base.fb_in22k_ft_in1k', 
        pretrained=True, 
        in_chans=1, # 1 kênh Grayscale
        num_classes=5,
        drop_path_rate=0.2
    ).to(device)
    
    criterion = FocalLossWithSmoothing(gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    scaler = GradScaler()
    
    best_f1 = 0.0
    num_epochs = 15
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_convnext_2h.pth')
            print(f"  -> Lưu Checkpoint tốt nhất: Val F1 = {best_f1*100:.2f}%")

if __name__ == '__main__':
    main()
