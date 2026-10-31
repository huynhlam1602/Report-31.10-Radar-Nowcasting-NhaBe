# -*- coding: utf-8 -*-
"""
Focal Loss with Label Smoothing
Khắc phục mất cân bằng cực đoan (1 : 148 giữa Mưa nhỏ và Bão lớn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        """
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        - gamma = 2.0: triệt tiêu mức phạt của các ca dễ đoán (mưa nhỏ), dồn lực vào ca khó.
        - label_smoothing = 0.1: chống hiện tượng overconfident vào các lớp chiếm đa số.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Tính Cross Entropy với Label Smoothing
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()
