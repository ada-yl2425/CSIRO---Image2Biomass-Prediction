# KnowledgeDistillation/loss.py

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32)

    def forward(self, y_pred, y_true):
        """
        y_pred: (log scale), shape [batch_size, 5]
        y_true: (log scale), shape [batch_size, 5]
        """
        weights = self.weights.to(y_pred.device)
        squared_errors = (y_pred - y_true) ** 2
        weighted_squared_errors = weights * squared_errors
        return torch.mean(weighted_squared_errors)


def calculate_weighted_r2(y_true, y_pred, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32).to(device)
    ss_res = torch.sum(weights * (y_true - y_pred) ** 2)
    sum_weighted_values = torch.sum(weights * y_true)
    weights_broadcasted = weights.expand_as(y_true)
    sum_of_all_weights = torch.sum(weights_broadcasted)
    y_mean_w = sum_weighted_values / (sum_of_all_weights + 1e-6)
    
    ss_tot = torch.sum(weights * (y_true - y_mean_w) ** 2)
    
    r2 = 1.0 - (ss_res / (ss_tot + 1e-6)) 
    
    return r2.item()


class StudentLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_fn = WeightedMSELoss()
        self.loss_soft_fn = nn.MSELoss()
        self.loss_feat_fn = nn.MSELoss()

    def forward(self, student_output, teacher_output, 
                student_features, teacher_features_expanded, y_true):
        # Loss 1: Hard Loss (学生 vs 真实标签)
        loss_hard = self.loss_fn(student_output, y_true)
        # Loss 2: Soft Loss (学生 vs 教师预测)
        loss_soft = self.loss_soft_fn(student_output, teacher_output)
        # Loss 3: Feature Loss (学生特征 vs 教师特征)
        loss_feat = self.loss_feat_fn(student_features, teacher_features_expanded)

        # 5. 组合总损失
        loss = (self.alpha * loss_hard) + (self.beta * loss_soft) + (self.gamma * loss_feat)
        
        return loss
