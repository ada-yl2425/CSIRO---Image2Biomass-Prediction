# KnowledgeDistillation/loss.py
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    [log_Dry_Green_g, log_Dry_Dead_g, log_Dry_Clover_g, log_GDM_g, log_Dry_Total_g]
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        # weighted: [0.1, 0.1, 0.1, 0.2, 0.5]
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], dtype=torch.float32)

    def forward(self, y_pred, y_true):
        """
        y_pred: (log scale), shape [batch_size, 5]
        y_true: (log scale), shape [batch_size, 5]
        """
        weights = self.weights.to(y_pred.device)

        squared_errors = (y_pred - y_true) ** 2

        # (broadcasts weights from [5] to [batch_size, 5])
        weighted_squared_errors = weights * squared_errors

        return torch.mean(weighted_squared_errors)

class StudentLoss(nn.Module):
    """
    知识蒸馏损失 (Knowledge Distillation Loss)
    
    它将 "Hard Loss" (Student vs. Ground Truth) 和 
    "Soft Loss" (Student vs. Teacher) 结合起来。
    """
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha (float): 平衡因子。
                           Loss = (alpha * Hard_Loss) + ((1 - alpha) * Soft_Loss)
                           
                           一个较小的 alpha (例如 0.1) 意味着
                           模型将 90% 的注意力用于模仿 Teacher，
                           10% 的注意力用于匹配真实标签。
        """
        super(StudentLoss, self.__init__()
        self.alpha = alpha
        
        # 我们对两个损失使用相同的加权 MSE
        self.loss_fn = WeightedMSELoss()

    def forward(self, student_output, teacher_output, y_true):
        """
        计算总的蒸馏损失

        Args:
            student_output: 学生的预测, shape [B, 5]
            teacher_output: 教师的预测, shape [B, 5]
            y_true:         真实标签, shape [B, 5]
        """
        
        # 1. Hard Loss (学生 vs 真实标签)
        loss_hard = self.loss_fn(student_output, y_true)
        
        # 2. Soft Loss (学生 vs 教师)
        loss_soft = self.loss_fn(student_output, teacher_output)
        
        # 3. 结合
        total_loss = (self.alpha * loss_hard) + ((1 - self.alpha) * loss_soft)
        
        return total_loss
