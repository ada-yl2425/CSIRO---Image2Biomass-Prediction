# KnowledgeDistillation/student_loss.py
import torch
import torch.nn as nn
from .loss import WeightedMSELoss # 导入我们已有的 WMSE

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失 (Knowledge Distillation Loss)
    L_total = α * L_hard + (1-α) * L_distill
    """
    def __init__(self, alpha=0.5):
        """
        Args:
            alpha (float): 平衡因子。
                           alpha=1.0 时, 等同于只训练硬标签。
                           alpha=0.0 时, 等同于只模仿教师。
                           推荐 0.5 (一半一半) 或 0.3 (更相信老师)。
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        
        # 两个损失都使用 WeightedMSELoss, 
        # 因为我们希望在重要目标上 (如 Dry_Total_g) 
        # 既匹配事实，也匹配教师。
        self.hard_loss_fn = WeightedMSELoss()
        self.distill_loss_fn = WeightedMSELoss()

    def forward(self, y_student, y_true, y_teacher):
        """
        y_student: 学生的预测 (log scale), shape [B, 5]
        y_true: 真实目标 (log scale), shape [B, 5]
        y_teacher: 教师的预测 (log scale), shape [B, 5]
        """
        
        # 1. 硬损失 (学生 vs 真实值)
        hard_loss = self.hard_loss_fn(y_student, y_true)
        
        # 2. 蒸馏损失 (学生 vs 教师)
        distill_loss = self.distill_loss_fn(y_student, y_teacher)
        
        # 3. 组合
        total_loss = (self.alpha * hard_loss) + ((1 - self.alpha) * distill_loss)
        
        return total_loss