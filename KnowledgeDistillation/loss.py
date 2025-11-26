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