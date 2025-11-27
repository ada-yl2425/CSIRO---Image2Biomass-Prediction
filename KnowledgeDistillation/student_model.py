# KnowledgeDistillation/student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -----------------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, in_dim, temperature=2.0):
        super(AttentionPool, self).__init__()
        self.conv = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=2) 
        self.temperature = temperature

    def forward(self, x):
        # x shape: [B, C, H, W]
        attn_map = self.conv(x) # [B, 1, H, W]
        flat_map = self.flatten(attn_map) # [B, 1, H*W]
        attn_weights = F.softmax(flat_map / self.temperature, dim=2) # [B, 1, H*W]
        x_flat = self.flatten(x) # [B, C, H*W]
        final_features = torch.sum(x_flat * attn_weights, dim=2) # [B, C]
        return final_features
# -----------------------------------------------------------------

class StudentModel(nn.Module):
    """
    纯图像到生物量的学生模型 (Student Model)
    (已更新为使用 Self-Attention Pooling)
    """
    def __init__(self, img_model_name='efficientnet_b2'):
        """
        Args:
            img_model_name (str): 使用的 timm 图像模型 (应与教师匹配)
        """
        super(StudentModel, self).__init__()

        # --- 1. 图像主干 (Image Backbone) ---
        
        # [关键修改] global_pool=''
        # 我们移除 GAP，以获取 [B, C, H, W] 的特征图
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0,      # 移除最终的分类层
            global_pool=''      # [修改] 移除 GAP
        )

        self.num_img_features = self.img_backbone.num_features # B2 是 1408

        # [新] 自注意力池化
        # Student 学会自己“看”哪里是重要的
        # 我们使用 T=2.0 来正则化它，防止它过拟合
        self.img_pool = AttentionPool(
            in_dim=self.num_img_features, 
            temperature=2.0
        )

        # --- 2. 预测头 (Prediction Head) ---
        
        # 预测头保持不变 (1408 -> 512 -> 128 -> 5)
        # 它现在接收的是“加权平均”后的特征，而不是“盲目平均”的
        self.prediction_head = nn.Sequential(
            nn.Linear(self.num_img_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 5) # 最终输出5个值
        )

    def forward(self, image):
        """
        前向传播
        Args:
            image: shape [B, 3, 260, 260]
        """

        # 1. 提取特征图
        # image [B, 3, 260, 260] -> x_map [B, 1408, 8, 8] (假设)
        x_map = self.img_backbone(image)

        # 2. 应用自注意力池化
        # x_map [B, 1408, 8, 8] -> img_features [B, 1408]
        img_features = self.img_pool(x_map)

        # 3. 通过预测头进行回归
        # img_features [B, 1408] -> output [B, 5]
        output = self.prediction_head(img_features)

        return output
