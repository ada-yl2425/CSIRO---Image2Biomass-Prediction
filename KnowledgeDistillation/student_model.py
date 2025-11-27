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
    (已更新：同步了主干冻结、预测头容量 和 Dropout)
    """
    def __init__(self, img_model_name='efficientnet_b2'):
        super(StudentModel, self).__init__()

        # --- 1. 图像主干 (Image Backbone) ---
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''  # 移除 GAP
        )

        self.num_img_features = self.img_backbone.num_features # B2 是 1408

        # [保持] 自注意力池化
        self.img_pool = AttentionPool(
            in_dim=self.num_img_features, 
            temperature=2.0
        )

        # --- 2. 预测头 (Prediction Head) ---
        
        # [修改] 提升预测头容量 (1408 -> 512 -> 256 -> 5)
        # [修改] 同步 Dropout 率为 0.3
        self.prediction_head = nn.Sequential(
            nn.Linear(self.num_img_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3), # [修改] 0.5 -> 0.3
            
            nn.Linear(512, 256), # [修改] 128 -> 256
            nn.BatchNorm1d(256), # [修改] 128 -> 256
            nn.ReLU(),
            nn.Dropout(0.3), # [保持] 0.3
            
            nn.Linear(256, 5)  # [修改] 128 -> 256
        )

        # --- [新] 冻结 (与 Teacher 保持一致) ---
        # 1. 彻底冻结主干
        for param in self.img_backbone.parameters():
            param.requires_grad = False
            
        # 2. [修改] 解冻最后 3 个 block (与 Teacher 匹配)
        for param in self.img_backbone.blocks[-3:].parameters(): 
            param.requires_grad = True

        # 3. [保持] 解冻 conv_head 和 bn2 (与 Teacher 匹配)
        if hasattr(self.img_backbone, 'conv_head'):
             for param in self.img_backbone.conv_head.parameters():
                  param.requires_grad = True
        if hasattr(self.img_backbone, 'bn2'):
             for param in self.img_backbone.bn2.parameters():
                  param.requires_grad = True

    def forward(self, image):
        # 1. 提取特征图: [B, 3, H, W] -> [B, 1408, H_map, W_map]
        x_map = self.img_backbone(image)

        # 2. 应用自注意力池化: [B, 1408, H_map, W_map] -> [B, 1408]
        img_features = self.img_pool(x_map)

        # 3. 通过预测头进行回归: [B, 1408] -> [B, 5]
        output = self.prediction_head(img_features)

        return output
