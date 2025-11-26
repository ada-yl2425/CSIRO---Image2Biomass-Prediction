"""teacher_model.ipynb
## 多模态架构：

1. Image Branch: 使用一个轻量级的预训练模型（EfficientNet-B1），并冻结大部分层。对于小数据，B1 比 B3/B4 更安全。保留GlobalAveragePooling

2. Table Branch:

 * 分类特征 (State_encoded, Species_encoded): 使用 Embedding Layers。这是处理LabelEncoder输出的正确方式，它比独热编码更节省参数，并且能学到类别间的语义关系。

 * 数值特征 (Pre_GSHH_NDVI, Height_Ave_cm, month_sin, month_cos): 直接输入。

3. Fusion: 将两个分支的输出Concatenate，并通过MLP来得到最终的5个值。
"""

# KnowledgeDistillation/teacher_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AttentionPool(nn.Module):
    """
    一个简单的空间注意力池化层。
    
    输入: [B, C, H, W] (例如 [16, 1280, 8, 8])
    输出: [B, C] (例如 [16, 1280])
    """
    def __init__(self, in_dim):
        super(AttentionPool, self).__init__()
        
        # 这个 1x1 卷积会学习一个“重要性”图
        # 它将 C 个通道压缩为 1 个通道的“注意力图”
        self.conv = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=2) # 展平 H, W
        self.softmax = nn.Softmax(dim=2) # 在 H*W 维度上 Softmax

    def forward(self, x):
        # x shape: [B, C, H, W]
        
        # 1. 生成注意力图
        # attn_map shape: [B, 1, H, W]
        attn_map = self.conv(x)
        
        # 2. 展平并 Softmax
        # flat_map shape: [B, 1, H*W]
        flat_map = self.flatten(attn_map)
        # attn_weights shape: [B, 1, H*W] (所有 H*W 位置的权重和为 1)
        attn_weights = self.softmax(flat_map)
        
        # 3. 展平原始特征
        # x_flat shape: [B, C, H*W]
        x_flat = self.flatten(x)
        
        # 4. 加权求和
        # ( [B, C, H*W] * [B, 1, H*W] ) -> 广播乘法
        # torch.sum(dim=2) -> 在 H*W 维度上求和
        # final_features shape: [B, C]
        final_features = torch.sum(x_flat * attn_weights, dim=2)
        
        return final_features

class TeacherModel(nn.Module):
    """
    Multi-modal Fusion Model (已更新为使用 Attention Pooling)
    """
    def __init__(self, num_states, num_species, img_model_name='efficientnet_b1'):
        super(TeacherModel, self).__init__()

        # --- 1. 图像分支 (Image Branch) ---
        
        # [关键修改] global_pool='' 移除了自动的 GAP
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''  # <--- 移除 GAP，获得 [B, C, H, W]
        )
        self.num_img_features = self.img_backbone.num_features # 1280

        # [新] 使用我们自定义的 AttentionPool
        self.img_pool = AttentionPool(in_dim=self.num_img_features)
        
        # [新] 图像投影层 (来自我们上一步的讨论，保持不变)
        self.img_projector = nn.Sequential(
            nn.Linear(self.num_img_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128) # 投影到 128 维
        )
        
        # --- 2. 表格分支 (Table Branch) ---
        # ... (这部分完全不变) ...
        self.num_numeric_features = 4
        self.state_embed_dim = 8
        self.species_embed_dim = 16
        self.state_embedding = nn.Embedding(num_states, self.state_embed_dim)
        self.species_embedding = nn.Embedding(num_species, self.species_embed_dim)
        self.tab_input_dim = (
            self.num_numeric_features + self.state_embed_dim + self.species_embed_dim
        )
        self.tab_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5), 
            nn.Linear(64, 128),
        )

        # --- 3. 融合头 (Fusion Head) ---
        # ... (这部分完全不变, 依然是 128 + 128) ...
        self.fusion_input_dim = 128 + 128 # = 256
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

        # --- 冻结 ... (冻结逻辑保持不变) ---
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        
        # 只解冻最后一个 block (来自 Run 2)
        for param in self.img_backbone.blocks[-1:].parameters():
            param.requires_grad = True

        # (解冻 conv_head 和 bn2 的代码保持不变)
        if hasattr(self.img_backbone, 'conv_head'):
             for param in self.img_backbone.conv_head.parameters():
                  param.requires_grad = True
        if hasattr(self.img_backbone, 'bn2'):
             for param in self.img_backbone.bn2.parameters():
                  param.requires_grad = True

    def forward(self, image, numeric_data, categorical_data):
        
        # 1. 处理图像
        # x shape: [B, 1280, H, W] (例如 [16, 1280, 8, 8])
        x = self.img_backbone(image)
        
        # [新] 应用注意力池化
        # img_features shape: [B, 1280]
        img_features = self.img_pool(x)
        
        # [改] 投影池化后的特征
        img_features_projected = self.img_projector(img_features) # [B, 128]

        # 2. 处理表格 (不变)
        state_idx = categorical_data[:, 0]
        species_idx = categorical_data[:, 1]
        state_emb = self.state_embedding(state_idx)
        species_emb = self.species_embedding(species_idx)
        tab_data = torch.cat([numeric_data, state_emb, species_emb], dim=1)
        tab_features = self.tab_mlp(tab_data) # [B, 128]

        # 3. 融合 (不变)
        fused_features = torch.cat([img_features_projected, tab_features], dim=1) # [B, 256]

        # 4. 预测 (不变)
        output = self.fusion_head(fused_features)
        
        return output