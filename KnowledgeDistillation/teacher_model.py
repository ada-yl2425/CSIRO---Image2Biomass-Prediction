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
    (已更新为使用 Temperature Scaling)
    """
    def __init__(self, in_dim, temperature=2.0): # <-- 1. 添加 temperature
        super(AttentionPool, self).__init__()
        self.conv = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=2) 
        
        self.temperature = temperature # <-- 2. 存储 T

        # [修改] 我们将在 forward 中应用 T，所以这里移除 softmax
        # self.softmax = nn.Softmax(dim=2) 

    def forward(self, x):
        # x shape: [B, C, H, W]
        
        # 1. 生成注意力图
        attn_map = self.conv(x) # [B, 1, H, W]
        
        # 2. 展平
        flat_map = self.flatten(attn_map) # [B, 1, H*W]
        
        # 3. [修改] 应用 Temperature 并进行 Softmax
        #    T > 1.0 会“软化”分布，迫使模型关注更广泛的区域
        #    T = 1.0 等同于标准 softmax
        attn_weights = F.softmax(flat_map / self.temperature, dim=2) # [B, 1, H*W]
        
        # 4. 展平原始特征
        x_flat = self.flatten(x) # [B, C, H*W]
        
        # 5. 加权求和
        final_features = torch.sum(x_flat * attn_weights, dim=2) # [B, C]
        
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
        self.img_pool = AttentionPool(in_dim=self.num_img_features, temperature=2.0)
        
        # [新] 图像投影层 (来自我们上一步的讨论，保持不变)
        self.img_projector = nn.Sequential(
            nn.Linear(self.num_img_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
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
        # [新架构] 增强的表格 MLP (Input -> 128 -> 128)
        self.tab_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, 128),  # <-- 更改
            nn.ReLU(),
            nn.BatchNorm1d(128),                 # <-- 更改
            nn.Dropout(0.5), 
            nn.Linear(128, 128)                  # <-- 更改
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

        # --- [关键修改] 冻结 ---
        
        # 1. 彻底冻结所有主干层 (这行保留)
        for param in self.img_backbone.parameters():
            param.requires_grad = False
            
        # 2. [删除] 我们不再解冻任何 block
        # for param in self.img_backbone.blocks[-1:].parameters():
        #     param.requires_grad = True

        # 3. [删除] 我们也不再解冻 conv_head
        # if hasattr(self.img_backbone, 'conv_head'):
        #      for param in self.img_backbone.conv_head.parameters():
        #           param.requires_grad = True
        
        # 4. [删除] 我们也不再解冻 bn2
        # if hasattr(self.img_backbone, 'bn2'):
        #      for param in self.img_backbone.bn2.parameters():
        #           param.requires_grad = True

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