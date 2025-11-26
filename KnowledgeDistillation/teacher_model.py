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
import timm # PyTorch Image Models (需要: pip install timm)

class TeacherModel(nn.Module):
    def __init__(self, num_states, num_species, img_model_name='efficientnet_b1'):
        super(TeacherModel, self).__init__()

        # --- 1. 图像分支 (Image Branch) ---
        self.img_backbone = timm.create_model(
            img_model_name, pretrained=True, num_classes=0
        )
        self.num_img_features = self.img_backbone.num_features

        # [新] 添加一个可训练的图像投影层
        # (1280 -> 128)
        # 这个层是可训练的, 它学会如何从1280个特征中“提取”有用的信息
        self.img_projector = nn.Sequential(
            nn.Linear(self.num_img_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128) # 最终投影到 128 维
        )
        
        # --- 2. 表格分支 (Table Branch) ---
        # ... (表格 Embedding 定义不变) ...
        self.state_embed_dim = 8
        self.species_embed_dim = 16 
        self.state_embedding = nn.Embedding(num_states, self.state_embed_dim)
        self.species_embedding = nn.Embedding(num_species, self.species_embed_dim)
        self.num_numeric_features = 4
        self.tab_input_dim = (
            self.num_numeric_features + self.state_embed_dim + self.species_embed_dim
        )
        
        # ... (表格 MLP 定义不变) ...
        self.tab_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5), # 保持 0.5
            nn.Linear(64, 128),
        )

        # --- 3. 融合头 (Fusion Head) ---
        
        # [改] 新的融合维度
        # (128 来自 img_projector) + (128 来自 tab_mlp)
        self.fusion_input_dim = 128 + 128 # = 256

        self.fusion_head = nn.Sequential(
            # [改] 输入维度现在是 256
            nn.Linear(self.fusion_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5), # 保持 0.5
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5) # 最终输出5个值
        )

        # --- 冻结 ... (冻结逻辑保持不变) ---
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        
        # 保持 Run 2 的设置: 只解冻最后一个 block
        for param in self.img_backbone.blocks[-1:].parameters():
            param.requires_grad = True

        if hasattr(self.img_backbone, 'conv_head'):
            for param in self.img_backbone.conv_head.parameters():
                 param.requires_grad = True
        if hasattr(self.img_backbone, 'bn2'):
            for param in self.img_backbone.bn2.parameters():
                 param.requires_grad = True


    def forward(self, image, numeric_data, categorical_data):
        """
        image: shape [batch_size, 3, H, W]
        numeric_data: shape [batch_size, 4] (numeric features)
        categorical_data: shape [batch_size, 2] (State_encoded, Species_encoded)
        """

        # 1. 处理图像
        img_features = self.img_backbone(image) # [B, 1280]
        
        # [新] 将图像特征投影到 128 维
        img_features_projected = self.img_projector(img_features) # [B, 128]

        # 2. 处理表格
        # ... (表格处理) ...
        state_idx = categorical_data[:, 0]
        species_idx = categorical_data[:, 1]
        state_emb = self.state_embedding(state_idx)
        species_emb = self.species_embedding(species_idx)
        tab_data = torch.cat([numeric_data, state_emb, species_emb], dim=1)
        
        tab_features = self.tab_mlp(tab_data) # [B, 128]

        # 3. 融合
        # [改] 融合两个 128 维的向量
        fused_features = torch.cat([img_features_projected, tab_features], dim=1) # [B, 256]

        # 4. 预测
        output = self.fusion_head(fused_features) # 融合头现在接收 [B, 256]
        return output