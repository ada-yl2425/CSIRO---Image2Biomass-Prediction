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
    """
    Multi-modal Fusion Model (Images + Tabels)

    Args:
        num_states (int): "State" 特征的唯一类别数
        num_species (int): "Species" 特征的唯一类别数
        img_model_name (str): 使用的 timm 图像模型
    """
    def __init__(self, num_states, num_species, img_model_name='efficientnet_b1'):
        super(TeacherModel, self).__init__()

        # --- 1. 图像分支 (Image Branch) ---
        # 加载预训练模型，不包括最后的分类器 (num_classes=0)
        # GlobalAveragePooling 会被自动应用
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0
        )

        # 获取图像模型的输出特征维度
        self.num_img_features = self.img_backbone.num_features

        # --- 2. 表格分支 (Table Branch) ---

        # 数值特征 (Pre_GSHH_NDVI, Height_Ave_cm, month_sin, month_cos)
        self.num_numeric_features = 4

        # 类别特征的 Embedding 层
        self.state_embed_dim = 8  # 超参数
        self.species_embed_dim = 16 # 超参数

        self.state_embedding = nn.Embedding(num_states, self.state_embed_dim)
        self.species_embedding = nn.Embedding(num_species, self.species_embed_dim)

        # 表格分支 MLP
        self.tab_input_dim = (
            self.num_numeric_features + self.state_embed_dim + self.species_embed_dim
        )

        self.tab_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
        )

        # --- 3. 融合头 (Fusion Head) ---
        self.fusion_input_dim = self.num_img_features + 128 # 128 来自 tab_mlp

        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 128), # <-- 从 256 降到 128
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5), # <-- 也许使用 0.5 (配合建议1)
            nn.Linear(128, 64),  # <-- 从 128 降到 64
            nn.ReLU(),
            nn.Linear(64, 5) # <-- 从 128 改为 64
        )

        # ** 冻结图像主干的大部分层 **
        # 这是在小数据集上训练的关键
        for param in self.img_backbone.parameters():
            param.requires_grad = False

        # 仅解冻最后几个 block (例如 EfficientNet-B1 的最后1个)
        # 您可以根据需要调整解冻的层数
        for param in self.img_backbone.blocks[-1:].parameters():
            param.requires_grad = True

        # 解冻主干的 BatchNorm (如果存在) 和头部
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
        # self.img_backbone(image) 的输出已经是 [batch_size, num_img_features]
        # 因为 timm 自动应用了 GlobalAveragePooling
        img_features = self.img_backbone(image)

        # 2. 处理表格
        # 提取类别数据
        state_idx = categorical_data[:, 0]
        species_idx = categorical_data[:, 1]

        # 获取 Embeddings
        state_emb = self.state_embedding(state_idx)
        species_emb = self.species_embedding(species_idx)

        # 拼接所有表格特征
        tab_data = torch.cat([numeric_data, state_emb, species_emb], dim=1)

        # 通过 MLP
        tab_features = self.tab_mlp(tab_data)

        # 3. 融合
        fused_features = torch.cat([img_features, tab_features], dim=1)

        # 4. 预测
        # 最终激活函数为 'linear' (即无激活), 因为是回归任务
        output = self.fusion_head(fused_features)

        return output
