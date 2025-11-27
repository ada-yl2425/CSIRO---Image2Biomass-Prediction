# KnowledgeDistillation/teacher_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# 移除了 AttentionPool，因为我们将使用 nn.MultiheadAttention
# class AttentionPool(nn.Module): ...

class TeacherModel(nn.Module):
    """
    Multi-modal Fusion Model (已更新为使用 Cross-Attention)
    """
    def __init__(self, num_states, num_species, img_model_name='efficientnet_b2'):
        super(TeacherModel, self).__init__()
        
        self.img_model_dim = 1408  # EfficientNet-B2 的特征维度
        self.tab_model_dim = 128   # 表格分支的输出维度
        self.num_heads = 4         # 交叉注意力的头数

        # --- 1. 图像分支 (Image Branch) ---
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''  # 移除 GAP
        )
        # [修改 3] (动态变化，自动适应)
        # 这一行会自动使用 self.img_model_dim (1408)
        # 所以它不需要改动，但它的输入维度已经变了
        self.img_kv_projector = nn.Linear(self.img_model_dim, self.tab_model_dim)
        
        # --- 2. 表格分支 (Table Branch) ---
        self.num_numeric_features = 4
        self.state_embed_dim = 8
        self.species_embed_dim = 16
        self.state_embedding = nn.Embedding(num_states, self.state_embed_dim)
        self.species_embedding = nn.Embedding(num_species, self.species_embed_dim)
        self.tab_input_dim = (
            self.num_numeric_features + self.state_embed_dim + self.species_embed_dim
        )
        
        # 保持增强的表格 MLP
        self.tab_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5), 
            nn.Linear(128, self.tab_model_dim) # 输出 128 维
        )

        # --- 3. 融合机制 (Cross-Attention) ---
        
        # [新] Query 投影层 (可选，但推荐)
        self.tab_q_projector = nn.Linear(self.tab_model_dim, self.tab_model_dim)
        
        # [新] 交叉注意力模块
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.tab_model_dim, # 128
            num_heads=self.num_heads,
            batch_first=True  # 接受 [Batch, Seq, Features] 格式
        )
        
        # 归一化层
        self.attn_norm = nn.LayerNorm(self.tab_model_dim)

        # --- 4. 最终预测头 (Final Head) ---
        # 融合“表格原始信息”和“被表格过滤后的图像信息”
        self.fusion_input_dim = self.tab_model_dim + self.tab_model_dim # 128 + 128 = 256
        
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

        # --- 冻结 ---
        # 1. 彻底冻结主干
        for param in self.img_backbone.parameters():
            param.requires_grad = False
            
        # 2. [保持] 解冻最后一个 block
        for param in self.img_backbone.blocks[-1:].parameters():
            param.requires_grad = True

        # 3. [保持] 解冻 conv_head 和 bn2
        if hasattr(self.img_backbone, 'conv_head'):
             for param in self.img_backbone.conv_head.parameters():
                  param.requires_grad = True
        if hasattr(self.img_backbone, 'bn2'):
             for param in self.img_backbone.bn2.parameters():
                  param.requires_grad = True

    def forward(self, image, numeric_data, categorical_data):
        
        # 1. 处理表格 (生成 Query)
        state_idx = categorical_data[:, 0]
        species_idx = categorical_data[:, 1]
        state_emb = self.state_embedding(state_idx)
        species_emb = self.species_embedding(species_idx)
        tab_data = torch.cat([numeric_data, state_emb, species_emb], dim=1)
        
        # tab_features shape: [B, 128]
        tab_features = self.tab_mlp(tab_data)
        
        # Q shape: [B, 1, 128] (1 代表“一个问题”)
        query = self.tab_q_projector(tab_features).unsqueeze(1) 

        # 2. 处理图像 (生成 Key 和 Value)
        # x_map shape: [B, 1280, H, W] (例如 [B, 1280, 8, 8])
        x_map = self.img_backbone(image)
        
        B, C, H, W = x_map.shape
        
        # x_patches shape: [B, H*W, C] (例如 [B, 64, 1280])
        x_patches = x_map.flatten(2).permute(0, 2, 1) 
        
        # K/V shape: [B, 64, 128] (投影到 128 维)
        key_value = self.img_kv_projector(x_patches)
        
        # 3. 执行交叉注意力
        # Q = [B, 1, 128] (来自表格)
        # K = [B, 64, 128] (来自图像)
        # V = [B, 64, 128] (来自图像)
        # attn_output shape: [B, 1, 128]
        attn_output, _ = self.cross_attn(
            query=query, 
            key=key_value, 
            value=key_value
        )
        
        # attended_img_features shape: [B, 128]
        attended_img_features = self.attn_norm(attn_output.squeeze(1))

        # 4. 融合与预测
        # [B, 128] (表格信息) + [B, 128] (表格"看到"的图像信息)
        fused_features = torch.cat([tab_features, attended_img_features], dim=1) # [B, 256]
        
        output = self.fusion_head(fused_features)
        
        return output
