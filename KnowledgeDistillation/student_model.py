# KnowledgeDistillation/student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# [删除] 我们不再需要 AttentionPool
# class AttentionPool(nn.Module): ...

class StudentModel(nn.Module):
    """
    纯图像到生物量的学生模型 (Student Model)
    (已更新为使用 Transformer 风格的 "Query Token Head")
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
        self.embed_dim = 256 # [新] 内部嵌入维度
        self.num_heads = 8   # [新] 注意力头数
        self.num_targets = 5 # [新] 5 个输出目标

        # [新] 1. 将图像块投影到 embed_dim
        self.patch_projector = nn.Linear(self.num_img_features, self.embed_dim)

        # [新] 2. 5 个可学习的 "查询标记" (Query Tokens)
        # 每个 token 学习 "询问" 一个特定的生物量
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_targets, self.embed_dim)
        )

        # [新] 3. 交叉注意力层
        # (与 Teacher 的 cross_attn 相同)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        
        # --- 2. 预测头 (Prediction Head) ---
        
        # [新] 4. 新的预测头
        # 我们有 5 个 [B, 1, 256] 的输出 token
        # 我们为每个 token 应用一个独立的线性层 (256 -> 1)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim), # [新] 在 MLP 之前先归一化
            nn.Linear(self.embed_dim, 1)
        )

        # --- [保持] 冻结策略 (与 Teacher 相同) ---
        for param in self.img_backbone.parameters():
            param.requires_grad = False
            
        # 解冻最后 3 个 block
        for param in self.img_backbone.blocks[-3:].parameters(): 
            param.requires_grad = True

        # 解冻 conv_head 和 bn2
        if hasattr(self.img_backbone, 'conv_head'):
             for param in self.img_backbone.conv_head.parameters():
                  param.requires_grad = True
        if hasattr(self.img_backbone, 'bn2'):
             for param in self.img_backbone.bn2.parameters():
                  param.requires_grad = True

    def forward(self, image):
        """
        前向传播
        Args:
            image: shape [B, 3, 260, 260]
        """
        B = image.shape[0]

        # 1. 提取特征图: [B, 3, 260, 260] -> [B, 1408, 8, 8]
        x_map = self.img_backbone(image)

        # 2. 展平为图像块序列: [B, 1408, 8, 8] -> [B, 64, 1408]
        patches = x_map.flatten(2).permute(0, 2, 1) 
        
        # 3. 投影图像块 (用作 Key 和 Value)
        # patches_kv shape: [B, 64, 256]
        patches_kv = self.patch_projector(patches)

        # 4. 准备查询 (Query)
        # query shape: [B, 5, 256]
        query = self.query_tokens.expand(B, -1, -1)

        # 5. 执行交叉注意力
        # Q = [B, 5, 256] (来自 5 个可学习的 token)
        # K = [B, 64, 256] (来自 64 个图像块)
        # V = [B, 64, 256] (来自 64 个图像块)
        # attn_output shape: [B, 5, 256]
        attn_output, _ = self.cross_attn(
            query=query, 
            key=patches_kv, 
            value=patches_kv
        )
        attn_output = self.attn_norm(attn_output) # (B, 5, 256)

        # 6. 预测
        # 将 (256 -> 1) 的预测头应用到 5 个 token 上
        # output shape: [B, 5, 1]
        output = self.prediction_head(attn_output)
        
        # 7. 压缩维度: [B, 5, 1] -> [B, 5]
        output = output.squeeze(-1)

        return output
