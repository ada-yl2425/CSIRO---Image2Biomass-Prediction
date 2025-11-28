# KnowledgeDistillation/student_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class StudentModel(nn.Module):
    """
    纯图像到生物量的学生模型 (Student Model)
    (已更新为使用完整的 3 层 Transformer 解码器)
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

        self.num_img_features = self.img_backbone.num_features # 1408
        self.embed_dim = 256 # 内部嵌入维度
        self.num_heads = 8   # 注意力头数
        self.num_targets = 5 # 5 个输出目标

        # [保持] 1. 将图像块投影到 embed_dim
        self.patch_projector = nn.Linear(self.num_img_features, self.embed_dim)

        # [保持] 2. 5 个可学习的 "查询标记" (Query Tokens)
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_targets, self.embed_dim)
        )

        # [新] 3. 完整的 Transformer 解码器
        # (替代了旧的 self.cross_attn 和 self.attn_norm)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,         # 256
            nhead=self.num_heads,           # 8
            dim_feedforward=self.embed_dim * 4, # 1024 (标准 MLP 扩展)
            dropout=0.1,                    # 标准 Dropout
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=3  # [新] 我们堆叠了 3 层
        )

        # --- 2. 预测头 (Prediction Head) ---
        
        # [新] 4. 更强大的 MLP 预测头
        # (替代了旧的 Linear(256, 1) 头)
        # 它将被独立应用到 5 个输出 token 上
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2), # 256 -> 512
            nn.ReLU(),
            nn.Dropout(0.3), # [保持] 与 Teacher 一致
            nn.Linear(self.embed_dim * 2, 1) # 512 -> 1
        )

        # --- [保持] 冻结策略 (与 Teacher 相同) ---
        for param in self.img_backbone.parameters():
            param.requires_grad = False
            
        for param in self.img_backbone.blocks[-3:].parameters(): 
            param.requires_grad = True

        if hasattr(self.img_backbone, 'conv_head'):
             for param in self.img_backbone.conv_head.parameters():
                  param.requires_grad = True
        if hasattr(self.img_backbone, 'bn2'):
             for param in self.img_backbone.bn2.parameters():
                  param.requires_grad = True

    def forward(self, image):
        B = image.shape[0]

        # 1. 提取特征图: [B, 1408, 8, 8]
        x_map = self.img_backbone(image)

        # 2. 展平为图像块序列: [B, 64, 1408]
        patches = x_map.flatten(2).permute(0, 2, 1) 
        
        # 3. 投影图像块 (用作 Key/Value, 也称为 "memory")
        # memory shape: [B, 64, 256]
        memory = self.patch_projector(patches)

        # 4. 准备查询 (Query, 也称为 "tgt")
        # query shape: [B, 5, 256]
        query = self.query_tokens.expand(B, -1, -1)

        # 5. [新] 执行完整的 Transformer 解码
        # tgt = query tokens [B, 5, 256]
        # memory = 图像块 [B, 64, 256]
        # attn_output shape: [B, 5, 256]
        attn_output = self.transformer_decoder(
            tgt=query, 
            memory=memory
        )

        # 6. [新] 通过 MLP 头进行预测
        # output shape: [B, 5, 1]
        output = self.prediction_head(attn_output)
        
        # 7. 压缩维度: [B, 5, 1] -> [B, 5]
        output = output.squeeze(-1)

        return output
