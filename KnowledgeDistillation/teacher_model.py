# KnowledgeDistillation/teacher_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # query shape: [B, 1, D]
        # key/value shape: [B, P, D]
        
        # attention output shape: [B, 1, D]
        attn_output, _ = self.cross_attn(
            query=query, 
            key=key, 
            value=value
        )
        
        # residual connection
        output = self.norm(query + attn_output)
        return output


class TeacherModel(nn.Module):

    def __init__(self, num_states, num_species, img_model_name='resnext50_32x4d'):
        super().__init__()
        
        # 1. Hyperparameters & Dimensions
        self.img_model_dim = 2048  # ResNeXt-50 layer4 output channels
        self.tab_model_dim = 256   # Fusion embedding dimension (D_tab)
        self.num_heads = 8         
        self.num_targets = 5       # 5 classes

        # 2. Image Branch Components (Key/Value source)
        self.img_backbone = self._init_image_backbone(img_model_name)
        self.img_kv_projector = nn.Linear(self.img_model_dim, self.tab_model_dim)

        # 3. Table Branch Components (Query source)
        self.tabular_embedder = self._init_tabular_embedder(num_states, num_species)
        self.tabular_processor = self._init_tabular_processor()
        
        # 添加缺失的 tab_self_attn
        self.tab_self_attn = nn.MultiheadAttention(
            embed_dim=self.tab_model_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        self.tab_q_projector = nn.Linear(self.tab_model_dim, self.tab_model_dim)
        self.tab_attn_norm = nn.LayerNorm(self.tab_model_dim)  # From forward pass

        # 4. Fusion Component (Cross-Attention)
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(self.tab_model_dim, self.num_heads),
            CrossAttentionBlock(self.tab_model_dim, self.num_heads)
        ])
        
        # 5. Prediction Head
        self.fusion_head = self._init_fusion_head()

        # 6. Fine-tuning Setup
        self._setup_selective_fine_tuning()


    # --- Initialization Helpers  ---
    
    def _init_image_backbone(self, img_model_name):
        return timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''  # Important: don't use GAP
        )

    def _init_tabular_embedder(self, num_states, num_species):
        self.num_numeric_features = 4
        self.state_embed_dim = 8
        self.species_embed_dim = 16
        # 将 tab_input_dim 存储为实例属性，而不是放在 ModuleDict 中
        self.tab_input_dim = (
            self.num_numeric_features + self.state_embed_dim + self.species_embed_dim
        )
        return nn.ModuleDict({
            'state_embedding': nn.Embedding(num_states, self.state_embed_dim),
            'species_embedding': nn.Embedding(num_species, self.species_embed_dim),
        })

    def _init_tabular_processor(self):
        # 使用实例属性 self.tab_input_dim
        input_dim = self.tab_input_dim
        return nn.Sequential(
            # MLP
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3), 
            nn.Linear(512, self.tab_model_dim),
            # Self-Attention is applied in forward pass
        )

    def _init_fusion_head(self):
        fusion_input_dim = self.tab_model_dim * 2  # 256 + 256 = 512
        return nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_targets)
        )

    def _setup_selective_fine_tuning(self):
        # 1. Freeze all parameters initially
        for param in self.img_backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the last few blocks (layer4 and layer3 for ResNeXt)
        if hasattr(self.img_backbone, 'layer4'):
            for param in self.img_backbone.layer4.parameters():
                param.requires_grad = True
        if hasattr(self.img_backbone, 'layer3'):
            for param in self.img_backbone.layer3.parameters():
                param.requires_grad = True

    # --- Forward Pass ---
    
    def forward(self, image, numeric_data, categorical_data):
        
        B = image.shape[0]

        # 1. Tabular Feature Extraction (Query Source)
        # 1.1 Embed categorical features
        state_idx = categorical_data[:, 0]
        species_idx = categorical_data[:, 1]
        state_emb = self.tabular_embedder['state_embedding'](state_idx)
        species_emb = self.tabular_embedder['species_embedding'](species_idx)
        tab_data = torch.cat([numeric_data, state_emb, species_emb], dim=1)
        
        # 1.2 MLP Processing
        tab_features = self.tabular_processor(tab_data)  # [B, 256]
        
        # 1.3 Self-Attention Enhancement
        tab_features_sa = tab_features.unsqueeze(1)  # [B, 1, 256]
        tab_sa_output, _ = self.tab_self_attn(
            query=tab_features_sa, 
            key=tab_features_sa, 
            value=tab_features_sa
        )
        tab_features = tab_features + tab_sa_output.squeeze(1) 
        tab_features = self.tab_attn_norm(tab_features)
        
        # 1.4 Project to Query
        query = self.tab_q_projector(tab_features).unsqueeze(1)  # [B, 1, 256]

        # 2. Image Feature Extraction (Key/Value Source)
        x_map = self.img_backbone(image)  # [B, C, H, W]
        x_patches = x_map.flatten(2).permute(0, 2, 1)  # [B, P, C]
        
        # Project to K/V
        key_value = self.img_kv_projector(x_patches)  # [B, P, 256]
        
        # 3. Cross Attention Fusion
        attn_output = query
        for block in self.cross_attention_blocks:
            attn_output = block(
                query=attn_output, 
                key=key_value, 
                value=key_value
            )
        
        # Attended Image Feature (aligned with tabular data)
        attended_img_features = attn_output.squeeze(1)  # [B, 256]

        # 4. Final Fusion and Prediction
        fused_features = torch.cat([tab_features, attended_img_features], dim=1)  # [B, 512]
        output = self.fusion_head(fused_features)  # [B, 5]
        
        return output, attended_img_features
