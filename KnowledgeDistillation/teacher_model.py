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
        super(TeacherModel, self).__init__()
        
        self.img_model_dim = 2048  # characteristics dims of ResNeXt-50
        self.tab_model_dim = 256   # output dims of tabel branch
        self.num_heads = 8         # head of cross attention

        # 1. Image Branch
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''  # don't use GAP!!!
        )

        self.img_kv_projector = nn.Linear(self.img_model_dim, self.tab_model_dim)
        
        # 2. Table Branch
        self.num_numeric_features = 4
        self.state_embed_dim = 8
        self.species_embed_dim = 16
        self.state_embedding = nn.Embedding(num_states, self.state_embed_dim)
        self.species_embedding = nn.Embedding(num_species, self.species_embed_dim)
        self.tab_input_dim = (
            self.num_numeric_features + self.state_embed_dim + self.species_embed_dim
        )
        
        # Enhance Tab MLP
        self.tab_mlp = nn.Sequential(
            nn.Linear(self.tab_input_dim, 512), # <-- increase width
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3), 
            nn.Linear(512, self.tab_model_dim) 
        )
        
        # Add a Self-Attention layer
        self.tab_self_attn = nn.MultiheadAttention(
            embed_dim=self.tab_model_dim, # 256
            num_heads=self.num_heads, # 8
            batch_first=True
        )

        # 3. Cross-Attention
        
        # 3.1 Query projector layer
        self.tab_q_projector = nn.Linear(self.tab_model_dim, self.tab_model_dim)
        
        # 3.2 Cross-Attention layer
        self.cross_attn = nn.ModuleList([
            CrossAttentionBlock(self.tab_model_dim, self.num_heads),
            CrossAttentionBlock(self.tab_model_dim, self.num_heads)
        ])        
        # 3.3 Attention Normalization
        self.attn_norm = nn.LayerNorm(self.tab_model_dim)

        # 4. Final Head
        self.fusion_input_dim = self.tab_model_dim + self.tab_model_dim # 256 + 256 = 512
        
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

        # Selective Fine-tuning
        # 1. Freeze all parameters initially
        for param in self.img_backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the last few blocks (layer4 and layer3)
        for param in self.img_backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.img_backbone.layer3.parameters():
            param.requires_grad = True


    def forward(self, image, numeric_data, categorical_data):
        
        # 1. tabel: get Query
        state_idx = categorical_data[:, 0]
        species_idx = categorical_data[:, 1]
        state_emb = self.state_embedding(state_idx)
        species_emb = self.species_embedding(species_idx)
        tab_data = torch.cat([numeric_data, state_emb, species_emb], dim=1)
        
        # tab_features shape: [B, 256]
        tab_features = self.tab_mlp(tab_data)
        
        # Apply Self Attention
        # shape: [B, 1, 256]
        tab_features_sa = tab_features.unsqueeze(1) 
        tab_sa_output, _ = self.tab_self_attn(
            query=tab_features_sa, 
            key=tab_features_sa, 
            value=tab_features_sa
        )
        tab_features = tab_features + tab_sa_output.squeeze(1) 
        tab_features = self.tab_attn_norm(tab_features)
        
        # Q shape: [B, 1, 256] (1 represents 1 Query)
        query = self.tab_q_projector(tab_features).unsqueeze(1) 

        # 2. image: get Key and Value
        # x_map shape: [B, 1280, H, W]
        x_map = self.img_backbone(image)
        
        B, C, H, W = x_map.shape
        
        # x_patches shape: [B, H*W, C]
        x_patches = x_map.flatten(2).permute(0, 2, 1) 
        
        # K/V shape: [B, 64, 256]
        key_value = self.img_kv_projector(x_patches)
        
        # 3. Cross Attention
        # Q = [B, 1, 256] 
        # K = [B, 64, 256] 
        # V = [B, 64, 256]
        # attn_output shape: [B, 1, 256]
        attn_output = query
        for block in self.cross_attn:
            attn_output = block(
                query=attn_output, 
                key=key_value, 
                value=key_value
            )
        
        # attended_img_features shape: [B, 256]
        attended_img_features = attn_output.squeeze(1)

        # 4. Fusion
        # [B, 256] (tabel) + [B, 256] (image with CA)
        fused_features = torch.cat([tab_features, attended_img_features], dim=1) # [B, 512]

        self.fusion_input_dim = self.tab_model_dim + self.tab_model_dim # 512
        
        output = self.fusion_head(fused_features)
        
        # --- MODIFICATION ---
        # Return both the final prediction AND the intermediate feature
        return output, attended_img_features
        # --- END MODIFICATION ---