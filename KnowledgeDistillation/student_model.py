# KnowledgeDistillation/student_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class StudentModel(nn.Module):

    def __init__(self, img_model_name='resnext50_32x4d'):
        super().__init__()

        # 1. Hyperparameters & Dimensions
        self.img_model_dim = 2048
        self.embed_dim = 256
        self.num_targets = 5
        self.num_heads = 8

        # 2. Image Branch Components (Key/Value source)
        self.img_backbone = self._init_image_backbone(img_model_name)
        self.img_patch_projector = nn.Linear(self.img_model_dim, self.embed_dim)

        # 3. Query source
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_targets, self.embed_dim)
        )

        # 4. Fusion Component (Self-Attention, Cross-Attention, FFN)
        self.transformer_decoder = self._init_transformer_decoder()

        # 5. Prediction Head
        self.prediction_head = self._init_prediction_head()

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

    def _init_transformer_decoder(self):
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,         # 256
            nhead=self.num_heads,           # 8
            dim_feedforward=self.embed_dim * 4, # 1024 (MLP  extension)
            dropout=0.3,                    
            batch_first=True
        )
        return nn.TransformerDecoder(
            decoder_layer, 
            num_layers=3  
        ) 

    def _init_prediction_head(self):
        return nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2), # 256 -> 512
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(self.embed_dim * 2, 1) # 512 -> 1
        )

    def _setup_selective_fine_tuning(self):
        # 1. Freeze all parameters initially
        for param in self.img_backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the last few blocks (layer4 and layer3 batchnorm1 for ResNeXt)
        if hasattr(self.img_backbone, 'layer4'):
            for param in self.img_backbone.layer4.parameters():
                param.requires_grad = True
        if hasattr(self.img_backbone, 'layer3'):
            for param in self.img_backbone.layer3.parameters():
                param.requires_grad = True
        if hasattr(self.img_backbone, 'bn1'):
            for param in self.img_backbone.bn1.parameters():
                param.requires_grad = True

    # --- Forward Pass ---

    def forward(self, image):

        B = image.shape[0]

        # 1. Image Feature Extraction (Key/Value Source)
        x_map = self.img_backbone(image)
        x_patches = x_map.flatten(2).permute(0, 2, 1) 
        
        # Project to K/V
        key_value = self.img_patch_projector(x_patches)

        # Query
        query = self.query_tokens.expand(B, -1, -1)

        # 2. Transformer encoding
        attn_output = self.transformer_decoder(
            tgt=query, 
            memory=key_value
        )

        # 3. Prediction
        output = self.prediction_head(attn_output)
        output = output.squeeze(-1)

        return output, attn_output
