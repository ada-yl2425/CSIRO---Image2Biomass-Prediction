# KnowledgeDistillation/student_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class StudentModel(nn.Module):

    def __init__(self, img_model_name='mobilenetv2_100'):
        super(StudentModel, self).__init__()

        # 1. Image Backbone
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=0,
            global_pool='' 
        )

        self.num_img_features = self.img_backbone.num_features # 1408
        self.embed_dim = 1280 # embedding dims
        self.num_heads = 8   # attention head
        self.num_targets = 5 # 5 targets

        # 1.2 Projector
        self.patch_projector = nn.Linear(self.num_img_features, self.embed_dim)

        # 2. Query
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_targets, self.embed_dim)
        )

        # 2.2 Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,         # 256
            nhead=self.num_heads,           # 8
            dim_feedforward=self.embed_dim * 4, # 1024 (MLP  extension)
            dropout=0.3,                    
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=3  
        )

        # 3. Prediction Head
        
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2), # 256 -> 512
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(self.embed_dim * 2, 1) # 512 -> 1
        )

        # Selective Fine-tuning
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

        # 1. get feature map: [B, 1280, 8, 8]
        x_map = self.img_backbone(image)

        # 2. flat: [B, 64, 1280]
        patches = x_map.flatten(2).permute(0, 2, 1) 
        
        # 3. projector (get Key/Value)
        # memory shape: [B, 64, 256]
        memory = self.patch_projector(patches)

        # 4. Query
        # query shape: [B, 5, 256]
        query = self.query_tokens.expand(B, -1, -1)

        # 5. Transformer encoding
        # tgt = query tokens [B, 5, 256]
        # memory = [B, 64, 256]
        # attn_output shape: [B, 5, 256]
        attn_output = self.transformer_decoder(
            tgt=query, 
            memory=memory
        )

        # 6. get prediction
        # output shape: [B, 5, 1]
        output = self.prediction_head(attn_output)
        
        # 7. squeeze: [B, 5, 1] -> [B, 5]
        output = output.squeeze(-1)

        # --- MODIFICATION ---
        # Return both the final prediction AND the intermediate feature
        return output, attn_output
        # --- END MODIFICATION ---
