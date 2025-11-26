# KnowledgeDistillation/student_model.py
import torch
import torch.nn as nn
import timm

class StudentModel(nn.Module):
    """
    Pure Image-to-Biomass Model (The Student)

    Args:
        img_model_name (str): 使用的 timm 图像模型 (应与教师匹配)
        pretrained (bool): 是否使用 ImageNet 预训练权重
    """
    def __init__(self, img_model_name='efficientnet_b1', pretrained=True):
        super(StudentModel, self).__init__()

        # --- 1. 图像主干 (Image Backbone) ---
        # 加载预训练模型，不包括最后的分类器 (num_classes=0)
        # GlobalAveragePooling 会被自动应用
        self.img_backbone = timm.create_model(
            img_model_name,
            pretrained=pretrained,
            num_classes=0
        )

        # 获取图像模型的输出特征维度 (例如 B1 是 1280)
        self.num_img_features = self.img_backbone.num_features

        # --- 2. 预测头 (Prediction Head) ---
        # 学生的头需要更强，因为它只能从图像中学习
        # (1280 -> 512 -> 128 -> 5)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.num_img_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5), # 使用 0.5 作为标准 Dropout
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3), # 稍低的 Dropout
            
            nn.Linear(128, 5) # 最终输出5个值
        )

        # ** 注意：默认不冻结主干 **
        # 学生模型需要训练图像主干来提取所有必要信息
        # (可以根据需要选择性冻结)
        # for param in self.img_backbone.parameters():
        #     param.requires_grad = False
        # ... (在这里添加选择性冻结逻辑) ...


    def forward(self, image):
        """
        image: shape [batch_size, 3, H, W]
        """

        # 1. 处理图像
        # [B, 1280]
        img_features = self.img_backbone(image)

        # 2. 预测
        # [B, 5]
        output = self.prediction_head(img_features)

        return output