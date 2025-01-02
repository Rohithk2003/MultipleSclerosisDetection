import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm  # For Vision Transformer
from einops import rearrange
from einops.layers.torch import Rearrange


class CNNBranch(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNBranch, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fc = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        multi_scale = torch.cat([x1, x3, x5], dim=1)
        attention = self.fc(multi_scale)
        return F.relu(attention)


class VisionTransformerBlock(nn.Module):
    def __init__(self, img_size=7, patch_size=1, embed_dim=768):
        super(VisionTransformerBlock, self).__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        self.vit.patch_size = patch_size

    def forward(self, x):
        batch, channels, height, width = x.size()
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.vit(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossAttention, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, axial_features, sagittal_features):
        combined_features = torch.cat((axial_features, sagittal_features), dim=1)
        attention_features = self.fc(combined_features)
        return F.relu(attention_features)


class HybridCNNViTMultiScaleAttention(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridCNNViTMultiScaleAttention, self).__init__()

        self.cnn_axial = CNNBranch()
        self.cnn_sagittal = CNNBranch()

        self.multi_scale_axial = MultiScaleAttention(in_channels=2048, out_channels=512)
        self.multi_scale_sagittal = MultiScaleAttention(
            in_channels=2048, out_channels=512
        )

        self.vit_axial = VisionTransformerBlock(embed_dim=512)
        self.vit_sagittal = VisionTransformerBlock(embed_dim=512)

        self.cross_attention = CrossAttention(in_dim=512 * 2, out_dim=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, axial_view, sagittal_view):
        axial_features = self.cnn_axial(axial_view)
        axial_features = self.multi_scale_axial(axial_features)
        axial_features = axial_features.view(axial_features.size(0), -1)
        axial_features = self.vit_axial(axial_features)

        sagittal_features = self.cnn_sagittal(sagittal_view)
        sagittal_features = self.multi_scale_sagittal(sagittal_features)
        sagittal_features = sagittal_features.view(sagittal_features.size(0), -1)
        sagittal_features = self.vit_sagittal(sagittal_features)

        fused_features = self.cross_attention(axial_features, sagittal_features)

        out = self.fc(fused_features)
        return out


model = HybridCNNViTMultiScaleAttention(num_classes=2)
print(model)
