from models.sobel import Sobel
from models.resnet18 import ResNet18
from models.vit import ViT
import torch
import torch.nn as nn

class CCT(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel()
        self.resnet = ResNet18()
        self.vit = ViT()
        self.weight = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        contours = self.sobel(x)
        features = self.resnet(x)
        fus_map = torch.cat([features, contours], dim=1)
        weight_map = self.weight(fus_map)
        token_weight = fus_map * weight_map
        logits = self.vit(token_weight)
        
        return logits
 

        









