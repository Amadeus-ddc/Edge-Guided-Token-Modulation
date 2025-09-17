import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(True)

        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_ch),
            )

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(64, 64, 2, stride_first=1)
        self.layer2 = self.make_layer(64, 128, 2, stride_first=2)
        self.layer3 = self.make_layer(128, 256, 2, stride_first=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    bn2 = m.residual_function[4]
                    assert isinstance(bn2, nn.BatchNorm2d)
                    bn2.weight.zero_()  

    def make_layer(self, in_ch, out_ch, num_blocks, stride_first):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride_first))
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        feat = self.layer3(out)
        return feat
        
model = ResNet18()
