import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from torch import Tensor


class Sobel(nn.Module):

    def __init__(self, in_ch = 3, return_xy = False):
        super().__init__()
        self.in_ch = in_ch
        self.return_xy = return_xy

        kx = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ], dtype=torch.float32)
        ky = torch.tensor([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0],
        ], dtype=torch.float32)

        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))
        self.downsample1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.downsample2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        weight_x: Tensor = self.kx.repeat(self.in_ch, 1, 1, 1).to(device=x.device, dtype=x.dtype) # type: ignore
        weight_y: Tensor = self.ky.repeat(self.in_ch, 1, 1, 1).to(device=x.device, dtype=x.dtype) # type: ignore

        gx: Tensor = F.conv2d(x, weight_x, bias=None, stride=1, padding=1, groups=self.in_ch)
        gy: Tensor = F.conv2d(x, weight_y, bias=None, stride=1, padding=1, groups=self.in_ch)

        if self.return_xy:
            return gx, gy

        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
        mag = mag.mean(dim=1, keepdim=True)
        contours = self.downsample1(mag)
        contours = self.downsample2(contours)

        return contours

__all__ = ["Sobel"]
