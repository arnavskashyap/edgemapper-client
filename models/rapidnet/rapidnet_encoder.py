import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.stem(x)

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, e=1.5):  # Further reduced expansion ratio
        super().__init__()
        # Simplified version with fewer operations
        expanded = max(int(in_dim * e), 8)
        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_dim, expanded, kernel_size=1),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(expanded, expanded, kernel_size=kernel, stride=1, padding=1, groups=expanded),
            nn.BatchNorm2d(expanded),
            # Pointwise projection
            nn.Conv2d(expanded, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim)
        )
    
    def forward(self, x):
        return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio, drop_path):
        super().__init__()
        self.dim = dim
        # Ultra-simplified residual block
        expanded = max(int(dim * expansion_ratio), 8)
        self.conv = nn.Sequential(
            # Simplified MBConv
            nn.Conv2d(dim, expanded, kernel_size=1),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded, expanded, kernel_size=kernel, padding=kernel//2, groups=expanded),
            nn.BatchNorm2d(expanded),
            nn.Conv2d(expanded, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        # Remove DropPath to save memory
        self.drop_path = nn.Identity()

    def forward(self, x):
        return x + self.conv(x)

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConv, self).__init__()
        # Super lightweight dilated convolution
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)  # ReLU uses less memory than GELU
        )

    def forward(self, x):
        return self.nn(x)

class MLDC(nn.Module):
    def __init__(self, in_channels):
        super(MLDC, self).__init__()
        # Ultra lightweight implementation
        mid_channels = max(in_channels // 4, 8)  # More aggressive reduction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
       
    def forward(self, x):
        return x + self.conv(x)  # Simple residual connection

class MLDC_LKFFN(nn.Module):
    def __init__(self, in_dim, drop_path=0.1, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        # Ultra-simplified global mixer
        self.mixer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim)
        )
        # Remove layer scaling and drop path
        self.drop_path = nn.Identity()
        self.use_layer_scale = False
        
    def forward(self, x):
        return x + self.mixer(x)

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Simplified downsampling
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        return self.bn(self.conv(x))

class RapidNetEncoder(nn.Module):
    def __init__(self, input_channels=3, blocks=[[2,0], [2,0], [6,2], [2,2]], channels=[32, 64, 112, 224], drop_path=0.1):
        super().__init__()
        self.stem = Stem(input_dim=input_channels, output_dim=channels[0])
        
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]
        dpr_idx = 0
        
        self.backbone = []
        for i in range(len(blocks)):
            local_stages, global_stages = blocks[i]
            
            # Stage with local and global blocks
            stage = []
            if i > 0:
                stage.append(Downsample(channels[i-1], channels[i]))
            for _ in range(local_stages):
                # RapidNet_Ti uses expansion_ratio=2
                stage.append(InvertedResidual(dim=channels[i], kernel=3, expansion_ratio=2, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(MLDC_LKFFN(channels[i], drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))
            
        self.backbone = nn.Sequential(*self.backbone)
        self.channels = channels
        
    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        return x