import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint

class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        mid_dim = max(output_dim // 4, 8)  # More aggressive reduction
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, mid_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),  # ReLU uses less memory than GELU
            nn.Conv2d(mid_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.stem(x)

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, e=1.0):  # Further reduced expansion ratio
        super().__init__()
        # Simplified version with fewer operations and group convolutions
        expanded = max(int(in_dim * e), 8)
        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_dim, expanded, kernel_size=1, groups=4 if in_dim % 4 == 0 else 1),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(expanded, expanded, kernel_size=kernel, stride=1, padding=1, groups=expanded),
            nn.BatchNorm2d(expanded),
            # Pointwise projection
            nn.Conv2d(expanded, in_dim, kernel_size=1, groups=4 if in_dim % 4 == 0 else 1),
            nn.BatchNorm2d(in_dim)
        )
    
    def forward(self, x):
        return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio, drop_path):
        super().__init__()
        self.dim = dim
        # Reduced expansion ratio
        expanded = max(int(dim * expansion_ratio * 0.5), 8)  # Halve expansion ratio
        self.conv = nn.Sequential(
            # Simplified MBConv with group convolutions
            nn.Conv2d(dim, expanded, kernel_size=1, groups=4 if dim % 4 == 0 else 1),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded, expanded, kernel_size=kernel, padding=kernel//2, groups=expanded),
            nn.BatchNorm2d(expanded),
            nn.Conv2d(expanded, dim, kernel_size=1, groups=4 if expanded % 4 == 0 else 1),
            nn.BatchNorm2d(dim)
        )
        # Only use DropPath if value is significant
        self.drop_path = DropPath(drop_path) if drop_path > 0.05 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.conv(x))

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConv, self).__init__()
        # Super lightweight dilated convolution with group convolutions
        groups = 4 if in_channels % 4 == 0 else 1
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.nn(x)

class MLDC(nn.Module):
    def __init__(self, in_channels):
        super(MLDC, self).__init__()
        # Ultra lightweight implementation
        mid_channels = max(in_channels // 8, 8)  # More aggressive reduction
        groups = 4 if in_channels % 4 == 0 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(in_channels)
        )
       
    def forward(self, x):
        return x + self.conv(x)  # Simple residual connection

class MLDC_LKFFN(nn.Module):
    def __init__(self, in_dim, drop_path=0.1, use_layer_scale=False, layer_scale_init_value=1e-5):
        super().__init__()
        # Ultra-simplified global mixer with group convolutions
        groups = 4 if in_dim % 4 == 0 else 1
        self.mixer = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, groups=groups),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(in_dim)
        )
        # Only use DropPath if value is significant
        self.drop_path = DropPath(drop_path) if drop_path > 0.05 else nn.Identity()
        self.use_layer_scale = False
        
    def forward(self, x):
        return x + self.drop_path(self.mixer(x))

class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # More efficient downsampling with intermediate dimension reduction
        mid_dim = max((in_dim + out_dim) // 4, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1),  # Reduce dimensions first
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim)
        )
        
    def forward(self, x):
        return self.conv(x)

class RapidNetEncoder(nn.Module):
    def __init__(self, input_channels=3, blocks=[[2,0], [2,0], [6,2], [2,2]], 
                 channels=[24, 48, 96, 192],  # Reduced channel counts by ~25%
                 drop_path=0.1,
                 use_checkpoint=False):
        super().__init__()
        self.stem = Stem(input_dim=input_channels, output_dim=channels[0])
        self.use_checkpoint = use_checkpoint
        
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
                # Reduced expansion_ratio from 2 to 1
                stage.append(InvertedResidual(dim=channels[i], kernel=3, expansion_ratio=1, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(MLDC_LKFFN(channels[i], drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))
            
        self.backbone = nn.Sequential(*self.backbone)
        self.channels = channels
        
    def forward(self, x):
        x = self.stem(x)
        
        if self.use_checkpoint and self.training:
            # Apply gradient checkpointing to save memory during training
            for i, block in enumerate(self.backbone):
                if i == 0:  # Skip checkpointing first block to avoid issues with autograd
                    x = block(x)
                else:
                    x = checkpoint.checkpoint(block, x)
        else:
            x = self.backbone(x)
            
        return x
        
    def clear_memory(self):
        """Explicitly clear unused variables to assist garbage collection"""
        if hasattr(self, 'temp_var'):
            del self.temp_var
        torch.cuda.empty_cache()