import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, groups=1):
        super(ConvFFN, self).__init__()
        self.out_channels = out_channels or in_channels
        self.hidden_channels = hidden_channels or in_channels
        
        # Remove depthwise convolution to save memory
        # Use group convolutions in the pointwise layers
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, 
                             stride=1, padding=0, groups=groups)
        self.norm1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.act = nn.ReLU(inplace=True)  # ReLU uses less memory than GELU
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, 
                             stride=1, padding=0, groups=groups)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x

class RapidNetDecoder(nn.Module):
    def __init__(self, in_channels, emb_dims=32, dropout=0., num_classes=1000, 
                 distillation=True, use_checkpoint=False):
        super().__init__()
        self.distillation = distillation
        self.use_checkpoint = use_checkpoint
        
        # Apply early pooling to reduce feature map sizes
        self.pool = nn.AdaptiveAvgPool2d(2)  # Reduce to 2x2 spatial dimension
        
        # Use group convolutions to reduce parameter count
        # Default groups to 4 if divisible by 4, otherwise 1
        groups = 4 if in_channels % 4 == 0 else 1
        
        # Ultra-simplified prediction pathway with reduced dimensions
        self.prediction = nn.Sequential(
            ConvFFN(in_channels, in_channels // 2, in_channels // 2, groups=groups),
            nn.Conv2d(in_channels // 2, emb_dims, kernel_size=1),
            nn.BatchNorm2d(emb_dims),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Final global pooling
        )
        
        # Minimal heads for pose estimation
        self.axisangle = nn.Conv2d(emb_dims, 3, kernel_size=1, bias=True)
        self.translation = nn.Conv2d(emb_dims, 3, kernel_size=1, bias=True)
        
        # Simple initialization
        self._init_weights()
        
    def _init_weights(self):
        # Simplified initialization for lower memory usage
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def model_init(self):
        # Alias for backward compatibility
        pass
    
    def forward(self, x):
        # Apply early pooling to reduce feature map sizes
        x = self.pool(x)
        
        # Apply gradient checkpointing if enabled during training
        if self.use_checkpoint and self.training:
            x = checkpoint.checkpoint(self.prediction, x)
        else:
            x = self.prediction(x)
        
        # Create pose outputs with the proper shape [B, 1, 3]
        axisangle = self.axisangle(x)  # [B, 3, 1, 1]
        translation = self.translation(x)  # [B, 3, 1, 1]
        
        # Reshape tensors to have shape [B, 1, 3]
        axisangle = axisangle.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1, 3]
        translation = translation.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1, 3]
        
        # Explicitly clear unused variables to assist garbage collection
        del x
        torch.cuda.empty_cache()
        
        return axisangle, translation
        
    def clear_memory(self):
        """Explicitly clear unused variables to assist garbage collection"""
        if hasattr(self, 'temp_var'):
            del self.temp_var
        torch.cuda.empty_cache()