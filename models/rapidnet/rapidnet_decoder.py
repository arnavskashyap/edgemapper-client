import torch
import torch.nn as nn

class ConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ConvFFN, self).__init__()
        self.out_channels = out_channels or in_channels
        self.hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias=False,
            ),
        )
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(num_features=hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm3(x)
        return x

class RapidNetDecoder(nn.Module):
    def __init__(self, in_channels, emb_dims=256, dropout=0., num_classes=1000, distillation=True):
        super().__init__()
        self.distillation = distillation
        
        # Ultra-simplified prediction pathway - eliminates unnecessary components
        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            # Direct projection to lower dimensions (skip intermediate layers)
            nn.Conv2d(in_channels, emb_dims // 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(emb_dims // 2),
            nn.ReLU(inplace=True)  # ReLU is more memory efficient than GELU
        )
        
        # Minimal heads for pose estimation
        self.axisangle = nn.Conv2d(emb_dims // 2, 3, kernel_size=1, bias=True)
        self.translation = nn.Conv2d(emb_dims // 2, 3, kernel_size=1, bias=True)
        
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
        x = self.prediction(x)
        
        # Create pose outputs with the proper shape [B, 1, 3]
        axisangle = self.axisangle(x)  # [B, 3, 1, 1]
        translation = self.translation(x)  # [B, 3, 1, 1]
        
        # Reshape tensors to have shape [B, 1, 3]
        axisangle = axisangle.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1, 3]
        translation = translation.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1, 3]
        
        return axisangle, translation