from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.depth_model import BaseDepthModel
from models.guidedepth.DDRNet_23_slim import DualResNet_Backbone
from models.guidedepth.modules import Guided_Upsampling_Block


class GuideDepthModel(BaseDepthModel):
    """
    GuideDepth model implementation with forward pass.
    """

    def __init__(self,
                 pretrained: bool = True,
                 up_features: Optional[List[int]] = None,
                 inner_features: Optional[List[int]] = None) -> None:
        """Initializes the GuideDepthModel

        Args:
            pretrained (bool, optional): Whether to load a pretrained DualResNet_Backbone. Defaults to True.
            up_features (Optional[List[int]], optional): List of feature sizes for upsampling blocks. Defaults to None.
            inner_features (Optional[List[int]], optional): List of feature sizes for internal processing. Defaults to None.
        """

        super(GuideDepthModel, self).__init__()

        if up_features is None:
            up_features = [64, 32, 16]
        if inner_features is None:
            inner_features = [64, 32, 16]  # Fixed incorrect assignment

        self.feature_extractor = DualResNet_Backbone(pretrained=pretrained,
                                                     features=up_features[0])

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                            expand_features=inner_features[0],
                                            out_features=up_features[1],
                                            kernel_size=3,
                                            channel_attention=True,
                                            guide_features=3,
                                            guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                            expand_features=inner_features[1],
                                            out_features=up_features[2],
                                            kernel_size=3,
                                            channel_attention=True,
                                            guide_features=3,
                                            guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                            expand_features=inner_features[2],
                                            out_features=1,
                                            kernel_size=3,
                                            channel_attention=True,
                                            guide_features=3,
                                            guidance_type="full")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of GuideDepth model.

        :param x: Input RGB image tensor of shape (B, 3, H, W).
        :return: Predicted depth map tensor of shape (B, 1, H, W).
        """
        x = x["image"]
        y = self.feature_extractor(x)

        x_half = F.interpolate(x,
                               scale_factor=0.5,
                               mode='bilinear',
                               align_corners=False)
        x_quarter = F.interpolate(x,
                                  scale_factor=0.25,
                                  mode='bilinear',
                                  align_corners=False)

        y = F.interpolate(y,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        y = self.up_2(x_half, y)

        y = F.interpolate(y,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False)
        y = self.up_3(x, y)

        return y
