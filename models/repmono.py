import torch
from torch import Tensor
from models.depth_model import BaseDepthModel

from Repmono.Repmono.networks.Repmono_depth_encoder import RepMono as RepMonoEncoder
from Repmono.Repmono.networks.depth_decoder import DepthDecoder


class RepMonoModel(BaseDepthModel):
    """
    RepMono model implementation with forward pass.
    """

    def __init__(self, in_channels, height, width, depth_scale=1.0):
        """
        Initializes the RepMonoModel with a ResNet-like backbone and a transposed convolution decoder.
        """
        super(RepMonoModel, self).__init__()

        self.encoder = RepMonoEncoder(in_channels=in_channels,
                                      height=height,
                                      width=width,
                                      depth_scale=depth_scale)
        # PUT USE SKIPS TO FALSE because shit was crashing otherwise
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                    use_skips=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of RepMono model.

        Args:
            x (Tensor): Input RGB image tensor of shape (B, 3, H, W).

        Returns:
            Tensor: Predicted depth map tensor of shape (B, 1, H, W).
        """
        features = self.encoder(x)
        depth_outputs = self.decoder(features)
        depth_map = depth_outputs[('disp', 1)][0, 0].unsqueeze(0).unsqueeze(0)
        return depth_map
