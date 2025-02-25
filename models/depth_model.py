from abc import ABC, abstractmethod
from torch import nn as nn
from torch import Tensor
from typing import Mapping, Any


class BaseDepthModel(nn.Module, ABC):
    """ Abstract Base Class for depth estimation models. """

    def __init__(self):
        """ General model wrapper for depth-estimation """
        super(BaseDepthModel, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model. Must be implemented by subclasses.

        Args:
            x (Tensor): Input image tensor (B, C, H, W)

        Returns:
            Tensor: Depth prediction tensor (B, 1, H, W)
        """
        pass

    def get_state(self):
        """ Returns model weights for federated learning updates"""
        return self.state_dict()

    def load_state(self, weights: Mapping[str, Any]):
        """ Load model weights received from the server """
        self.load_state_dict(weights)
