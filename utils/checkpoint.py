import os
from typing import Tuple

import torch
from torch import nn as nn


def save_checkpoint(epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, loss: float,
                    checkpoint_dir: str) -> None:
    """
    Saves the model checkpoint.

    Args:
        epoch (int): Current local epoch number.
        model (nn.Module): Model whose state will be restored.
        optimizer (torch.optim.Optimizer): Optimizer used during training.
        loss (float): Current training loss.
        checkpoint_dir (str, optional): Directory to save checkpoint. Defaults to "latest.pth".
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
        model: nn.Module, optimizer: torch.optim.Optimizer,
        checkpoint_path: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Loads a checkpoint state into the model and optimizer.

    Args:
        model (nn.Module): Model whose state will be restored.
        optimizer (torch.optim.Optimizer): Optimizer whose state will be restored.
        checkpoint_path (str): Path to saved checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer
