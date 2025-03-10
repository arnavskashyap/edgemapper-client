import time
import os
from typing import Optional
import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader

from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from datasets import get_dataset
from models import get_model
from training.loss_functions import DepthLoss, RepMonoUnsupervisedLoss
from training.metrics import AverageMeter, Result, plot_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 training_dataset_path: str,
                 device: torch.device,
                 batch_size: int,
                 lr: float,
                 val_dataset_path: Optional[str],
                 loss_function=None):
        """
        Initialize the trainer.
        """
        self.device = device
        self.model = model
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if loss_function is None:
            self.loss_function = RepMonoUnsupervisedLoss()
        else:
            self.loss_function = loss_function
        logger.info(
            f"Model, Optimizer, and Loss Function initialized on device {self.device}"
        )

        self.local_epoch = 0
        self.batch_size = batch_size

        self.training_dataset = get_dataset(model_name, training_dataset_path,
                                            False)
        self.train_loader = DataLoader(self.training_dataset,
                                       self.batch_size,
                                       shuffle=False,
                                       drop_last=True)
        logger.info("Created training dataset and dataloader")
        logger.info(f"Created training dataset of size {len(self.training_dataset)} and dataloader of size {len(self.train_loader)}")

        if val_dataset_path:
            self.val_dataset = get_dataset(model_name, val_dataset_path, True)
            self.val_loader = DataLoader(self.val_dataset,
                                         self.batch_size,
                                         shuffle=False,
                                         drop_last=True)
            logger.info(f"Created validation dataset of size {len(self.val_dataset)} and dataloader of size {len(self.val_loader)}")

        self.metrics = {
                "RMSE": [],
                "MAE": [],
                "Delta1": [],
                "Delta2": [],
                "Delta3": [],
                "REL": [],
                "Lg10": [],
            }

    def _train_one_epoch(self) -> float:
        """Trains the model for one epoch."""
        logger.info("Training model")
        
        torch.cuda.empty_cache()

        self.model.train()
        total_loss = 0.0

        # Supervised learning
        # for batch_idx, batch in enumerate(tqdm(self.train_loader)):
        #     image, gt = self._unpack_and_move(batch)
        #     self.optimizer.zero_grad()

        #     prediction = self.model(image)
        #     gt = gt / gt.max()

        #     loss = self.loss_function(prediction, gt)
        #     loss.backward()
        #     self.optimizer.step()

        #     total_loss += loss.item()

        # Unsupervised learning
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            if batch_idx >= len(self.train_loader) // 3:
                break
            # if batch_idx >= 10:
            #     break
            batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            images = batch
            self.optimizer.zero_grad()

            depth_predictions = self.model(images)

            depth_pred = depth_predictions[('disp', 0)]
            loss = self.loss_function(images, depth_predictions)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Report
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = total_loss / (len(self.train_loader) // 2)
        logger.info(
            f"{current_time} - Average Training Loss: {average_loss:3.4f}")
        return average_loss

    def _unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt

        if isinstance(data, dict):
            keys = data.keys()
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt

    def validate(self):
        """Validates current model on pre-loaded validation dataset."""
        self.model.eval()

        total_loss = 0.0
        average_meter = AverageMeter()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader)):
                if batch_idx >= len(self.val_loader) // 2:
                    break
                t0 = time.time()
                
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                images = batch
                gt_depths = batch["depth"]  # Ground truth depth (B, 1, H, W)
                data_time = time.time() - t0

                # Forward pass
                t0 = time.time()
                pred_depths = self.model(images)
                gpu_time = time.time() - t0

                # Compute L1 loss
                loss_func = DepthLoss(1, 0, 0, 10.0)
                pred_depth = pred_depths[("disp", 0)][0, 0]  # Convert to (H, W)
                gt_depth = gt_depths[0, 0]  # Convert to (H, W)
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
                gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
                loss = loss_func(pred_depth, gt_depth)
                total_loss += loss.item()

                result = Result()
                result.evaluate(pred_depth.data, gt_depth.data)
                average_meter.update(result, gpu_time, data_time)

        # Report
        avg = average_meter.average()
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = total_loss / (len(self.val_loader.dataset) // 2)
        logger.info(
            f"{current_time} - Average Validation Loss: {average_loss:3.4f}")

        logger.info('\n*\n'
                    'RMSE={average.rmse:.3f}\n'
                    'MAE={average.mae:.3f}\n'
                    'Delta1={average.delta1:.3f}\n'
                    'Delta2={average.delta2:.3f}\n'
                    'Delta3={average.delta3:.3f}\n'
                    'REL={average.absrel:.3f}\n'
                    'Lg10={average.lg10:.3f}\n'
                    't_GPU={time:.3f}\n'.format(average=avg,
                                                time=avg.gpu_time))

        self.metrics["RMSE"].append(avg.rmse)
        self.metrics["MAE"].append(avg.mae)
        self.metrics["Delta1"].append(avg.delta1)
        self.metrics["Delta2"].append(avg.delta2)
        self.metrics["Delta3"].append(avg.delta3)
        self.metrics["REL"].append(avg.absrel)
        self.metrics["Lg10"].append(avg.lg10)
        return self.metrics

    def plot_val(self):
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader)):                
                t0 = time.time()
                batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                images = batch
                gt_depths = batch["depth"]  # Ground truth depth (B, 1, H, W)

                # Forward pass
                pred_depths = self.model(images)  
               
                # Visualization (showing the first image in the batch)
                image = images[("image", 0, 0)][0].cpu().permute(1, 2, 0).numpy()  # Extract first image from first batch and convert to (H, W, 3)
                pred_depth = pred_depths[("disp", 0)][0, 0].cpu().numpy()  # Convert to (H, W)
                # pred_depth = pred_depths[("depth", 0, 0)][0, 0].cpu()  # Convert to (H, W)
                # depth_pred = torch.clamp(depth_pred, min=1e-3, max=100)
                gt_depth = gt_depths[0, 0].cpu().numpy()  # Convert to (H, W)

                # Normalize for visualization
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
                gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image)
                axes[0].set_title("RGB Image")
                axes[0].axis("off")

                axes[1].imshow(pred_depth, cmap="plasma")
                axes[1].set_title("Predicted Depth")
                axes[1].axis("off")

                axes[2].imshow(gt_depth, cmap="plasma")
                axes[2].set_title("Ground Truth Depth")
                axes[2].axis("off")

                save_path = os.path.join("./results", f"depth_comparison_{t0}.png")
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                plt.close(fig)  # Close the figure to free memory
                break  # Show only the first batch
    
    def plot_results(self, results_dir: str = "./results"):
        plot_metrics(self.metrics, "./results")

    def train(self, num_epochs: int):
        """Trains the model for multiple epochs and saves checkpoints."""
        for epoch in range(num_epochs):
            self.local_epoch = epoch
            logger.info(f"Training Epoch {epoch+1}/{num_epochs}")

            loss = self._train_one_epoch()

            save_checkpoint(epoch + 1, self.model, self.optimizer, loss,
                            "./checkpoints")
            current_time = time.strftime('%H:%M', time.localtime())
            logger.info(
                f"{current_time} - Checkpoint for local epoch {epoch + 1} saved")

        logger.info("Training Complete.")

    # TODO: Need to fix this for continous streams. Maybe we won't save the images?
    def update_dataset(self, training_dataset_path: str):
        self.training_dataset = get_dataset(self.model_name, training_dataset_path, False)
        self.train_loader = DataLoader(self.training_dataset,
                                       self.batch_size,
                                       shuffle=False,
                                       drop_last=True)
        logger.info("Updated training dataset and dataloader")

    def get_model_weights(self):
        """Returns the model's parameters for federated learning updates."""
        return self.model.get_state()

    def load_model_weights(self, weights):
        """Loads new model parameters from federated updates."""
        self.model.load_state(weights)
