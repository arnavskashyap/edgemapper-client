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
        """Plot validation results"""
        # with torch.no_grad():
        #     for batch_idx, batch in enumerate(tqdm(self.val_loader)):                
        #         t0 = time.time()
        #         batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        #         images = batch
        #         gt_depths = batch["depth"]  # Ground truth depth (B, 1, H, W)

        #         # Forward pass
        #         pred_depths = self.model(images)  
               
        #         # Create a figure with multiple rows for different frames
        #         fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                
        #         # Define frame indices to visualize (current=0, previous=-1, next=1)
        #         frame_indices = [0, -1, 1]
                
        #         # Set row titles
        #         row_titles = ["Current Frame", "Previous Frame", "Next Frame"]
                
        #         # Process each frame
        #         for i, frame_idx in enumerate(frame_indices):
        #             # Get image from the specific frame (first batch item)
        #             if ("image", frame_idx, 0) in images:
        #                 image = images[("image", frame_idx, 0)][0].cpu().permute(1, 2, 0).numpy()
                        
        #                 # For previous and next frames, we only have images, not depth predictions
        #                 axes[i, 0].imshow(image)
        #                 axes[i, 0].set_title(f"{row_titles[i]} RGB")
        #                 axes[i, 0].axis("off")
                    
        #             # Only show depth for the current frame (frame_idx=0)
        #             if i == 0:
        #                 # Get predicted depth for current frame
        #                 pred_depth = pred_depths[("disp", 0)][0, 0].cpu().numpy()
                        
        #                 # Get ground truth depth (first batch item)
        #                 if len(gt_depths.shape) == 5 and gt_depths.shape[-1] == 3:  # Handle RGB depth
        #                     gt_depth = gt_depths[0, 0].cpu().numpy().mean(axis=2)  # Average RGB channels
        #                 else:
        #                     gt_depth = gt_depths[0, 0].cpu().numpy()
                        
        #                 # Normalize for visualization
        #                 pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
        #                 gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
                        
        #                 # Display predicted depth
        #                 axes[i, 1].imshow(pred_depth, cmap="plasma")
        #                 axes[i, 1].set_title(f"{row_titles[i]} Predicted Depth")
        #                 axes[i, 1].axis("off")
                        
        #                 # Display ground truth depth
        #                 axes[i, 2].imshow(gt_depth, cmap="plasma")
        #                 axes[i, 2].set_title(f"{row_titles[i]} Ground Truth")
        #                 axes[i, 2].axis("off")
        #             else:
        #                 # For other frames, just gray out the depth plots
        #                 axes[i, 1].set_facecolor('lightgray')
        #                 axes[i, 1].text(0.5, 0.5, "No depth prediction\nfor this frame", 
        #                                horizontalalignment='center', verticalalignment='center')
        #                 axes[i, 1].axis("off")
                        
        #                 axes[i, 2].set_facecolor('lightgray')
        #                 axes[i, 2].text(0.5, 0.5, "No ground truth\nfor this frame", 
        #                                horizontalalignment='center', verticalalignment='center')
        #                 axes[i, 2].axis("off")
                
        #         # Adjust spacing between subplots
        #         plt.tight_layout()
                
        #         # Save the figure
        #         save_path = os.path.join("./results", f"depth_comparison_{t0}.png")
        #         directory = os.path.dirname(save_path)
        #         if not os.path.exists(directory):
        #             os.makedirs(directory)
        #         plt.savefig(save_path, bbox_inches="tight", dpi=300)
        #         plt.close(fig)  # Close the figure to free memory
        #         if batch_idx == 1:
        #             break

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
