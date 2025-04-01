import torch
from typing import Optional, List
from torch import Tensor
from torch.nn import functional as F
from models.depth_model import BaseDepthModel

from models.guidedepth.DDRNet_23_slim import DualResNet_Backbone
from models.guidedepth.modules import Guided_Upsampling_Block
from models.hybrid_guidedepth import HybridGuideDepthModel
from models.repmono.resnet_encoder import ResnetEncoder
from models.repmono.pose_decoder import PoseDecoder
from models.repmono.layers import BackprojectDepth, Project3D, disp_to_depth, transformation_from_parameters


class HybridModel(BaseDepthModel):
    """
    RepMono model implementation with forward pass.
    """

    def __init__(self, in_channels, height, width, depth_scale=1.0):
        """
        Initializes the RepMonoModel with a ResNet-like backbone and a transposed convolution decoder.
        """
        super(HybridModel, self).__init__()
        self.batch_size = 4
        self.height = height
        self.width = width
        self.scales = [0]

        self.depth_model = HybridGuideDepthModel()
        self.pose_encoder = ResnetEncoder(18, "pretrained", 2)
        self.pose_decoder = PoseDecoder(num_ch_enc=self.pose_encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=2)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.height // (2**scale)
            w = self.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.batch_size, h, w)
            self.backproject_depth[scale].to("cuda")

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to("cuda")
            
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of RepMono model.

        Args:
            x (Tensor): Input RGB image tensor of shape (B, 3, H, W).

        Returns:
            Tensor: Predicted depth map tensor of shape (B, 1, H, W).
        """
        depth_outputs = self.depth_model(x["image", 0, 0])
        depth_outputs.update(self._predict_poses(x))
        self._generate_images_pred(x, depth_outputs)
        
            # Visualize depth predictions for debugging
#         if not self.training:
#             self._visualize_depth_predictions(x, depth_outputs)
    

        return depth_outputs

    def _visualize_depth_predictions(self, inputs, outputs):
        """
        Visualize and save depth predictions and ground truth for debugging purposes.
        
        Args:
            inputs: Input dictionary containing RGB images and ground truth depth
            outputs: Output dictionary containing predicted depths
        """
        try:
            import matplotlib.pyplot as plt
            import os
            import numpy as np
            import time
            
            # Create visualization directory
            viz_dir = os.path.join("./results", "depth_predictions")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Get a timestamp for unique filenames
            timestamp = int(time.time())
            
            # Process first items in batch
            for batch_idx in range(min(2, inputs["image", 0, 0].shape[0])):  # Only visualize up to 2 items from batch
                # Get input RGB image
                rgb_image = inputs["image", 0, 0][batch_idx].detach().cpu().permute(1, 2, 0).numpy()
                # Normalize for visualization
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
                
                # Get predicted depth
                pred_depth = outputs[("disp", 0)][batch_idx, 0].detach().cpu().numpy()
                # Normalize for visualization
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
                
                # Create figure with RGB, predicted depth, and ground truth depth side by side
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Display RGB image
                axes[0].imshow(rgb_image)
                axes[0].set_title("Input RGB Image")
                axes[0].axis("off")
                
                # Display predicted depth
                depth_display = axes[1].imshow(pred_depth, cmap="plasma")
                axes[1].set_title("Predicted Depth")
                axes[1].axis("off")
                
                # Add colorbar for predicted depth
                plt.colorbar(depth_display, ax=axes[1], shrink=0.7)
                
                # Display ground truth depth if available
                if "depth" in inputs:
                    gt_depth = inputs["depth"][batch_idx, 0].detach().cpu().numpy()
                    # Normalize for visualization
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
                    gt_display = axes[2].imshow(gt_depth, cmap="plasma")
                    axes[2].set_title("Ground Truth Depth")
                    axes[2].axis("off")
                    # Add colorbar for ground truth
                    plt.colorbar(gt_display, ax=axes[2], shrink=0.7)
                else:
                    axes[2].text(0.5, 0.5, "Ground Truth Not Available", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes[2].transAxes)
                    axes[2].set_title("Ground Truth Depth")
                    axes[2].axis("off")
                
                # Save figure
                plt.savefig(f"{viz_dir}/depth_comparison_{timestamp}_{batch_idx}.png", bbox_inches="tight", dpi=300)
                plt.close(fig)
                
                # Print debug info
                print(f"Saved depth comparison to {viz_dir}/depth_comparison_{timestamp}_{batch_idx}.png")
        except Exception as e:
            print(f"Debug visualization error: {str(e)}")

    def _predict_poses(self, inputs):
        outputs = {}
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input
        self.frame_ids = [0, -1, 1]
        pose_features = {f_i: inputs["image_aug", f_i, 0] for f_i in self.frame_ids}

        for f_i in self.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_features[f_i], pose_features[0]]
            else:
                pose_inputs = [pose_features[0], pose_features[f_i]]

            pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]

            axisangle, translation = self.pose_decoder(pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0,
                     f_i)] = transformation_from_parameters(axisangle[:, 0],
                                                            translation[:, 0],
                                                            invert=(f_i < 0))
        return outputs

    def _generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.height, self.width],
                                 mode="bilinear",
                                 align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, 0.1, #HARDCODE
                                     100)   #HARDCODE

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                t = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], t)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("image", frame_id, scale)] = F.grid_sample(
                    inputs[("image", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                outputs[("image_identity", frame_id,
                         scale)] = inputs[("image", frame_id, source_scale)]
