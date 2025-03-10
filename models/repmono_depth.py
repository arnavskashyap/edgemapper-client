import torch
from torch import Tensor
from torch.nn import functional as F
from models.depth_model import BaseDepthModel

from models.repmono.Repmono_depth_encoder import RepMono as RepMonoEncoder
from models.repmono.depth_decoder import DepthDecoder
from models.repmono.resnet_encoder import ResnetEncoder
from models.repmono.pose_decoder import PoseDecoder
from models.repmono.layers import BackprojectDepth, Project3D, disp_to_depth, transformation_from_parameters


class RepMonoSupervisedModel(BaseDepthModel):
    """
    RepMono model implementation with forward pass.
    """

    def __init__(self, in_channels, height, width, depth_scale=1.0):
        """
        Initializes the RepMonoModel with a ResNet-like backbone and a transposed convolution decoder.
        """
        super(RepMonoSupervisedModel, self).__init__()

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


class RepMonoUnsupervisedModel(BaseDepthModel):
    """
    RepMono model implementation with forward pass.
    """

    def __init__(self, in_channels, height, width, depth_scale=1.0):
        """
        Initializes the RepMonoModel with a ResNet-like backbone and a transposed convolution decoder.
        """
        super(RepMonoUnsupervisedModel, self).__init__()
        self.batch_size = 4
        self.height = height
        self.width = width
        self.scales = [0, 1, 2]

        self.encoder = RepMonoEncoder(in_channels=in_channels,
                                      height=height,
                                      width=width,
                                      depth_scale=depth_scale)
        # PUT USE SKIPS TO FALSE because shit was crashing otherwise
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                    use_skips=False, scales=self.scales)
        self.pose_encoder = ResnetEncoder(18, "pretrained", 2)
        self.pose_decoder = PoseDecoder(num_ch_enc=self.pose_encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=2)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = 480 // (2**scale)   #HARDCODE
            w = 640 // (2**scale)   #HARDCODE

            self.backproject_depth[scale] = BackprojectDepth(
                self.batch_size, h, w)    #HARDCODE
            self.backproject_depth[scale].to("cuda")

            self.project_3d[scale] = Project3D(self.batch_size, h, w) #HARDCODE
            self.project_3d[scale].to("cuda")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of RepMono model.

        Args:
            x (Tensor): Input RGB image tensor of shape (B, 3, H, W).

        Returns:
            Tensor: Predicted depth map tensor of shape (B, 1, H, W).
        """
        print(x.keys())
        features = self.encoder(x["image", 0, 0])
        depth_outputs = self.decoder(features)
        depth_outputs.update(self._predict_poses(x))
        self._generate_images_pred(x, depth_outputs)
        return depth_outputs

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

            _, depth = disp_to_depth(disp, 0.001, #HARDCODE
                                     1.0)   #HARDCODE

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
