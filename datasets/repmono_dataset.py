import os
import random
from typing import List
import torch
from torchvision import transforms
import numpy as np

from loguru import logger
from PIL import Image

from datasets.transforms import Transforms
from datasets.nyu_dataset import NYUDataset


class RepMonoUnsupervisedDataset(NYUDataset):

    def __init__(self,
                 height: int = 480,
                 width: int = 640,
                 frame_idxs: List[int] = [0, -1, 1],
                 num_scales: int = 4,
                 *args,
                 **kwargs):
        super(NYUDataset, self).__init__()
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.K = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32)

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast,
                                              self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.interp = Image.Resampling.LANCZOS
        self.resize = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = transforms.ToTensor(f)
                inputs[(n + "_aug", im, i)] = transforms.ToTensor(color_aug(f))

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("image", <frame_id>, <scale>)          for raw colour images,
            ("image_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_image_aug = not (self.val) and random.random() > 0.5
        do_flip = not (self.val) and random.random() > 0.5

        rgb_path, _ = self.filenames[index]
        frame_index = os.path.splitext(os.path.basename(rgb_path))[0]
        room_path = os.path.dirname(rgb_path)

        for i in self.frame_idxs:
            new_frame_path = os.path.join(room_path, f"{frame_index + i}.jpg")
            new_frame = self.get_image(new_frame_path)
            if do_flip:
                new_frame = new_frame.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT)
            inputs[("image", i, -1)] = new_frame

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2**scale)
            K[1, :] *= self.height // (2**scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_image_aug:
            image_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            image_aug = (lambda x: x)

        self.preprocess(inputs, image_aug)

        for i in self.frame_idxs:
            del inputs[("image", i, -1)]
            del inputs[("image_aug", i, -1)]

        gt_path = os.path.join(room_path, f"{frame_index + i}.png")
        depth_gt = self.get_image(gt_path)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        inputs["depth"] = np.expand_dims(depth, 0)
        inputs["depth"] = torch.from_numpy(inputs["depth_gt"].astype(
            np.float32))

        return inputs
