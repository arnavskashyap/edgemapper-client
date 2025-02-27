import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class NYUDataset(Dataset):
    """
    Loads images and applies model-specific transformations.
    """

    def __init__(self, dataset_path: str, val: bool, transform):
        super(NYUDataset, self).__init__()
        self.val = val
        self.transform = transform
        self.filenames = self._load_images(dataset_path)

    def _load_images(self, room_path):
        """Recursively search for matching RGB and depth images inside rooms."""
        images = []
        if os.path.isdir(room_path):
            rgb_images = sorted(
                [f for f in os.listdir(room_path) if f.endswith(".jpg")])
            gt_depths = [
                filename.replace(".jpg", ".png") for filename in rgb_images
            ]

            for rgb_image, gt_depth in zip(rgb_images, gt_depths):
                rgb_path = os.path.join(room_path, rgb_image)
                gt_path = os.path.join(room_path, gt_depth)
                paths = (rgb_path, gt_path)
                images.append(paths)

        return images

    def __len__(self):
        return len(self.filenames)

    def get_image(self, image_path):
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                return img

    def __getitem__(self, index):
        rgb_path, depth_path = self.filenames[index]
        image = self.get_image(rgb_path)
        depth = self.get_image(depth_path)

        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)

        return sample
