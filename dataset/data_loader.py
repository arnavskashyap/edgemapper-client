from dataset.transforms import Transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class ImageDataset:
    """
    Loads images and applies model-specific transformations.
    """

    def __init__(self, dataset_path: str, model_name: str, val: bool):
        self.transforms = Transforms.get_transforms(model_name, val)
        # TODO: Need to fix this for whatever file formats the images are in. We may want to use something like TransformedDataset in dataset.py.
        self.dataset = ImageFolder(root=dataset_path,
                                   transform=self.transforms)

    def get_dataloader(self,
                       batch_size: int,
                       shuffle: bool = False,
                       drop_last: bool = True):
        return DataLoader(self.dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last)
