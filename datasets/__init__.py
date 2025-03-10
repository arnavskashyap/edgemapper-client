from torch import nn as nn
from datasets.nyu_dataset import NYUDataset
from datasets.guidedepth_dataset import GuideDepthDataset
from datasets.repmono_dataset import RepMonoUnsupervisedDataset
from datasets.transforms import Transforms

SUPPORTED_MODEL_DATASETS = {
    "guidedepth": GuideDepthDataset,
    "repmono-u": RepMonoUnsupervisedDataset,
    "repmono-s": NYUDataset,
    "hybrid": RepMonoUnsupervisedDataset,
    "guidedepth-t": GuideDepthDataset
}


def get_dataset(model_name: str, dataset_path: str, val: bool, *args,
                **kwargs) -> nn.Module:
    """
    Returns the dataset class based on the given name.
    :param model_name: Name of the model (e.g., "guidedepth", "repmono").
    :return: Dataset class if found, else raises an error.
    """
    if model_name.lower() in SUPPORTED_MODEL_DATASETS:
        transform = Transforms.get_transforms(model_name, val)
        return SUPPORTED_MODEL_DATASETS[model_name](dataset_path, val,
                                                    transform, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Choose from {list(SUPPORTED_MODEL_DATASETS.keys())}."
        )
