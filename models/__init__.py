from torch import nn as nn
from models.guide_depth import GuideDepthModel
from models.repmono_depth import RepMonoSupervisedModel, RepMonoUnsupervisedModel

SUPPORTED_MODELS = {
    "guidedepth": GuideDepthModel,
    "repmono-u": RepMonoUnsupervisedModel,
    "repmono-s": RepMonoSupervisedModel
}


def get_model(model_name: str, *args, **kwargs) -> nn.Module:
    """
    Returns the model class based on the given name.
    :param model_name: Name of the model (e.g., "guidedepth", "repmono").
    :return: Model class if found, else raises an error.
    """
    if model_name.lower() in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name](**kwargs)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Choose from {list(SUPPORTED_MODELS.keys())}."
        )
