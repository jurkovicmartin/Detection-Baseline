"""
Utils for segmentation models.
"""
from segmentation.base import BaseModel
from segmentation.models import *

def determine_model(name: str,
                    type: str,
                    saves_dir: str ="./saves/",
                    logs_dir: str ="./logs/",
                    device: str ="cpu") -> BaseModel:
    """
    Determine the model to use based on the given type.

    Args:
        name (str): The name of the model.
        type (str): The type of the model.
        saves_dir (str, optional): The directory of models saves. Defaults to "./saves/".
        logs_dir (str, optional): The directory of logs. Defaults to "./logs/".
        device (str, optional): The device to use for training. Defaults to "cpu".

    Returns:
        BaseModel subclass: The model to use.

    Raises:
        ValueError: If the given type is not supported.
    """
    if type == "nested_unet":
        return NestedUNet(name, saves_dir, logs_dir, device)
    elif type == "segformer":
        return SegFormer(name, saves_dir, logs_dir, device)
    elif type == "dpt":
        return DPT(name, saves_dir, logs_dir, device)
    else:
        raise ValueError(f"Not supported model: {name}")