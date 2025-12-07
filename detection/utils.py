"""
Utils for detection models.

Include:
    - determine_model - determine the model to use based on provided type
    - extract_patches - extract patches from images
    - reconstruct_from_patches - reconstruct images from patches
"""
import torch
import math


def determine_model(name: str,
                    type: str,
                    saves_dir: str ="./saves/",
                    logs_dir: str ="./logs/",
                    device: str ="cpu"):
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
    # Lazy import to avoid circular import
    import detection.models as models

    if type == "stfpm":
        return models.STFPM(name, saves_dir, logs_dir, device)
    else:
        raise ValueError(f"Not supported model: {name}")


def extract_patches(image: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    Extract patches from the given image (overlapping).

    Parameters:
        image (torch.Tensor): The image to extract patches from.
        patch_size (int): The size of the patches to be extracted.

    Returns:
        tuple[
        torch.Tensor: The tensor containing the extracted patches.
        list[tuple[int, int]]: A list of tuples containing the positions of the patches in the image.
        ]
    """
    # Remove batch dim
    if image.dim() == 4:
        image = image[0]

    _, height, width = image.shape

    patches = []
    positions = []

    ys = _compute_positions(height, patch_size)
    xs = _compute_positions(width, patch_size)

    # Extract patches
    for y in ys:
        for x in xs:
            patch = image[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((y, x))

    return torch.stack(patches, dim=0), positions


def _compute_positions(image_size: int, patch_size: int) -> list[int]:
        """
        Computes positions of patches in an image.

        Parameters:
            image_size (int): Size of the image.
            patch_size (int): Size of the patches.

        Returns:
            list[int]: A list of positions of the patches in the image.
        """
        # Only one patch needed
        if image_size <= patch_size:
            return [0]

        num = math.ceil(image_size / patch_size)
        stride = (image_size - patch_size) / (num - 1)

        # Compute all positions
        return [int(round(i * stride)) for i in range(num)]


def reconstruct_from_patches(patches: torch.Tensor, positions: list[tuple[int, int]], image_size: int) -> torch.Tensor:
    """
    Reconstruct an image from overlapping patches.

    Parameters:
        patches (torch.Tensor): The tensor containing the overlapping patches.
        positions (list[tuple[int, int]]): A list of tuples containing the positions of the patches in the image.
        image_size (int): The size of the image to be reconstructed.

    Returns:
        torch.Tensor: The reconstructed image.

    """
    _, channel, patch_size, _ = patches.shape
    height = width = image_size

    output = torch.zeros((channel, height, width))
    # Weights to blend overlapping patches
    weight = torch.zeros((channel, height, width))

    # Create 2D blending Hann window
    w1d = torch.hann_window(patch_size)
    window = w1d[:, None] * w1d[None, :]  # outer product → (P, P)
    window = window / window.max()       # normalize to 1
    window = window.unsqueeze(0).repeat(channel, 1, 1)  # (C, P, P)

    for patch, (y, x) in zip(patches, positions):
        # multiply patch by window
        weighted_patch = patch * window

        output[:, y:y+patch_size, x:x+patch_size] += weighted_patch
        weight[:, y:y+patch_size, x:x+patch_size] += window
        
    # Avoid division by 0
    weight = torch.clamp(weight, min=1e-8)
    return output / weight


### TEST
# if __name__ == "__main__":
#     from PIL import Image
#     import numpy as np
#     from dataset.dataset import SmallDataset
#     from utils.visual import separate_visual

#     img_size = 224
#     patch_size = 128

#     img = np.array(Image.open("dataset/data/img/01.png").convert("L"))
#     img = SmallDataset.resize_and_pad(img, img_size)
#     # Add batch and channel dimension
#     img = img.reshape(1, 1, img_size, img_size)
#     img = torch.from_numpy(img)

#     patches, positions = extract_patches(img, patch_size)

#     # Show patches
#     patches_list = [patches[i][0] for i in range(patches.shape[0])]
#     labels = [f"Patch {i}" for i in range(patches.shape[0])]
#     separate_visual(patches_list, labels)

#     # Show images
#     rec_img = reconstruct_from_patches(patches, positions, img_size)
#     separate_visual([img[0][0], rec_img[0]], ["Original", "Reconstructed"])

