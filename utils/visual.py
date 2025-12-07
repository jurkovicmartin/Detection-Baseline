"""
Images and maps visualization functions.

Includes:
    - separate_visual: Visualize multiple images in separate subplots.
    - overlap_visual: Visualize the image with the mask overlapping on top of it.
"""
import matplotlib.pyplot as plt
import numpy as np

def separate_visual(images: list, titles: list):
    """
    Visualize multiple images in separate subplots.

    Args:
        images (list): List of images to visualize.
        titles (list): List of titles for each image.

    Raises:
        ValueError: If the number of images and titles is not the same.
    """
    num_images = len(images)

    if num_images != len(titles):
        raise ValueError("Number of images and titles must be the same.")
    
    fig, axes = plt.subplots((num_images + 1) // 2, 2, figsize=(8, 4))

    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()



def overlap_visual(img: np.ndarray, mask: np.ndarray, title: str = "Image"):
    """
    Visualize the image with the mask overlapping on top of it.

    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Input mask.
        title (str, optional): Title for the plot. Defaults to "Image".
    """
    
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.imshow(mask, cmap="Reds", alpha=0.5)
    ax.axis("off")
    plt.title(title)
    plt.show()

