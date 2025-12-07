"""
Custom pytorch friendly dataset. 

SmallDataset class - loads images and masks from the given paths
                   - resizes and pads them to the given size
                   - splits the dataset into folds for cross-validation

Fold class - actual pytorch dataset
"""
from __future__ import annotations
import numpy as np
import os
from PIL import Image
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class SmallDataset:
    def __init__(self, images_path: str, masks_path: str =None, size: int =256):
        """
        Constructor for custom PyTorch Dataset. Loads images and masks from the given paths.

        Args:
            images_path (str): Path to the images directory.
            masks_path (str, optional): Path to the masks directory, if not provided, empty masks will be created. Defaults to None.
            size (int, optional): Size of the images to resize to. Defaults to 256.

        Raises:
            ValueError: If the number of images and masks is different.
        """
        self.images = SmallDataset.load_images_to_list(images_path, size=size)
        if masks_path:
            self.masks = SmallDataset.load_images_to_list(masks_path, size=size)
        else:
            # Create empty masks
            self.masks = [np.zeros((size, size), dtype=np.uint8) for _ in range(len(self.images))]

        if len(self.images) != len(self.masks):
            raise ValueError("Images and masks must have the same length.")
        
        # Convert from list to numpy
        self.images = np.stack(self.images)
        self.masks = np.stack(self.masks)
        
        self.folds = []

    # For normalization
    # ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)
    MEAN = 0.485
    STD = 0.229

    def __len__(self) -> int:
        return self.images.shape[0]


    @staticmethod
    def load_images_to_list(path: str, size: int =256) -> list[np.ndarray]:
        """
        Loads images from the given path into a list, resizes and pads them to the given size.

        Args:
            path (str): Path to the images directory.
            size (int, optional): Size of the images to resize to. Defaults to 256.

        Returns:
            list[np.ndarray]: List of resized and padded images.
        
        Raises:
            ValueError: If the given path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        out = []

        for filename in sorted(os.listdir(path)):
            if filename.endswith(".png"):
                file_path = os.path.join(path, filename)
                # Load grayscale image
                image = np.array(Image.open(file_path).convert("L"))

                image = SmallDataset.resize_and_pad(image, size=size)

                out.append(np.array(image))
        return out
    

    @staticmethod
    def resize_and_pad(img: np.ndarray, size: int =256) -> np.ndarray:
        """
        Resizes the given image to the given size, and pads it to fit the size if necessary.

        Args:
            img (np.ndarray): The image to be resized and padded.
            size (int, optional): The size to resize the image to. Defaults to 256.

        Returns:
            np.ndarray: The resized and padded image.

        Raises:
            ValueError: If the given image is not a 2D array.
        """
        height, width = img.shape
        scale = size / max(height, width)
        new_w, new_h = int(width * scale), int(height * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        new_img = np.full((size, size), 0, dtype=img.dtype)
        
        # Center the resized image
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        new_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return new_img


    def normalize_set(self):
        """
        Normalize the images and masks.
        """
        self.images = self.images / 255
        self.images = (self.images - self.MEAN) / self.STD

        self.masks = self.masks / 255


    def cross_validation_split(self, folds: int =1):
        """
        Split the dataset into folds for cross-validation.

        Args:
            folds (int, optional): The number of folds to split the dataset into. Defaults to 1.
        """
        if folds == 1:
            fold_images = torch.from_numpy(self.images).float()
            fold_masks = torch.from_numpy(self.masks).float()
            self.folds.append(Fold(fold_images, fold_masks))
            return

        samples_num = len(self)
        splits = np.array_split(np.arange(samples_num), folds)

        for fold_indices in splits:
            fold_images = torch.from_numpy(self.images[fold_indices]).float()
            fold_masks = torch.from_numpy(self.masks[fold_indices]).float()
            self.folds.append(Fold(fold_images, fold_masks))

            
class Fold(Dataset):
    def __init__(self, images: torch.Tensor, masks: torch.Tensor, augment: bool =False):
        """
        Fold class prepared for pytorch dataloader.

        Args:
            images (torch.Tensor): The images of the fold.
            masks (torch.Tensor): The masks of the fold.
            augment (bool, optional): Whether to apply data augmentation to the sample. Defaults to False.
        """
        # Add a channel dimension
        self.images = images.unsqueeze(1).float()
        self.masks = masks.unsqueeze(1).long()

        self.augment = augment


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment:
            # Apply data augmentation
            return self.transform(self.images[index], self.masks[index])
        else:
            return self.images[index], self.masks[index]
    

    def transform(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies random transformations to the given image and mask.

        Args:
            image (torch.Tensor): The image to be transformed.
            mask (torch.Tensor): The mask to be transformed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed image and mask.
        """
        # Geometry
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.5:
            # Normalized black
            black_norm = (0 - SmallDataset.MEAN) / SmallDataset.STD
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle, fill=black_norm)
            mask = TF.rotate(mask, angle, fill=0)

        # Pixel level
        if random.random() < 0.5:
            # Brightness for normalized image
            factor = random.uniform(0.8, 1.2)
            offset = (SmallDataset.MEAN * (factor - 1)) / SmallDataset.STD
            image = image * factor + offset

        if random.random() < 0.5:
            # Contrast for normalized image
            factor = random.uniform(0.8, 1.2)
            offset = (SmallDataset.MEAN * (1 - factor)) / SmallDataset.STD
            image = image * factor + offset

        # Gaussian noise
        if random.random() < 0.5:
            image = image + torch.randn_like(image) * 0.02  # std = 0.02

        # Keep image normalized
        min_val = (0.0 - SmallDataset.MEAN) / SmallDataset.STD
        max_val = (1.0 - SmallDataset.MEAN) / SmallDataset.STD
        image = image.clamp(min_val, max_val)

        return image, mask
    

    @staticmethod
    def combine_folds(folds: list[Fold]) -> Fold:
        """
        Combine multiple folds into one.

        Args:
            folds (list[Fold]): A list of folds to combine.

        Returns:
            Fold: A new fold containing all the images and masks from the input folds.
        """
        images = torch.cat([fold.images for fold in folds], dim=0).squeeze(1)
        masks = torch.cat([fold.masks for fold in folds], dim=0).squeeze(1)

        return Fold(images, masks)
    

    @staticmethod
    def split_fold(fold: Fold, ratio: float=0.5) -> tuple[Fold, Fold]:
        """
        Split a fold into two folds based on the given ratio.

        Args:
            fold (Fold): The fold to be split.
            ratio (float, optional): The ratio of the first fold to the total size. Defaults to 0.5.

        Returns:
            tuple[Fold, Fold]: A tuple containing the two folds resulting from the split.
        """

        images = fold.images.squeeze(1)
        masks = fold.masks.squeeze(1)

        size = int(len(images) * ratio)

        return Fold(images[:size], masks[:size]), Fold(images[size:], masks[size:])
