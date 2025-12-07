import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
from pathlib import Path
from tqdm import tqdm


class PatchesExtractor:
    def __init__(self,
                 patch_size: int,
                 min_anomaly_ratio: float,
                 normal_ratio: float,
                 background_threshold: int,
                 bright_fraction_threshold: float):
        """
        Constructor for PatchesExtractor object.

        Parameters:
            patch_size (int): Size of the patches to be extracted.
            min_anomaly_ratio (float): Minimum fraction of anomaly pixels in an anomaly patch.
            normal_ratio (float): How many normal patches per anomaly patch.
            background_threshold (int): Threshold for what is considered background.
            bright_fraction_threshold (float): Threshold for what part of the patch should be bright.
        """
        self.patch_size = patch_size
        self.min_anomaly_ratio = min_anomaly_ratio
        self.normal_ratio = normal_ratio
        self.background_threshold = background_threshold
        self.bright_fraction_threshold = bright_fraction_threshold


    def extract(self, image: np.ndarray, mask: np.ndarray) -> tuple[list, list, list, list]:
        """
        Extract patches from the given image and mask.

        Parameters:
            image (np.ndarray): The image to extract patches from.
            mask (np.ndarray): The mask to extract patches from.

        Returns:
            tuple[
            list[np.ndarray]: Anomaly patches
            list[np.ndarray]: Anomaly masks
            list[np.ndarray]: Normal patches
            list[np.ndarray]: Normal masks
            ]
            
        """
        H, W = mask.shape
        half = self.patch_size // 2

        anomaly_img_patches = []
        anomaly_mask_patches = []
        normal_img_patches = []
        normal_mask_patches = []

        # Anomaly patches
        labeled = label(mask)
        for region in regionprops(labeled):
            cy, cx = map(int, region.centroid)

            y1 = np.clip(cy - half, 0, H - self.patch_size)
            x1 = np.clip(cx - half, 0, W - self.patch_size)
            y2 = y1 + self.patch_size
            x2 = x1 + self.patch_size

            img_patch = image[y1:y2, x1:x2]
            msk_patch = mask[y1:y2, x1:x2]

            if (msk_patch > 0).mean() >= self.min_anomaly_ratio:
                anomaly_img_patches.append(img_patch)
                anomaly_mask_patches.append(msk_patch * 255)

        # Normal patches
        n_target = int(len(anomaly_img_patches) * self.normal_ratio)
        attempts = 0

        while len(normal_img_patches) < n_target and attempts < n_target * 20:
            attempts += 1

            y = np.random.randint(0, H - self.patch_size)
            x = np.random.randint(0, W - self.patch_size)

            img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
            msk_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

            bright_fraction = np.mean(img_patch > self.background_threshold)
            if msk_patch.sum() == 0 and bright_fraction > self.bright_fraction_threshold:
                normal_img_patches.append(img_patch)
                normal_mask_patches.append(msk_patch * 255)

        return anomaly_img_patches, anomaly_mask_patches, normal_img_patches, normal_mask_patches



def main():
    ### PARAMETERS
    images_dir = "dataset/data/img"
    masks_dir = "dataset/data/mask"

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory {images_dir} does not exist.")

    normal_patches_dir = "dataset/patches/normal"
    anomaly_patches_dir = "dataset/patches/anomaly"

    if not os.path.exists(normal_patches_dir):
        os.makedirs(normal_patches_dir)
        os.makedirs(normal_patches_dir + "/img")
        os.makedirs(normal_patches_dir + "/mask")
    if not os.path.exists(anomaly_patches_dir):
        os.makedirs(anomaly_patches_dir)
        os.makedirs(anomaly_patches_dir + "/img")
        os.makedirs(anomaly_patches_dir + "/mask")

    patch_size = 256
    # Minimum fraction of anomaly pixels in an anomaly patch
    min_anomaly_ratio = 0.05
    # How many normal patches per anomaly patch      
    normal_ratio = 2.0
    # Threshold for what is considered background       
    background_threshold = 10
    # Threshold for what part of the patch should be bright
    bright_fraction_threshold = 0.5


    ### PATCH EXTRACTION
    extractor = PatchesExtractor(patch_size,
                                 min_anomaly_ratio,
                                 normal_ratio,
                                 background_threshold,
                                 bright_fraction_threshold)

    image_files = sorted(Path(images_dir).glob("*.png"))

    anomaly_patch_id = 0
    normal_patch_id = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        fname = img_path.name
        mask_path = os.path.join(masks_dir, fname)
        if not os.path.exists(mask_path):
            print(f"Missing mask for {fname}")
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Ensure binary mask
        mask = (mask > 0).astype(np.uint8)

        anomaly_images, anomaly_masks, normal_images, normal_masks = extractor.extract(image, mask)

        for img, mask in zip(anomaly_images, anomaly_masks):
            img_path = os.path.join(anomaly_patches_dir, "img", f"{anomaly_patch_id:04d}.png")
            mask_path = os.path.join(anomaly_patches_dir, "mask", f"{anomaly_patch_id:04d}.png")

            if not cv2.imwrite(img_path, img):
                print(f"Cannot save image {img_path}")
            if not cv2.imwrite(mask_path, mask):
                print(f"Cannot save mask {mask_path}")

            anomaly_patch_id += 1

        for img, mask in zip(normal_images, normal_masks):
            img_path = os.path.join(normal_patches_dir, "img", f"{normal_patch_id:04d}.png")
            mask_path = os.path.join(normal_patches_dir, "mask", f"{normal_patch_id:04d}.png")

            if not cv2.imwrite(img_path, img):
                print(f"Cannot save image {img_path}")
            if not cv2.imwrite(mask_path, mask):
                print(f"Cannot save mask {mask_path}")

            normal_patch_id += 1

    print(f"Extracted {anomaly_patch_id} anomalous patches to {anomaly_patches_dir}.")
    print(f"Extracted {normal_patch_id} normal patches to {normal_patches_dir}.")
    

if __name__ == "__main__":
    main()
