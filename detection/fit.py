"""
Entry point for creating, training and testing anomaly detection models.
"""
import logging

import torch
from torch.utils.data import DataLoader

from dataset.dataset import SmallDataset, Fold
from detection.utils import determine_model

SUPPORTED_MODELS = ["stfpm"]

### HYPERPARAMETERS
# Dataset
# Training patches
patches_path = "./dataset/patches/normal/img"
patch_size = 128
# Testing images
images_path = "./dataset/data/img"
masks_path = "./dataset/data/mask"
img_size = 224
batch_size = 4
val_ratio = 0.2         # Validation ratio (0.2 = 20% of the testing fold will be taken for validation)
# Model
saves_dir = "./detection/saves/"      # Directory for saving models
model_name = "m8"
# Type is also applied as model's save suffix
model_type = "stfpm"
load_pretrained = True     # Load pretrained model from save
save_model = False
# Training
skip_training = True
epochs = 100                # Number of training epochs
learning_rate = 1e-3
eta_min = 1e-5              # Minimum learning rate (scheduler)
val_interval = 1           # Interval at which to validate [epochs] (also file logging interval)
augment = True
# Logging interval = val_interval
logs_dir = "./detection/logs/"
# Testing
skip_testing = False
display_batches = []        # Batches that will be displayed

device = "cuda" if torch.cuda.is_available() else "cpu"

metrics = {
    "pixel_auroc": [],
    "pixel_ap": [],
    "iou": [],
    "f1": []
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    ### DATASETS
    # Train (patches)
    train_set = SmallDataset(patches_path, size=patch_size)
    train_set.normalize_set()
    train_set.cross_validation_split(1)
    train_fold = train_set.folds[0]
    # Do not augment for memory banks
    if model_type == "padim" or model_type == "patchcore":
        train_fold.augment = False
    else:
        train_fold.augment = augment
    # Test (images)
    test_set = SmallDataset(images_path, masks_path, img_size)
    test_set.normalize_set()
    test_set.cross_validation_split(1)
    test_fold = test_set.folds[0]

    if val_ratio is None or val_ratio == 0:
        val_fold = []
        val_loader = None
    else:
        val_fold, test_fold = Fold.split_fold(test_fold, val_ratio)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_fold, batch_size=batch_size, shuffle=True)

    logging.info(f"""Datasets loaded.
                 Number of training patches: {len(train_fold)}
                 Patch size: {patch_size}x{patch_size}
                 Number of validation images: {len(val_fold)}
                 Number of testing images: {len(test_fold)} 
                 Image size: {img_size}x{img_size}                
                 """)
    
    ### TRAINING and TESTING
    try:
        model = determine_model(model_name, model_type, saves_dir, logs_dir, device)
        model.patch_size = patch_size
        model.image_size = img_size
        
        if load_pretrained:
            model.load_model()
        else:
            model.create_model()

        if not skip_training:
            # PaDiM and PatchCore train without optimizer
            if model_type == "padim" or model_type == "patchcore":
                model.train(train_loader)
            # Standard training
            else:
                model.train(train_loader, val_loader, epochs, val_interval, learning_rate, eta_min)

        if not skip_testing:
            metrics = model.test(test_loader, display_batches)

        if save_model:
            model.save_model()

    # Premature training exit
    except KeyboardInterrupt:
        logging.info("""
        *************************
        * Training interrupted. *
        *************************
                        """)
        if input("Do you want to save the model? (y/n): ").lower() == "y":
            model.save_model()
        return
    




if __name__ == "__main__":
    main()