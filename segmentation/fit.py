"""
Entry point for creating, training and testing segmentation models.
"""
import logging

import torch
from torch.utils.data import DataLoader

from dataset.dataset import SmallDataset, Fold
from segmentation.utils import determine_model


SUPPORTED_MODELS = ["nested_unet", "dpt", "segformer"]  # Model type
### HYPERPARAMETERS

# Dataset
images_path = "./dataset/data/img"
masks_path = "./dataset/data/mask"
size = 224
# 1 folds will be used for validation and testing
folds_num = 3       # Number of folds for cross validation
# Set to 0 or None not use validation
val_ratio = 0.4     # Validation ratio (0.2 = 20% of the testing fold will be taken for validation)
batch_size = 4
# Model
saves_dir = "./segmentation/saves/"      # Directory for saving models
model_name = "m1"
# Type is also applied as model's save suffix
model_type = "nested_unet"  
load_pretrained = True     # Load pretrained model from save
save_model = False  
# Training
skip_training = True
epochs = 300                 # Number of training epochs
learning_rate = 1e-4   
eta_min = 1e-5              # Minimum learning rate (scheduler)
val_interval = 50           # Interval at which to validate [epochs] (also file logging interval)
augmentation = True
# Logging interval = val_interval
logs_dir = "./segmentation/logs/"
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
    ### DATASET
    dataset = SmallDataset(images_path, masks_path, size)
    dataset.normalize_set()
    dataset.cross_validation_split(folds_num)
    folds = dataset.folds

    logging.info(f"""Dataset loaded.
                 Number of samples: {len(dataset)}
                 Number of folds: {folds_num}
                 Image size: {size}x{size}
                 Augmentation: {augmentation}
                 Validation ratio: {val_ratio * 100}%
                 """)

    ### TRAINING and TESTING
    try:
        for i in range(folds_num):
            logging.info(f"""
                         *************
                         * Model {i + 1}  *
                         *************
                         """)
            
            # Model
            current_name = f"{model_name}_{i + 1}"
            model = determine_model(current_name, model_type, saves_dir, logs_dir, device)
            if load_pretrained:
                model.load_model()
            else:
                model.create_model()

            # Dataloaders
            test_fold = folds[i]
            if val_ratio is None or val_ratio == 0:
                val_loader = None
            else:
                # Training with validation
                val_fold, test_fold = Fold.split_fold(test_fold, val_ratio)

                val_loader = DataLoader(val_fold, batch_size=batch_size)

            test_loader = DataLoader(test_fold, batch_size=batch_size)

            train_fold = folds[:i] + folds[i+1:]
            train_fold = Fold.combine_folds(train_fold)
            # Set augmentation
            train_fold.augment = augmentation

            train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
           
            # Training
            if not skip_training:
                model.train(train_loader, val_loader, epochs, val_interval, learning_rate, eta_min)

            # Testing
            if not skip_testing:
                model_metrics = model.test(test_loader, display_batches)

                metrics["pixel_auroc"].append(model_metrics["pixel_auroc"])
                metrics["pixel_ap"].append(model_metrics["pixel_ap"])
                metrics["iou"].append(model_metrics["iou"])
                metrics["f1"].append(model_metrics["f1"])

            # Save
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

    ### RESULTS
    if not skip_testing:
        result_mes = """
                     ***********
                     * Summary *
                     ***********
                    """
        
        for i in range(folds_num):
            result_mes += f"""
                         ***********
                         * Model {i + 1} *
                         ***********

                         Pixel AUROC: {metrics["pixel_auroc"][i]}
                         Pixel AP: {metrics["pixel_ap"][i]}
                         IoU: {metrics["iou"][i]}
                         F1: {metrics["f1"][i]}
                         """
            
        result_mes += f"""
                     ***********
                     * Overall *
                     ***********

                     Pixel AUROC: {sum(metrics["pixel_auroc"]) / folds_num}
                     Pixel AP: {sum(metrics["pixel_ap"]) / folds_num}
                     IoU: {sum(metrics["iou"]) / folds_num}
                     F1: {sum(metrics["f1"]) / folds_num}
                     """
        
        logging.info(result_mes)





if __name__ == "__main__":
    main()