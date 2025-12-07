"""
Inference entry point for anomaly detection models.

Allows inference on specific samples from the dataset.
"""
import torch

from dataset.dataset import SmallDataset
from detection.utils import determine_model, extract_patches, reconstruct_from_patches
from utils.visual import separate_visual
from utils.metrics import evaluate_metrics, threshold_pred


### PARAMETERS
# Dataset
images_path = "./dataset/data/img"
masks_path = "./dataset/data/mask"
image_size = 224
patch_size = 128
threshold = 0.1 # Threshold for binary mask

# Model
saves_dir = "./detection/saves/"
logs_dir = "./detection/logs/"
model_type = "stfpm"
model_name = "m8"
# Inference on specific samples from the dataset (indexes start at 1)
samples = [] # Empty for full set

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    dataset = SmallDataset(images_path, masks_path, image_size)
    dataset.normalize_set()
    dataset.cross_validation_split(1)
    fold = dataset.folds[0]
    num_samples = len(fold)

    model = determine_model(model_name, model_type, saves_dir, logs_dir, device)
    model.load_model()

    global samples
    if samples == []:
        samples = range(1, num_samples + 1)

    with torch.inference_mode():
        for i in samples:
            img, gt_mask = fold[i - 1]
            orig_img = img.numpy()
            gt_mask = gt_mask.numpy()
            # Add batch dimension
            img = img.unsqueeze(0).repeat(1, 3, 1, 1).to(device)

            img_patches, patches_positions = extract_patches(img, patch_size)
            output = model(img_patches)

            patches_maps = output.anomaly_map.cpu()

            prediction_map = reconstruct_from_patches(patches_maps, patches_positions, image_size).numpy()

            metrics = evaluate_metrics(prediction_map, gt_mask, threshold)

            show = [
                orig_img.transpose(1, 2, 0),
                gt_mask.transpose(1, 2, 0),
                prediction_map.transpose(1, 2, 0),
                threshold_pred(prediction_map, threshold).transpose(1, 2, 0)
            ]

            labels = [f"Input {i}", "Ground-Truth mask", "Prediction map", "Prediction mask"]

            print(f"""
                Metrics:
                Pixel AUROC: {metrics["pixel_auroc"]}
                Pixel AP: {metrics["pixel_ap"]}
                IoU: {metrics["iou"]}
                F1: {metrics["f1"]}
                """)

            separate_visual(show, labels)



if __name__ == "__main__":
    main()