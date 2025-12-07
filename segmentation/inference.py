"""
Inference entry point for segmentation models.

Allows inference on specific samples from the dataset.
"""
import torch
import numpy as np

from dataset.dataset import SmallDataset
from segmentation.utils import determine_model
from utils.visual import separate_visual
from utils.metrics import evaluate_metrics, threshold_pred


### PARAMETERS
# Dataset
images_path = "dataset/data/img"
masks_path = "dataset/data/mask"
image_size = 224
threshold = 0.5 # Threshold for binary mask

# Model
saves_dir = "segmentation/saves/"
logs_dir = "segmentation/logs/"
model_type = "segformer"
model_name = "m1"
# Number of trained models (cross validation)
num_models = 3
# Inference on specific samples from the dataset (indexes start at 1)
samples = [1, 61] # Empty for full set

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    dataset = SmallDataset(images_path, masks_path, image_size)
    dataset.normalize_set()
    dataset.cross_validation_split(1)
    fold = dataset.folds[0]
    num_samples = len(fold)

    models = []
    for i in range(1, num_models + 1):
        model = determine_model(f"{model_name}_{i}", model_type, saves_dir, logs_dir, device)
        model.load_model()
        models.append(model)

    # If samples is empty, use all samples
    global samples
    if samples == []:
        samples = range(1, num_samples + 1)

    with torch.inference_mode():
        processed_samples = 0
        for i in samples:
            sample_predictions = []
            sample_metrics = {
                "pixel_auroc": [],
                "pixel_ap": [],
                "iou": [],
                "f1": []
            }

            img, gt_mask = fold[i - 1]
            # Add batch dimension
            img = img.to(device).unsqueeze(0)

            for model in models:
                prediction = torch.sigmoid(model(img)).detach().squeeze(0).cpu().numpy()
                sample_predictions.append(prediction)

                metrics = evaluate_metrics(prediction, gt_mask.numpy(), threshold)

                sample_metrics["pixel_auroc"].append(metrics["pixel_auroc"])
                sample_metrics["pixel_ap"].append(metrics["pixel_ap"])
                sample_metrics["iou"].append(metrics["iou"])
                sample_metrics["f1"].append(metrics["f1"])

            processed_samples += 1

            final_prediction = np.stack(sample_predictions, axis=0)
            final_prediction = np.mean(final_prediction, axis=0)

            final_metrics = {
                "pixel_auroc": sum(sample_metrics["pixel_auroc"]) / len(sample_metrics["pixel_auroc"]),
                "pixel_ap": sum(sample_metrics["pixel_ap"]) / len(sample_metrics["pixel_ap"]),
                "iou": sum(sample_metrics["iou"]) / len(sample_metrics["iou"]),
                "f1": sum(sample_metrics["f1"]) / len(sample_metrics["f1"])
            }

            pred_mask = threshold_pred(final_prediction, threshold)

            imgs = [img.squeeze(0).cpu().numpy().transpose(1, 2, 0),
                    gt_mask.numpy().transpose(1, 2, 0),
                    prediction.transpose(1, 2, 0),
                    pred_mask.transpose(1, 2, 0)]
            
            labels = [f"Input {samples[processed_samples-1]}", "Ground-Truth mask", "Prediction map", "Prediction mask"]

            print(f"""
                Sample {samples[processed_samples-1]} metrics:
                Pixel AUROC: {final_metrics["pixel_auroc"]}
                Pixel AP: {final_metrics["pixel_ap"]}
                IoU: {final_metrics["iou"]}
                F1: {final_metrics["f1"]}
                """)
            
            separate_visual(imgs, labels)

        



if __name__ == "__main__":
    main()