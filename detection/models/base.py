"""
Base class for anomaly detection models.
"""
import os
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch

from utils.visual import separate_visual
from utils.metrics import evaluate_metrics, threshold_pred
from utils.logger import FileLogger
from detection.utils import extract_patches, reconstruct_from_patches

class BaseModel(torch.nn.Module, ABC):
    def __init__(self,
                 model_name: str,
                 model_type: str,
                 saves_dir: str ="./saves/",
                 logs_dir: str ="./logs/",
                 device: str ="cpu"):
        """
        BaseModel for anomalib anomaly detection models. Default set-up for working patch size 64 and image size 224. 

        Args:
            model_name (str): The name of the model (save name).
            model_type (str): The type of the model (e.g. "patchcore").
            saves_dir (str, optional): The directory of models saves. Defaults to "./saves/".
            logs_dir (str, optional): The directory of logs. Defaults to "./logs/".
            device (str, optional): The device where the model will be loaded. Defaults to "cpu".
        """
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.image_size = 224
        self.patch_size = 64
        self.saves_dir = saves_dir
        self.device = device
        self.model = None

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        self.file_log = FileLogger(os.path.join(logs_dir, f"{model_name + "_" + model_type}.txt"))

        if not os.path.exists(self.saves_dir):
            os.makedirs(self.saves_dir)
        self.model_path = os.path.join(saves_dir, f"{model_name + "_" + model_type}.pth")


    @abstractmethod
    def create_model(self, backbone: str):
        pass

    @abstractmethod
    def load_model(self, backbone: str):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        # Different models have different training methods
        pass


    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

        logging.info(f"Model saved as {self.model_path}.")


    def forward(self, image: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            Output depends of model type.
        """
        return self.model(image)
    

    def test(self,
             dataloader: torch.utils.data.DataLoader,
             display_batches: list[int] = None,
             ) -> dict[str, float]:
        """
        Test the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the testing data.
            display_batches (list[int], optional): A list of batch indices that will be displayed. Defaults to None.

        Raises:
            ValueError: If the model does not exist yet.
        """
        if not self.model:
            raise ValueError("Model not exists. Create or load model first.")
        
        msg = f"""Begin testing
                     Model type: {self.model_type}
                     Model name: {self.model_name}
                     Batch size: {dataloader.batch_size}
                     Testing set size: {len(dataloader.dataset)}
                     Number of batches {len(dataloader)}
                     Batches that will be displayed: {display_batches}
                     """
        logging.info(msg)
        self.file_log.log(msg)

        self.model.to(self.device)
        self.model.eval()
        
        metrics = self._eval_with_metrics(dataloader, display_batches)
            
        msg = f"""Testing
                Pixel AUROC: {metrics["pixel_auroc"]}
                Pixel AP: {metrics["pixel_ap"]}
                IoU: {metrics["iou"]}
                F1: {metrics["f1"]}
                """
        
        logging.info(msg)
        self.file_log.log(msg)
        
        return metrics
    

    def _eval_with_metrics(self,
                           dataloader: torch.utils.data.DataLoader,
                           show_batches: list[int] =None) -> dict[str, float]:
        """
        Evaluate the model on the given dataloader and return metrics.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the testing data.
            show_batches (list[int], optional): A list of batch indices that will be displayed. Defaults to None.

        Returns:
            dict[str, float]: A dictionary containing the metrics of the model.
        """
        self.model.eval()

        with torch.inference_mode():
            # Metrics
            metrics = {"pixel_auroc": 0.0,
                    "pixel_ap": 0.0,
                    "iou": 0.0,
                    "f1": 0.0
                    }
            # Threshold for binary mask
            threshold = 0.3

            for i, (images, masks) in tqdm(enumerate(dataloader), desc="Testing"):
                # Keep original for plotting
                orig_images = images.numpy()
                # Add channels (repeat) for backbone
                images = images.repeat(1, 3, 1, 1).to(self.device)
                masks = masks.numpy()

                # Per sample handling
                for j in range(images.shape[0]):
                    img_patches, patches_positions = extract_patches(images[j], self.patch_size)
                    output = self.model(img_patches)

                    patches_maps = output.anomaly_map.cpu()

                    prediction_map = reconstruct_from_patches(patches_maps, patches_positions, self.image_size).numpy()

                    sample_metrics = evaluate_metrics(prediction_map, masks[j], threshold)
                    # Invalid sample (empty mask)
                    if sample_metrics is None: continue

                    metrics["pixel_auroc"] += sample_metrics["pixel_auroc"]
                    metrics["pixel_ap"] += sample_metrics["pixel_ap"]
                    metrics["iou"] += sample_metrics["iou"]
                    metrics["f1"] += sample_metrics["f1"]

                    # Handles list or None
                    if i in (show_batches or []):
                        input = orig_images[j].transpose(1, 2, 0)
                        mask = masks[j].transpose(1, 2, 0)
                        raw = prediction_map.transpose(1, 2, 0)
                        thresholded = threshold_pred(raw, threshold)

                        separate_visual([input, mask, raw, thresholded], ["Input image", "Ground truth mask", "Prediction map", "Thresholded mask"])

            # Average metrics
            metrics["pixel_auroc"] /= len(dataloader.dataset)
            metrics["pixel_ap"] /= len(dataloader.dataset)
            metrics["iou"] /= len(dataloader.dataset)
            metrics["f1"] /= len(dataloader.dataset)

        return metrics