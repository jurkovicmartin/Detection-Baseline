"""
Base class for segmentation models.
"""
import logging
from tqdm import tqdm
import os
from abc import ABC, abstractmethod

import torch
import segmentation_models_pytorch as smp

from utils.visual import separate_visual
from utils.metrics import evaluate_metrics, threshold_pred
from utils.logger import FileLogger


class BaseModel(torch.nn.Module, ABC):
    def __init__(self,
                 model_name: str,
                 model_type: str,
                 saves_dir: str ="./saves/",
                 logs_dir: str ="./logs/",
                 device: str ="cpu"):
        """
        Abstract BaseModel object.

        Args:
            model_name (str): The name of the model (save name).
            model_type (str): The type of the model (e.g. "segformer").
            saves_dir (str, optional): The directory of models saves. Default is "./saves/".
            logs_dir (str, optional): The directory of logs. Default is "./logs/".
            device (str, optional): The device where the model will be loaded. Default is "cpu".
        """
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.saves_dir = saves_dir
        self.device = device
        self.model = None

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        self.file_log = FileLogger(os.path.join(logs_dir, f"{model_name + "_" + model_type}.txt"))

        if not os.path.exists(self.saves_dir):
            os.makedirs(self.saves_dir)
        self.model_path = os.path.join(saves_dir, f"{model_name + "_" + model_type}")


    @abstractmethod
    def create_model(self, encoder: str):
        pass


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output mask tensor.
        """
        return self.model(image)
    

    def load_model(self):
        """
        Load a pre-trained model from saves directory.

        Raises:
            FileNotFoundError: If the model at the given path does not exist.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model {self.model_path} not found.")
        
        self.model = smp.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

        logging.info(f"Model {self.model_path} successfully loaded to {self.device} device.")


    def save_model(self):
        """
        Save the model to the saves directory.

        Raises:
            ValueError: If the model does not exist yet.
        """
        if not self.model:
            raise ValueError("Model not exists. Create or load model first.")

        self.model.save_pretrained(self.model_path)

        logging.info(f"Model saved as {self.model_path}.")


    def train(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader =None,
              epochs: int =1,
              val_interval: int =1,
              learning_rate: float =1e-3,
              eta_min: float =1e-5,
              ):
        """
        Train the model using the given dataloader.

        Args:
            train_loader (DataLoader): DataLoader containing the training data.
            val_loader (DataLoader): DataLoader containing the validation data.
            epochs (int, optional): Number of epochs to train the model. Defaults to 1.
            val_interval (int, optional): Interval at which to validate the model. Defaults to 10.
            learning_rate (float, optional): Learning rate for the AdamW optimizer. Defaults to 1e-3.
            eta_min (float, optional): Minimum learning rate for the scheduler. Defaults to 1e-5.

        Raises:
            ValueError: If the model does not exist yet.
        """
        if not self.model:
            raise ValueError("Model not exists. Create or load model first.")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        iterations = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min)

        loss_fn = smp.losses.DiceLoss(mode="binary")

        msg = f"""Begin training
                     Model type: {self.model_type}
                     Model name: {self.model_name}
                     Epochs: {epochs}
                     Batch size: {train_loader.batch_size}
                     Training set size: {len(train_loader.dataset)}
                     Learning rate: {learning_rate}
                     Eta min: {eta_min}
                     Validation set size: {len(val_loader.dataset)}
                     Validation interval: {val_interval}
                     """
        logging.info(msg)
        self.file_log.log(msg)

        self.model.to(self.device)

        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()

            train_loss = 0.0
            
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                # Logits
                outputs = self.model(images)

                loss = loss_fn(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()
            
            logging.info(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}")

            # Validation
            if not val_loader: continue
             
            if (epoch + 1) % val_interval == 0:
                self.model.eval()

                metrics = self._eval_with_metrics(val_loader)
                
                msg = f"""Validation
                        Pixel AUROC: {metrics["pixel_auroc"]}
                        Pixel AP: {metrics["pixel_ap"]}
                        IoU: {metrics["iou"]}
                        F1: {metrics["f1"]}
                        """

                logging.info(msg)
                self.file_log.log(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}" + msg)


        self.model.eval()


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

            for i, batch in enumerate(dataloader, 1):
                images, masks = batch
                images = images.to(self.device)
                masks = masks.numpy()

                # Logits converted to probabilities
                output = torch.sigmoid(self.model(images))

                output = output.cpu().numpy()

                # Metrics per sample
                for j in range(output.shape[0]):
                    sample_metrics = evaluate_metrics(output[j], masks[j])
                    # Invalid sample (empty mask)
                    if sample_metrics is None: continue

                    metrics["pixel_auroc"] += sample_metrics["pixel_auroc"]
                    metrics["pixel_ap"] += sample_metrics["pixel_ap"]
                    metrics["iou"] += sample_metrics["iou"]
                    metrics["f1"] += sample_metrics["f1"]

                # Handles list or None
                if i in (show_batches or []):
                    # Display the batch
                    for j in range(dataloader.batch_size):
                        input = images[j].cpu().numpy().transpose(1, 2, 0)
                        mask = masks[j].transpose(1, 2, 0)
                        raw = output[j].transpose(1, 2, 0)
                        thresholded = threshold_pred(raw)

                        separate_visual([input, mask, raw, thresholded], ["Input image", "Ground truth mask", "Raw mask", "Thresholded mask"])

            # Average metrics
            metrics["pixel_auroc"] /= len(dataloader.dataset)
            metrics["pixel_ap"] /= len(dataloader.dataset)
            metrics["iou"] /= len(dataloader.dataset)
            metrics["f1"] /= len(dataloader.dataset)

        return metrics
        