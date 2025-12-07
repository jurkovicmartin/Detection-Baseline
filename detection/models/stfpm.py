"""
STFPM model implementation.
"""
import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from anomalib.models.image.stfpm.torch_model import STFPMModel
from anomalib.models.image.stfpm.loss import STFPMLoss

from detection.models.base import BaseModel


class STFPM(BaseModel):
    def __init__(self,
                 model_name: str,
                 saves_dir: str ="./saves/",
                 logs_dir: str ="./logs/",
                 device: str ="cpu"):        
        super().__init__(model_name, "stfpm", saves_dir, logs_dir, device)

        self.backbone = "resnet18"


    def create_model(self):
        self.model = STFPMModel(["layer1", "layer2", "layer3"], self.backbone).to(self.device)

        logging.info(f"""Created model:
                     Model: STFPM
                     Backbone: {self.backbone}
                     Device: {self.device}
                     """)

    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model {self.model_path} not found.")
        
        self.model = STFPMModel(["layer1", "layer2", "layer3"], self.backbone)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        logging.info(f"Model {self.model_path} successfully loaded to {self.device} device.")


    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader =None,
              epochs: int = 1,
              val_interval: int = 1,
              learning_rate: float = 1e-3,
              eta_min: float = 1e-5,
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
        
        optimizer = torch.optim.AdamW(self.model.student_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        iterations = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min)

        loss_fn = STFPMLoss()

        msg = f"""Begin training
                     Model: STFPM
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

            for patches, _ in train_loader:
                # Repeat the channel bcs backbone expects 3 channels
                patches = patches.repeat(1, 3, 1, 1).to(self.device)

                optimizer.zero_grad()

                teacher_out, student_out = self.model(patches)

                loss = loss_fn(student_out, teacher_out)
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
