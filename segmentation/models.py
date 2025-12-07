"""
Implemented segmentation models.

Include:
    - UNet++
    - SegFormer
    - DPT
"""
import logging

import segmentation_models_pytorch as smp

from segmentation.base import BaseModel


class NestedUNet(BaseModel):
    def __init__(self,
                 model_name: str,
                 saves_dir: str ="./saves/",
                 logs_dir: str ="./logs/",
                 device: str ="cpu"):
        super().__init__(model_name, "nested_unet", saves_dir, logs_dir, device)

    def create_model(self, encoder: str ="resnet18"):
        self.model = smp.UnetPlusPlus(
            encoder_name = encoder, 
            encoder_weights = "imagenet", 
            in_channels = 1, # Input channels - 1 for grayscale
            classes = 1, # Output classes - 1 for binary segmentation
        ).to(self.device)

        logging.info(f"""Created model:
                     Decoder: UNet++
                     Encoder: {encoder}
                     Input channels: 1
                     Output classes: 1
                     Device: {self.device}
                     """)
        

class SegFormer(BaseModel):
    def __init__(self,
                 model_name: str,
                 saves_dir: str ="./saves/",
                 logs_dir: str ="./logs/",
                 device: str ="cpu"):
        super().__init__(model_name, "segformer", saves_dir, logs_dir, device)


    def create_model(self, encoder: str ="mit_b0"):
        self.model = smp.Segformer(
            encoder_name = encoder, 
            encoder_weights = "imagenet", 
            in_channels = 1, # Input channels - 1 for grayscale
            classes = 1, # Output classes - 1 for binary segmentation
        ).to(self.device)

        # Freeze encoder
        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False

        logging.info(f"""Created model:
                     Decoder: SegFormer
                     Encoder: {encoder}
                     Input channels: 1
                     Output classes: 1
                     Device: {self.device}
                     """)

class DPT(BaseModel):
    def __init__(self,
                 model_name: str,
                 saves_dir: str ="./saves/",
                 logs_dir: str ="./logs/",
                 device: str ="cpu"):
        super().__init__(model_name, "dpt", saves_dir, logs_dir, device)


    def create_model(self, encoder: str ="tu-vit_base_patch16_224"):
        self.model = smp.DPT(
            encoder_name = encoder, 
            encoder_weights = "imagenet", 
            in_channels = 1, # Input channels - 1 for grayscale
            classes = 1, # Output classes - 1 for binary segmentation
        ).to(self.device)

        logging.info(f"""Created model:
                     Decoder: DPT
                     Encoder: {encoder}
                     Input channels: 1
                     Output classes: 1
                     Device: {self.device}
                     """)
        
