# Detection Baseline
> This repository contains the implementation of a baseline of anomaly detection models developed for my thesis.

For the project a specific **dataset was provided**, composed of CT scan of welds. A major characteristic of this dataset is that it consists exclusively of **anomalous grayscale images**. This unique composition presented several challenges during fitting it into existing models, leading to a relatively small number of models included in the baseline. Fortunately, the dataset includes few **ground-truth masks** indicating the location of anomalies for a small subset of images. This results in an extremely small and highly specific set, suggesting that segmentation models would be the most suitable option for this work.

While the initial assumption favors segmentation, the intention with the baseline was not to be strictly limited with these models. Therefore, the implementation is divided into two groups. The first is [Segmentation models](/segmentation/README.md) and second [Detection models](/detection/README.md). The detection group covers models with several different concepts, such as teacher-student, memory bank, and others. The implementation is based on [PyTorch](https://pytorch.org/).

### Models Included
> For more information about the models, see the specific directories.

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Description</th>
            <th>Group</th>
            <th>Year</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://arxiv.org/abs/1807.10165v1">UNet++</a></td>
            <td>nested U-Net</td>
            <td>Segmentation</td>
            <td>2018</td>
        </tr>
        <tr>
            <td><a href="https://arxiv.org/abs/2105.15203">SegFormer</a></td>
            <td>semantic Segmentation with transFormers</td>
            <td>Segmentation</td>
            <td>2021</td>
        </tr>
        <tr>
            <td><a href="https://arxiv.org/abs/2103.13413">DPT</a></td>
            <td>Dense Prediction Transformer</td>
            <td>Segmentation</td>
            <td>2021</td>
        </tr>
        <tr>
            <td><a href="https://arxiv.org/abs/2103.04257">STFPM</a></td>
            <td>Student-Teacher Feature Pyramid Matching</td>
            <td>Detection</td>
            <td>2021</td>
        </tr>
        <tr>
            <td><a href="https://arxiv.org/abs/1802.02611v3">DeepLabV3+</a></td>
            <td>extended DeepLabV3</td>
            <td>Segmentation</td>
            <td>2018</td>
        </tr>
    </tbody>
</table>

### Evaluation Metrics
* **AUROC** (Area Under ROC curve) at the pixel level
* **AP** (Average Precision) at the pixel level
* **IoU** (Intersection over Union)
* **F1** score

### Repository Structure

**`dataset`**
* [**`dataset.py`**](dataset/dataset.py) - Custom dataset implementation for the project
* [**`patches.py`**](dataset/patches.py) - Script for extracting patches from images

**`utils`**
* [**`metrics.py`**](utils/metrics.py) - Evaluation metrics for the baseline
* [**`visual.py`**](utils/visual.py) - Visualization of images / masks
* [**`logger.py`**](utils/logger.py) - Custom file logger

**`segmentation`**
* [**`fit.py`**](segmentation/fit.py) - Entry point for creating / training and testing model
* [**`inference.py`**](segmentation/inference.py) - Entry point for model inference
* [**`base.py`**](segmentation/base.py) - Abstract `BaseModel` class for other models with shared methods
* [**`models.py`**](segmentation/models.py) - Specific model implementations (Extends `BaseModel`)
* [**`utils.py`**](segmentation/utils.py) - Helper functions

**`detection`**
* [**`fit.py`**](detection/fit.py) - Entry point for creating / training and testing model
* [**`inference.py`**](detection/inference.py) - Entry point for model inference
* [**`utils.py`**](detection/utils.py) - Helper functions
    **`models`**
    * Specific implementations (including `BaseModel`)

**`README_img`** - Displayed images in READMEs

[**`requirements.txt`**](requirements.txt) - Required packages

### Results
> Due to the hardware limitations the baseline uses 224x224 inputs.

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Backbone</th>
            <th>Pixel AUROC</th>
            <th>Pixel AP</th>
            <th>IoU</th>
            <th>F1 score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Unet++</td>
            <td>Resnet-18</td>
            <td>0.637</td>
            <td>0.343</td>
            <td>0.304</td>
            <td>0.435</td>
        </tr>
        <tr>
            <td>SegFormer</td>
            <td>MiT-b0</td>
            <td>0.718</td>
            <td>0.289</td>
            <td>0.198</td>
            <td>0.305</td>
        </tr>
        <tr>
            <td>DPT</td>
            <td>ViT-base-16-224</td>
            <td>0.504</td>
            <td>0.236</td>
            <td>0.191</td>
            <td>0.289</td>
        </tr>
        <tr>
            <td>STFPM</td>
            <td>Resnet-18</td>
            <td>0.945</td>
            <td>0.188</td>
            <td>0.112</td>
            <td>0.196</td>
        </tr>
        <tr>
            <td>DeepLabV3+</td>
            <td>Resnet-18</td>
            <td>0.719</td>
            <td>0.278</td>
            <td>0.216</td>
            <td>0.329</td>
        </tr>
    </tbody>
</table>

Based on the **results table** provided above, it is clear that none of the models performed particularly well on the dataset. This outcome is most likely due to the highly **specific dataset** that was provided for this work. Each of the baseline models would probably need a complex optimization or modifications to achieve a significantly better performance.
