# Building Segmentation with U-Net

This project implements a U-Net architecture for building segmentation from satellite images. The U-Net model is designed to accurately segment buildings, leveraging PyTorch for model training and evaluation.

## Features

- U-Net architecture for semantic segmentation.
- Custom dataset loader for handling satellite images.
- Metrics calculation including IoU, Dice coefficient, Precision, and Recall.
- Data augmentation techniques for robust model training.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- PyTorch 1.8.1 or higher
- torchvision, albumentations, Pillow, numpy, rasterio, pyyaml, matplotlib

## Installation

1. Clone the repository:
   https://github.com/ShahabEJ/Building-Segmentation.git

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ````

4. Organize your dataset as follows:
     ```
    .dataset/
    ├── train/
    │ ├── images/
    │ └── masks/
    ├── val/
    │ ├── images/
    │ └── masks/
    └── test/
    ├── images/
    └── masks/
    ```

Ensure that for each image in `images/`, there is a corresponding segmentation mask in `masks/`.

## Usage

To train the model, run:
  ```
  python main.py train
  ```

For evaluating the model on the test set, run:
   ```
  python main.py test
  ```


## Configuration
Adjust training parameters by editing the `config.yaml` file. Available settings include model architecture parameters, training epochs, batch size, and paths to your dataset.

## License
Free for use in both academic and comercial projects.

