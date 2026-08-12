# CIFAR-10 Dataset Information

## Overview
The **CIFAR-10 Dataset** is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 $32 \times 32$ color images containing one of 10 object classes, with 6,000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

## File Characteristics
- **Source**: Loaded automatically via `torchvision.datasets.CIFAR10`
- **Number of Instances**: 60,000 images total
  - **Training Set**: 50,000 images (split into 45,000 train, 5,000 validation in this experiment)
  - **Test Set**: 10,000 images
- **Number of Features**: $32 \times 32$ pixels, 3 color channels (RGB)
- **Target Variable**: 1 multiclass label (`0` to `9`)
- **Missing Values**: None

## Attribute Information
1. **Image Features** (`float tensor`): 3-channel RGB images, transformed to PyTorch Tensors. Dimensions: `[3, 32, 32]`.
2. **Class** (`int`): Target classification label corresponding to 10 classes.

## Classes
The dataset is completely balanced, with exactly 6,000 images per class:
- `0`: airplane
- `1`: automobile
- `2`: bird
- `3`: cat
- `4`: deer
- `5`: dog
- `6`: frog
- `7`: horse
- `8`: ship
- `9`: truck

The classes are mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

## Usage in CNN Experiment
In this experiment, the dataset is loaded using PyTorch's `DataLoader`. The images are transformed into tensors using `transforms.ToTensor()`, which scales the pixel values from $[0, 255]$ to $[0.0, 1.0]$. The input data is then fed through a series of Convolutional (`Conv2d`), ReLU, and Pooling layers for feature extraction, followed by a fully connected (`Linear`) layer for 10-class classification.
