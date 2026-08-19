# CIFAR-10 Dataset Information

## Overview
The **CIFAR-10 Dataset** is a well-known benchmark dataset in machine learning and computer vision. In this experiment, it is loaded directly using `torchvision.datasets.CIFAR10`. The dataset consists of 60,000 32x32 color images across 10 distinct classes, with 6,000 images per class. It is widely used for image classification tasks.

## File Characteristics
- **Source**: `torchvision.datasets` (downloaded automatically)
- **Format**: PyTorch Tensors (converted from PIL Images)
- **Number of Instances**: 60,000
  - Training Set: 50,000 (split into 45,000 train and 5,000 validation in this experiment)
  - Test Set: 10,000
- **Number of Features**: 32x32x3 = 3,072 raw pixel values (resized to 224x224x3 for VGG16)
- **Target Variable**: 1 multiclass classification label (`0` to `9`)
- **Missing Values**: None

## Attribute Information
1. **Features**: RGB images
2. **Label** (`int`): Target classification label:
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

## Class Distribution
The dataset is perfectly balanced.
- Each of the 10 classes contains exactly 6,000 images (5,000 in the training set and 1,000 in the test set).

## Usage in Transfer Learning Experiment
In this experiment, the CIFAR-10 images are preprocessed using `torchvision.transforms`:
1. Resized from their original `32x32` resolution to `224x224` to meet the input size requirements of the pre-trained VGG16 model.
2. Converted to PyTorch tensors with pixel values scaled to the `[0, 1]` range.
The transformed dataset is then fed in batches to the VGG16 model for both feature extraction (frozen base) and fine-tuning.
