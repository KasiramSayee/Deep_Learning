# Oxford-IIIT Pet Dataset Information

## Overview
The **Oxford-IIIT Pet Dataset** is sourced directly via `torchvision.datasets.OxfordIIITPet`. It is a 37-category pet image dataset with roughly 200 images for each class. The images have large variations in scale, pose, and lighting. All images have an associated ground truth annotation of breed.

## File Characteristics
- **Source**: `torchvision.datasets.OxfordIIITPet`
- **Format**: RGB Images
- **Number of Categories (Classes)**: 37 (different breeds of cats and dogs)
- **Total Images**: ~7,349 images

## Attribute Information
- **Input**: Raw RGB images of varying dimensions.
- **Target Variable**: 1 classification label (`category`), representing the breed of the pet (0 to 36).

## Usage in Comprehensive CNN Experiment
In this experiment, images are processed as follows:
1. Resized to $224 \times 224$ pixels.
2. Normalized using ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`), as required for the ImageNet-pretrained MobileNetV2 backbone.
3. Augmented during training using `RandomHorizontalFlip`.
4. Split into Train (80%), Validation (20%) from the `trainval` split, and evaluated on a separate independent `test` split.
