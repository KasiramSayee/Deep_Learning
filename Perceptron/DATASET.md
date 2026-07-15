# Banknote Authentication Dataset Information

## Overview
The **Banknote Authentication Dataset** (`data_banknote_authentication.txt`) is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have $400 \times 400$ pixels. Wavelet Transform tools were used to extract numerical features from the images.

## File Characteristics
- **Filename**: `data_banknote_authentication.txt`
- **Format**: Comma-Separated Values (CSV), no header row
- **Number of Instances**: 1,372
- **Number of Features**: 4 continuous numerical features
- **Target Variable**: 1 binary classification label (`Class`)
- **Missing Values**: None

## Attribute Information
1. **Variance** (`float`): Variance of Wavelet Transformed image (continuous)
2. **Skewness** (`float`): Skewness of Wavelet Transformed image (continuous)
3. **Curtosis** (`float`): Curtosis of Wavelet Transformed image (continuous)
4. **Entropy** (`float`): Entropy of image (continuous)
5. **Class** (`int`): Target classification label:
   - `0` for genuine (authentic) banknote
   - `1` for forged banknote

## Class Distribution
- **Class 0 (Genuine)**: ~55.5% (762 samples)
- **Class 1 (Forged)**: ~44.5% (610 samples)

## Usage in Perceptron Experiment
In this experiment, the continuous input features ($X_1, X_2, X_3, X_4$) are scaled using `MinMaxScaler` to the $[0, 1]$ range before feeding into the single-layer Perceptron model. The binary target `Class` is predicted directly using a step activation function.
