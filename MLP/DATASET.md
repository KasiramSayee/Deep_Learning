# Fashion-MNIST Dataset Information

## Overview
The **Fashion-MNIST Dataset** is sourced from [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) and is integrated into Keras (`tensorflow.keras.datasets.fashion_mnist`). It was created as a modern, direct drop-in replacement for the original MNIST digit dataset for benchmarking machine learning and deep learning algorithms. It consists of grayscale images of clothing and fashion articles across 10 distinct categories. Each image is standardized to $28 \times 28$ pixels.

## File Characteristics
- **Source / Loader**: Dynamic loading via `keras.datasets.fashion_mnist.load_data()`
- **Format**: 3D NumPy array (`numpy.ndarray`) of type `uint8` with grayscale intensities ($0$ to $255$)
- **Number of Instances**: 70,000 total images
  - **Training Split**: 60,000 images
  - **Testing Split**: 10,000 images
- **Number of Features**: 784 numerical features ($28 \times 28$ pixels flattened into a 1D vector)
- **Target Variable**: 1 integer classification label (`Y`) per sample
- **Missing Values**: None

## Attribute Information
1. **Input Features (`X`)**: Each image is a $28 \times 28$ matrix of integer pixel intensities ranging from `0` (pure black/background) to `255` (pure white/foreground). When flattened during preprocessing, each image becomes a vector of 784 features ($X_1, X_2, \dots, X_{784}$).
2. **Target Class (`Y`)**: Integer label from `0` to `9` representing the product category:
   - `0`: T-shirt/Top
   - `1`: Trouser
   - `2`: Pullover
   - `3`: Dress
   - `4`: Coat
   - `5`: Sandal
   - `6`: Shirt
   - `7`: Sneaker
   - `8`: Bag
   - `9`: Ankle Boot

## Class Distribution
The dataset is perfectly balanced across all 10 classes:
- **Training Set**: Exactly 6,000 samples per class (`10.0%` distribution per class across 60,000 samples)
- **Testing Set**: Exactly 1,000 samples per class (`10.0%` distribution per class across 10,000 samples)

## Usage in MLP Experiment
In this experiment, the raw image arrays are preprocessed as follows before being fed into the Multi-Layer Perceptron (MLP):
1. **Flattening**: The 3D image arrays (`N, 28, 28`) are reshaped (`X.reshape(shape[0], -1)`) into 2D feature matrices (`N, 784`).
2. **Normalization**: Grayscale values in $[0, 255]$ are converted to `float32` and divided by `255.0` (`X.astype("float32") / 255.0`) to scale features into the $[0.0, 1.0]$ interval. This stabilizes gradient descent and accelerates neural network convergence.
3. **Label Encoding**: 
   - For baseline `Sequential` MLP training with `categorical_crossentropy` loss, target labels (`Y`) are converted into 10-dimensional one-hot encoded vectors using `to_categorical`.
   - For hyperparameter optimization (`KerasClassifier` / `RandomizedSearchCV`) with `sparse_categorical_crossentropy` loss, target labels are preserved as integers (`0` to `9`).
