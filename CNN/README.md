# Week 3 - Experiment 3: Convolutional Neural Network (CNN)

## Experiment Objectives
1. Understand and load the **CIFAR-10 dataset** using PyTorch and `torchvision`.
2. Implement and apply Convolutional layers with varying kernel sizes, strides, and padding.
3. Visualize feature maps generated after applying convolutional filters.
4. Apply and visualize the effects of Max Pooling and Average Pooling operations.
5. Construct a complete **Convolutional Neural Network (CNN)** from scratch using PyTorch's `nn.Module`.
6. Train the CNN using Cross-Entropy Loss and Adam Optimizer over multiple epochs.
7. Evaluate the trained model using standard classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
8. Visualize training vs. validation loss and accuracy curves to monitor model convergence.

---

## Folder Structure & Contents
```text
CNN/
│
├── cnn.ipynb                         # Comprehensive Jupyter Notebook (Data Loading, Implementation, & Visualizations)
├── cnn.py                            # Standalone Python training and evaluation script
├── DATASET.md                        # Detailed documentation and attribute information for the dataset
├── requirements.txt                  # List of Python dependencies
└── README.md                         # This experiment documentation and execution instructions
```

---

## Dataset Information
- **Source**: `torchvision.datasets.CIFAR10` (CIFAR-10 Dataset)
- **Instances**: 60,000 color images (45,000 train, 5,000 validation, 10,000 test)
- **Features**: 32x32 pixel RGB images (3 channels)
- **Target**: 10 mutually exclusive classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- *For detailed feature statistics and descriptions, see [DATASET.md](DATASET.md).*

---

## Dependencies
This experiment relies on the following standard Python packages:
- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter` (for running the notebook)

You can install all required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## Execution Instructions

You can run this experiment either via the **Jupyter Notebook** (interactive step-by-step plots) or the **Standalone Python Script** (quick command-line execution).

### Option 1: Standalone Python Script (`cnn.py`)
1. Open your terminal or command prompt inside the `CNN` directory:
   ```bash
   cd CNN
   ```
2. Run the script:
   ```bash
   python cnn.py
   ```
3. **Expected Output**:
   - Downloads the CIFAR-10 dataset (if not already downloaded).
   - Displays sample images and feature map visualizations.
   - Outputs training progress (loss and accuracy) per epoch.
   - Prints final evaluation metrics (Accuracy, Precision, Recall, F1-Score) and a classification report.
   - Plots the confusion matrix and loss/accuracy learning curves.

### Option 2: Jupyter Notebook (`cnn.ipynb`)
1. Start the Jupyter Notebook server from your terminal inside the `CNN` directory:
   ```bash
   jupyter notebook cnn.ipynb
   ```
   *Or open `cnn.ipynb` directly in VS Code / Cursor / JupyterLab.*
2. Execute the cells sequentially from top to bottom (`Shift + Enter` or `Run All Cells`) to step through:
   - **Task 1**: Load CIFAR-10 dataset and plot class distribution.
   - **Task 2**: Implement convolutional layers with different kernel sizes.
   - **Task 3**: Study the effect of stride and padding hyperparameters.
   - **Task 4**: Visualize feature maps after the first convolution layer.
   - **Task 5**: Implement and visualize Max Pooling and Average Pooling.
   - **Task 6**: Build and train the complete CNN architecture.
   - **Task 7**: Evaluate the model using advanced metrics and confusion matrix.
   - **Visualization**: Plot the learning curves (loss and accuracy vs. epochs).
