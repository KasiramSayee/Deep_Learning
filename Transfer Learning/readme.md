# Week 4 - Experiment 4: Transfer Learning with VGG16

## Experiment Objectives
1. Understand the concept of **Transfer Learning** and how to leverage pre-trained Convolutional Neural Networks (CNNs).
2. Load and preprocess the CIFAR-10 dataset using PyTorch's `torchvision`, resizing images to 224x224 for compatibility with the VGG16 architecture.
3. Modify a pre-trained VGG16 model (trained on ImageNet) by replacing its original classifier head to suit a 10-class classification task.
4. Perform **feature extraction** by freezing the convolutional base and training only the newly added classifier head.
5. Perform **fine-tuning** by unfreezing the final convolutional block of the VGG16 feature extractor to adapt high-level features to the CIFAR-10 dataset.
6. Train and evaluate the model using standard classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) and analyze the performance before and after fine-tuning.

---

## Folder Structure & Contents
```text
Transfer Learning/
│
├── Transfer_Learning.ipynb           # Comprehensive Jupyter Notebook (Data loading, Implementation, & Visualizations)
├── transfer_learning.py              # Standalone Python training and evaluation script
├── dataset.md                        # Detailed documentation and attribute information for the dataset
├── requirements.txt                  # List of Python dependencies
└── readme.md                         # This experiment documentation and execution instructions
```

---

## Dataset Information
- **Source**: `torchvision.datasets.CIFAR10`
- **Instances**: 60,000 samples (50,000 train, 10,000 test)
- **Features**: 32x32 color images (resized to 224x224 via transforms)
- **Target**: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- *For detailed feature descriptions and distribution, see [dataset.md](dataset.md).*

---

## Dependencies
This experiment relies on the following standard Python machine learning and data science packages:
- `torch`
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

You can run this experiment either via the **Jupyter Notebook** (interactive step-by-step EDA and plots) or the **Standalone Python Script** (quick command-line execution).

### Option 1: Standalone Python Script (`transfer_learning.py`)
1. Open your terminal or command prompt inside the `Transfer Learning` directory:
   ```bash
   cd "Transfer Learning"
   ```
2. Run the script:
   ```bash
   python transfer_learning.py
   ```
3. **Expected Output**:
   - Downloads the CIFAR-10 dataset if not already present.
   - Logs model architecture, total parameters, and trainable parameters.
   - Displays training and validation progress across epochs for both the feature extraction and fine-tuning phases.
   - Outputs final evaluation metrics (Testing Accuracy, Precision, Recall, F1 Score).
   - Generates and displays visualization plots (Confusion Matrix, Accuracy/Loss curves, and Misclassified samples).

### Option 2: Jupyter Notebook (`Transfer_Learning.ipynb`)
1. Start the Jupyter Notebook server from your terminal inside the `Transfer Learning` directory:
   ```bash
   jupyter notebook Transfer_Learning.ipynb
   ```
2. Execute the cells sequentially from top to bottom to step through:
   - **Task 1**: Dataset Preparation and sample visualization.
   - **Task 2**: Transfer Learning - Model Setup (VGG16 instantiation and modification).
   - **Task 3**: Model Training (feature extraction with frozen base).
   - **Task 4**: Fine Tuning (unfreezing the last convolutional block).
   - **Task 5**: Model Evaluation (metrics, confusion matrix, and misclassified examples).
