# Week 5 - Experiment 5: Comprehensive CNN Study

## Experiment Objectives
1. Implement a complete CNN classification pipeline using a pretrained **MobileNetV2** backbone on the **Oxford-IIIT Pet Dataset** (37 pet breeds).
2. Evaluate different **Weight Initialization** strategies (Zero, Random, Xavier, He) for newly-added classifier layers.
3. Analyze the impact of various **Regularization techniques** (L2 Weight Decay, Dropout, Batch Normalization) on the generalization gap.
4. Compare the convergence speed and final accuracy of multiple **Optimization algorithms** (SGD, SGD+Momentum, RMSProp, Adam).
5. Perform **Hyperparameter Tuning** to find the optimal Learning Rate, Batch Size, and Dropout Rate.
6. Contrast **Transfer Learning** approaches: Feature Extraction (frozen base) vs. Fine-Tuning (unfreezing the last few convolutional blocks).
7. Validate model robustness using **K-Fold Cross-Validation** (5 folds) with the best discovered hyperparameter configurations.

---

## Folder Structure & Contents
```text
Comprehensive CNN/
│
├── comprehensive_cnn.ipynb           # Comprehensive Jupyter Notebook with all tasks
├── comprehensive_cnn.py              # Standalone Python script with all tasks
├── dataset.md                        # Detailed documentation for the Oxford-IIIT Pet dataset
├── requirements.txt                  # List of Python dependencies
└── readme.md                         # This experiment documentation and execution instructions
```

---

## Dataset Information
- **Dataset**: Oxford-IIIT Pet Dataset
- **Source**: Loaded automatically via `torchvision.datasets.OxfordIIITPet`
- **Classes**: 37 breeds (dogs and cats)
- **Image Size**: Resized and normalized to 224x224 RGB images
- *For detailed attributes and usage, see [dataset.md](dataset.md).*

---

## Dependencies
This experiment relies on the following standard Python and Deep Learning packages:
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `jupyter` (for running the notebook)

You can install all required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## Execution Instructions

You can run this experiment either via the **Jupyter Notebook** (for interactive visualizations and step-by-step output) or the **Standalone Python Script**.

### Option 1: Standalone Python Script (`comprehensive_cnn.py`)
1. Open your terminal or command prompt inside the `Comprehensive CNN` directory:
   ```bash
   cd "Comprehensive CNN"
   ```
2. Run the script:
   ```bash
   python comprehensive_cnn.py
   ```
3. **Expected Output**:
   - Downloads the Oxford-IIIT Pet Dataset if not already present.
   - Executes all tasks sequentially: Data Prep, Weight Init, Regularization, Batch Norm, Optimizers, Hyperparameter Tuning, Transfer Learning, and K-Fold CV.
   - Generates and displays Matplotlib plots summarizing the metrics of each task.

### Option 2: Jupyter Notebook (`comprehensive_cnn.ipynb`)
1. Start the Jupyter Notebook server from your terminal inside the `Comprehensive CNN` directory:
   ```bash
   jupyter notebook comprehensive_cnn.ipynb
   ```
2. Execute the cells sequentially from top to bottom (`Shift + Enter` or `Run All Cells`) to step through each phase of the comprehensive study.
