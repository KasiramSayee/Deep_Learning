# Week 1 - Experiment 1: Perceptron Classifier

## Experiment Objectives
1. Understand the theoretical foundation of the **Single-Layer Perceptron** and linear separability.
2. Implement the Perceptron learning algorithm from scratch using NumPy (forward propagation, step activation function, error computation, and weight/bias updates).
3. Perform Exploratory Data Analysis (EDA) and data preprocessing (scaling and splitting) on the Banknote Authentication dataset.
4. Train and evaluate the Perceptron model using standard classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
5. Analyze and visualize model convergence, weight evolution, bias adjustments, and learning rate impacts.

---

## Folder Structure & Contents
```text
Perceptron/
│
├── perceptron.ipynb                  # Comprehensive Jupyter Notebook (EDA, Implementation, & Visualizations)
├── perceptron.py                     # Standalone Python training and evaluation script
├── data_banknote_authentication.txt  # Banknote Authentication Dataset (UCI)
├── DATASET.md                        # Detailed documentation and attribute information for the dataset
├── requirements.txt                  # List of Python dependencies
└── README.md                         # This experiment documentation and execution instructions
```

---

## Dataset Information
- **Source**: UCI Machine Learning Repository (`data_banknote_authentication.txt`)
- **Instances**: 1,372 samples
- **Features**: 4 continuous features extracted from Wavelet Transformed images:
  - `Variance`, `Skewness`, `Curtosis`, `Entropy`
- **Target**: `Class` (`0` = Genuine Banknote, `1` = Forged Banknote)
- *For detailed feature statistics and descriptions, see [DATASET.md](DATASET.md).*

---

## Dependencies
This experiment relies on the following standard Python data science packages:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter` (for running the notebook)

You can install all required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## Execution Instructions

You can run this experiment either via the **Jupyter Notebook** (interactive step-by-step EDA and plots) or the **Standalone Python Script** (quick command-line execution).

### Option 1: Standalone Python Script (`perceptron.py`)
1. Open your terminal or command prompt inside the `Perceptron` directory:
   ```bash
   cd Perceptron
   ```
2. Run the script:
   ```bash
   python perceptron.py
   ```
3. **Expected Output**:
   - Logs dataset loading and sample splits (`1,097` train / `275` test).
   - Displays training progress and misclassified sample count for each epoch.
   - Outputs evaluation metrics (Accuracy ~`98.55%`, Precision ~`98.43%`, Recall ~`98.43%`, F1-Score ~`98.43%`).
   - Automatically generates and saves high-resolution visualization plots (`confusion_matrix.png` and `training_error.png`).

### Option 2: Jupyter Notebook (`Perceptron.ipynb`)
1. Start the Jupyter Notebook server from your terminal inside the `Perceptron` directory:
   ```bash
   jupyter notebook Perceptron.ipynb
   ```
   *Or open `Perceptron.ipynb` directly in VS Code / Cursor / JupyterLab.*
2. Execute the cells sequentially from top to bottom (`Shift + Enter` or `Run All Cells`) to step through:
   - **Task 1 & 2**: Dataset Loading, Summary Statistics, Histograms, Scatter Plots, and Correlation Heatmaps.
   - **Task 3**: MinMax Scaling and Train-Test Split (`80%` / `20%`).
   - **Task 4 & 5**: Perceptron class instantiation and model training over `100` epochs.
   - **Task 6**: Performance metrics evaluation and confusion matrix visualization.
   - **Task 7**: In-depth analysis of training error curves, weight/bias evolution over epochs, and learning rate comparison (`0.001`, `0.01`, `0.1`).
