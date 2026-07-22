# Week 2 - Experiment 2: Multi-Layer Perceptron (MLP) Classifier

## Experiment Objectives
1. Understand the theoretical foundations and architecture of the **Multi-Layer Perceptron (MLP)**, fully connected dense layers, and non-linear activation functions (`ReLU`, `Softmax`).
2. Construct and train a deep neural network using **TensorFlow/Keras** (`Sequential` API) for multi-class image classification.
3. Perform Exploratory Data Analysis (EDA) and data preprocessing on the **Fashion-MNIST** dataset (flattening $28 \times 28$ images into 784-dimensional feature vectors and normalizing pixel intensities to $[0, 1]$).
4. Evaluate model performance using multi-class classification metrics (Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, Confusion Matrix, and Classification Report).
5. Conduct automated **Hyperparameter Optimization** using `SciKeras` (`KerasClassifier`) and `scikit-learn` (`RandomizedSearchCV`) across network depth, neuron counts, learning rates, optimizers, activation functions, and dropout rates, and compare baseline versus optimized model performance.

---

## Folder Structure & Contents
```text
MLP/
│
├── mlp.ipynb                         # Comprehensive Jupyter Notebook (EDA, Preprocessing, Baseline MLP, & Hyperparameter Optimization)
├── mlp.py                            # Standalone Python script for training, evaluation, and hyperparameter search
├── DATASET.md                        # Detailed documentation and attribute information for the Fashion-MNIST dataset
├── requirements.txt                  # List of Python dependencies
└── README.md                         # This experiment documentation and execution instructions
```

---

## Dataset Information
- **Source**: Zalando Research / Keras Datasets (`keras.datasets.fashion_mnist.load_data()`)
- **Instances**: 70,000 total samples (`60,000` training / `10,000` testing)
- **Features**: 784 continuous features ($28 \times 28$ grayscale pixels flattened and normalized to $[0, 1]$)
- **Target**: 10 fashion product categories (`T-shirt/Top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle Boot`)
- *For detailed feature statistics, class distributions, and preprocessing steps, see [DATASET.md](DATASET.md).*

---

## Dependencies
This experiment relies on the following Python packages:
- `tensorflow` (and Keras)
- `scikit-learn`
- `scikeras`
- `numpy`
- `matplotlib`
- `jupyter` (for running the notebook)

You can install all required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## Execution Instructions

You can run this experiment either via the **Standalone Python Script** (quick command-line execution and automated search) or the **Jupyter Notebook** (interactive step-by-step EDA and visualizations).

### Option 1: Standalone Python Script (`mlp.py`)
1. Open your terminal or command prompt inside the `MLP` directory:
   ```bash
   cd MLP
   ```
2. Run the script:
   ```bash
   python mlp.py
   ```
3. **Expected Output**:
   - Displays dataset dimensions (`60,000` train / `10,000` test) and logs class distributions.
   - Generates and saves visualization plots (`sample_images.eps` and `class_distribution.eps`).
   - Preprocesses data (flattening images to shape `(N, 784)` and normalizing pixel values to $[0, 1]$).
   - Builds and trains a baseline 3-layer MLP (`128` ReLU $\to$ `64` ReLU $\to$ `10` Softmax) over `20` epochs using the `Adam` optimizer and `categorical_crossentropy` loss.
   - Outputs baseline evaluation metrics (Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, Confusion Matrix, and Classification Report) and plots accuracy/loss learning curves.
   - Executes a 30-iteration Randomized Hyperparameter Search using 3-fold cross-validation (`RandomizedSearchCV`) over network depth, hidden units, learning rates, optimizers (`adam`, `sgd`, `rmsprop`), activations (`relu`, `tanh`, `sigmoid`), and dropout (`0.0`, `0.2`, `0.5`).
   - Displays the best found hyperparameters and generates bar charts comparing baseline versus optimized model accuracy.

### Option 2: Jupyter Notebook (`mlp.ipynb`)
1. Start the Jupyter Notebook server from your terminal inside the `MLP` directory:
   ```bash
   jupyter notebook mlp.ipynb
   ```
   *Or open `mlp.ipynb` directly in VS Code / Cursor / JupyterLab.*
2. Execute the cells sequentially from top to bottom (`Shift + Enter` or `Run All Cells`) to step through:
   - **Task 1**: Dataset Loading, Dimensions verification, displaying 10 sample fashion images, and plotting class distributions.
   - **Task 2**: Flattening 2D image matrices into 784D feature vectors and normalizing intensities ($0-255 \to 0-1$).
   - **Task 3 & 4**: Constructing the Keras Sequential architecture, compiling the model, and monitoring training/validation loss and accuracy across 20 epochs (`validation_split=0.2`).
   - **Task 5**: Detailed model evaluation on the test set, including confusion matrix heatmap generation and full classification reports.
   - **Hyperparameter Optimization**: Installing compatible `SciKeras`/`scikit-learn` versions, constructing the wrapper model, performing randomized search, and plotting parameter exploration behavior against baseline results.
