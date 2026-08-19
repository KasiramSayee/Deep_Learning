# Deep Learning & Neural Networks Lab Repository

Welcome to the **Deep Learning & Neural Networks Lab Experiments Repository**. This repository maintains structured, standalone implementations, datasets, dependencies, and execution instructions for each laboratory experiment conducted throughout the course.

---

## Repository Standards & Structure
To ensure reproducibility, clarity, and modularity, every experiment is encapsulated in its own dedicated folder. Each experiment directory contains the following **mandatory 5 components**:

1. **README (`README.md`)**: Comprehensive overview containing experiment objectives, theory, dependency summary, and step-by-step execution instructions.
2. **Source Code (`*.ipynb`, `*.py`)**: Clean, well-documented code maintained in both interactive Jupyter Notebook format and standalone Python script format.
3. **Dataset Information (`DATASET.md` / `*.txt` / `*.csv`)**: The dataset file(s) along with detailed documentation of attribute descriptions, target classes, and preprocessing steps.
4. **Dependency List (`requirements.txt`)**: Exact list of required Python packages and libraries needed to run the experiment.
5. **Execution Instructions**: Detailed commands and guidance located inside each experiment's README to easily run training, evaluation, and visualizations from scratch.

---

## Experiments Index

| Week / Exp # | Experiment Name | Folder Link | Key Topics & Algorithms | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Week 1 - Exp 1** | **Single-Layer Perceptron** | [`Perceptron/`](Perceptron/) | Perceptron learning rule, step activation, binary classification, Banknote Authentication dataset, Logic Gates (AND, OR, NOT, XOR) | Completed |
| **Week 2 - Exp 2** | **Multi-Layer Perceptron (MLP)** | [`MLP/`](MLP/) | Keras Sequential API, dense layers, ReLU/Softmax, multi-class classification, Fashion-MNIST dataset, hyperparameter optimization (`RandomizedSearchCV`) | Completed |
| **Week 3 - Exp 3** | **Convolutional Neural Network (CNN)** | [`CNN/`](CNN/) | PyTorch `nn.Module`, Convolutional layers, Max/Avg Pooling, feature map visualization, CIFAR-10 classification | Completed |
| **Week 4 - Exp 4** | **Transfer Learning with VGG16** | [`Transfer Learning/`](Transfer Learning/) | PyTorch `torchvision`, VGG16 pre-trained model, Feature Extraction, Fine-tuning, CIFAR-10 classification | Completed |

---

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd DL_Lab
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Navigate to an Experiment & Install Dependencies
Navigate into your desired experiment folder (`Perceptron`, `MLP`, `CNN`, or `Transfer Learning`) and install its specific dependencies:
```bash
# Example for Single-Layer Perceptron:
cd Perceptron
pip install -r requirements.txt

# Or for Multi-Layer Perceptron (MLP):
cd ../MLP
pip install -r requirements.txt

# Or for Convolutional Neural Network (CNN):
cd ../CNN
pip install -r requirements.txt

# Or for Transfer Learning:
cd "../Transfer Learning"
pip install -r requirements.txt
```

### 4. Run the Experiment
You can execute the standalone script:
```bash
# Inside Perceptron/:
python perceptron.py

# Inside MLP/:
python mlp.py

# Inside CNN/:
python cnn.py

# Inside Transfer Learning/:
python transfer_learning.py
```
Or launch the interactive Jupyter Notebook:
```bash
# Inside Perceptron/:
jupyter notebook perceptron.ipynb

# Inside MLP/:
jupyter notebook mlp.ipynb

# Inside CNN/:
jupyter notebook cnn.ipynb

# Inside Transfer Learning/:
jupyter notebook Transfer_Learning.ipynb
```
