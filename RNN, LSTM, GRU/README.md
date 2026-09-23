# Week 6 - Experiment 6: End-to-End Study of RNN, LSTM and GRU

## Experiment Objectives
1. Understand the theoretical foundations and differences between Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRUs).
2. Process and classify multivariate time-series sensor data from the UCI Human Activity Recognition dataset.
3. Compare the training dynamics, performance, and computational complexity (number of parameters, training time) of RNN, LSTM, and GRU on sequence data.
4. Analyze the effect of sequence length on model performance.
5. Implement a Video Understanding pipeline combining a pre-trained CNN (MobileNetV2) for spatial feature extraction with RNNs for temporal classification using a subset of the UCF101 dataset.
6. Build and evaluate an Encoder-Decoder (Seq2Seq) model to solve a synthetic sequence reversal task.

---

## Folder Structure & Contents
```text
RNN, LSTM, GRU/
│
├── Experiment_6_RNN_LSTM_GRU.ipynb  # Comprehensive Jupyter Notebook for all tasks
├── experiment_6_rnn_lstm_gru.py     # Standalone Python script with the full pipeline
├── UCF101_subset.zip                # Subset of UCF101 Video Dataset
├── DATASET.md                       # Detailed documentation of the three datasets used
├── requirements.txt                 # List of Python dependencies
└── README.md                        # This experiment documentation and execution instructions
```

---

## Dataset Information
This experiment utilizes three distinct datasets:
1. **UCI HAR Using Smartphones**: Time-series sensor data for 6 human activities.
2. **UCF101 Subset**: Video data for 5 action recognition classes.
3. **Synthetic Sequence Reversal**: Synthetically generated digit sequences for Seq2Seq learning.

*For detailed dataset descriptions and preprocessing steps, see [DATASET.md](DATASET.md).*

---

## Dependencies
This experiment relies on standard deep learning and computer vision packages:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tensorflow`
- `opencv-python`
- `jupyter` (for running the notebook)

You can install all required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## Execution Instructions

You can run this experiment either via the **Jupyter Notebook** or the **Standalone Python Script**.

### Option 1: Standalone Python Script (`experiment_6_rnn_lstm_gru.py`)
1. Open your terminal or command prompt inside the `RNN, LSTM, GRU` directory:
   ```bash
   cd "RNN, LSTM, GRU"
   ```
2. Run the script:
   ```bash
   python experiment_6_rnn_lstm_gru.py
   ```
3. **Expected Output**:
   - The script will automatically download and extract the UCI HAR dataset if not present.
   - It will extract the `UCF101_subset.zip` archive.
   - The models (RNN, LSTM, GRU) will be built, trained, and evaluated.
   - Performance metrics and plots (Loss, Accuracy, Confusion Matrices) will be displayed.
   - It will conclude with the Sequence-to-Sequence task training and evaluation.

### Option 2: Jupyter Notebook (`Experiment_6_RNN_LSTM_GRU.ipynb`)
1. Start the Jupyter Notebook server from your terminal inside the directory:
   ```bash
   jupyter notebook Experiment_6_RNN_LSTM_GRU.ipynb
   ```
2. Execute the cells sequentially from top to bottom.
