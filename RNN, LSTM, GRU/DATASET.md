# RNN, LSTM, GRU Datasets Information

This experiment utilizes three distinct datasets to evaluate RNN, LSTM, and GRU architectures across different domains: time-series sensor data, video sequences, and synthetic text/sequence data.

## 1. UCI HAR Using Smartphones Dataset
- **Domain**: Human Activity Recognition (Sensor Data)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Features**: 9 raw inertial signals (3-axis body acceleration, 3-axis body gyroscope, 3-axis total acceleration).
- **Sequence Length**: 128 timestamps per window.
- **Classes**: 6 activities (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`).
- **Usage**: Used to train and compare standard RNN, LSTM, and GRU models for sequence classification.

## 2. UCF101 Subset
- **Domain**: Video Understanding / Action Recognition
- **Source**: Subset of the [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- **Format**: `.zip` archive containing `.avi` video files.
- **Classes**: 5 selected action categories (`CricketShot`, `PlayingCello`, `Punch`, `ShavingBeard`, `TennisSwing`).
- **Preprocessing**: 
  - 10 frames sampled uniformly per video.
  - Resized to $224 \times 224 \times 3$.
  - Features extracted using a pre-trained MobileNetV2 CNN, resulting in sequences of feature vectors.
- **Usage**: Used to train CNN-LSTM and CNN-GRU models for spatiotemporal video classification.

## 3. Synthetic Sequence Reversal Dataset
- **Domain**: Sequence-to-Sequence (Seq2Seq) Learning
- **Format**: Synthetically generated arrays of integers.
- **Vocabulary**: Digits 1-9 (0 is reserved).
- **Sequence Length**: 6 tokens.
- **Task**: The model receives a sequence of digits and must predict the reversed sequence.
- **Usage**: Used to demonstrate Encoder-Decoder architectures for tasks where the input and output are both sequences.
