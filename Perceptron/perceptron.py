"""
Perceptron Implementation & Exploratory Data Analysis
Week 1 - Experiment 1: Banknote Authentication
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Configure Matplotlib global styling to match notebook exactly
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.size"] = 12


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Perceptron Classifier Constructor

        Parameters:
            learning_rate (float): Learning rate for weight updates
            epochs (int): Number of iterations over the dataset
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.misclassified_history = []
        self.weights_history = []
        self.bias_history = []

    def _step_function(self, z):
        """Step Activation Function"""
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        """
        Train the Perceptron model using the Perceptron learning rule.
        """
        n_samples, n_features = X.shape

        # Weight Initialization
        self.weights = np.zeros(n_features)

        # Bias Initialization
        self.bias = 0

        print(f"--- Training Perceptron (LR={self.learning_rate}, Epochs={self.epochs}) ---")

        for epoch in range(self.epochs):
            misclassified = 0

            for i in range(n_samples):
                x = X[i]
                target = y.iloc[i] if isinstance(y, pd.Series) else y[i]

                # Forward Propagation
                z = np.dot(self.weights, x) + self.bias
                prediction = 1 if z >= 0 else 0

                # Error computation
                error = target - prediction

                # Weight and Bias Update
                if error != 0:
                    misclassified += 1
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error

            self.misclassified_history.append(misclassified)
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

            if (epoch + 1) % 10 == 0 or epoch == 0 or misclassified == 0:
                print(f"Epoch {epoch + 1:3d}/{self.epochs} - Misclassified Samples: {misclassified} - Bias: {self.bias:.4f}")

            # Early Stopping
            if misclassified == 0:
                print(f"Training Complete (Converged at Epoch {epoch + 1})")
                break

    def predict(self, X):
        """Predict class labels for given test samples."""
        z = np.dot(X, self.weights) + self.bias
        return self._step_function(z)

    def score(self, X, y):
        """Calculate classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def save_plot(filename):
    """Helper to save plots in both EPS and PNG formats."""
    eps_name = f"{filename}.eps"
    png_name = f"{filename}.png"
    plt.savefig(eps_name, dpi=600, format="eps", bbox_inches="tight")
    plt.savefig(png_name, dpi=300, format="png", bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {eps_name} (and {png_name})")


def generate_eda_plots(df):
    """Generate all Exploratory Data Analysis plots (Graphs 1 to 4)."""
    print("\n--- Generating Exploratory Data Analysis Plots ---")
    features = df.columns

    # Graph 1: Histogram Subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, col in enumerate(features[:4]):
        axes[i].hist(df[col], bins=15, edgecolor="black")
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel(col, fontweight="bold")
        axes[i].set_ylabel("Frequency", fontweight="bold")
    plt.tight_layout()
    save_plot("Histogram_Subplots")

    # Graph 2: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap", fontweight="bold")
    save_plot("Correlation_Heatmap")

    # Graph 3: Scatter Subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, col in enumerate(features[:4]):
        axes[i].scatter(df["Class"], df[col])
        axes[i].set_xlabel("Class", fontweight="bold")
        axes[i].set_ylabel(f"{col}", fontweight="bold")
        axes[i].set_title(f"{col} vs Class", fontweight="bold")
    plt.tight_layout()
    save_plot("Scatter_Subplots")

    # Graph 4: Boxplot Subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, col in enumerate(features[:4]):
        axes[i].boxplot(df[col])
        axes[i].set_title(col, fontweight="bold")
    plt.tight_layout()
    save_plot("Boxplots")


def generate_evaluation_plots(cm):
    """Generate Model Evaluation plots (Graph 5)."""
    print("\n--- Generating Model Evaluation Plots ---")
    # Graph 5: Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"]
    )
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.title("Confusion Matrix", fontweight="bold")
    save_plot("Confusion_Matrix")


def generate_learning_process_plots(model, df, X_train, y_train):
    """Generate Perceptron Learning Process & Hyperparameter plots (Graphs 6 to 9)."""
    print("\n--- Generating Learning Process Analysis Plots ---")

    # Graph 6: Training Error vs Epoch
    epochs = range(1, len(model.misclassified_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, model.misclassified_history, marker="o", linestyle="-", color="blue")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Number of Misclassified Samples", fontweight="bold")
    plt.title("Training Error vs Epoch", fontweight="bold")
    plt.grid(True)
    save_plot("Training_Error_vs_Epoch")

    # Graph 7: Weight Evolution Over Epochs
    weights_history = np.array(model.weights_history)
    n_features = weights_history.shape[1]
    feature_names = df.columns[:-1]
    epochs = range(1, len(weights_history) + 1)

    fig, axes = plt.subplots(n_features, 1, figsize=(10, 2 * n_features), sharex=True)
    fig.suptitle("Perceptron Weight Evolution Over Epochs", fontweight="bold", fontsize=16)
    for i in range(n_features):
        axes[i].plot(epochs, weights_history[:, i], label=f"Weight {feature_names[i]}")
        axes[i].set_ylabel(f"Weight {feature_names[i]}", fontweight="bold")
        axes[i].grid(True)
    axes[-1].set_xlabel("Epoch", fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot("Weight_Evolution")

    # Graph 8: Bias Evolution Over Epochs
    bias_history = model.bias_history
    epochs = range(1, len(bias_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bias_history, marker="o", linestyle="-", color="green")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Bias Value", fontweight="bold")
    plt.title("Perceptron Bias Evolution Over Epochs", fontweight="bold")
    plt.grid(True)
    save_plot("Bias_Evolution")

    # Graph 9: Learning Rate Comparison
    learning_rates = [0.001, 0.01, 0.1]
    all_misclassified_histories = []
    print("\nPerforming Learning Rate Comparison Study...")
    for lr in learning_rates:
        model_lr = Perceptron(learning_rate=lr, epochs=100)
        model_lr.fit(X_train, y_train)
        all_misclassified_histories.append(model_lr.misclassified_history)

    plt.figure(figsize=(12, 7))
    for i, history in enumerate(all_misclassified_histories):
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, history, marker="o", linestyle="-", label=f"LR = {learning_rates[i]}")
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Number of Misclassified Samples", fontweight="bold")
    plt.title("Training Error vs Epoch for Different Learning Rates", fontweight="bold")
    plt.legend()
    plt.grid(True)
    save_plot("Learning_Rate_Comparison")


def main():
    # 1. Load Dataset
    dataset_path = "data_banknote_authentication.txt"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found in current directory.")

    columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]
    df = pd.read_csv(dataset_path, header=None, names=columns)
    print(f"Dataset loaded successfully. Shape: {df.shape}\n")

    # 2. Exploratory Data Analysis (EDA Plots 1 to 4)
    generate_eda_plots(df)

    # 3. Preprocessing
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nTraining Features Shape : {X_train.shape}")
    print(f"Testing Features Shape  : {X_test.shape}\n")

    # 4. Model Training
    model = Perceptron(learning_rate=0.01, epochs=100)
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # 6. Evaluation Plots (Plot 5)
    generate_evaluation_plots(cm)

    # 7. Learning Process & Hyperparameter Plots (Plots 6 to 9)
    generate_learning_process_plots(model, df, X_train, y_train)

    print("\nAll 9 graphs have been successfully generated and saved (.eps and .png format)!")


if __name__ == "__main__":
    main()
