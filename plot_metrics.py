import sys
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import SimpleCNN_v1, SimpleCNN_v2, SimpleCNN_v3
import random
def plot_history(history_files):
    """
    Plots the training and validation metrics for one or more models.

    Parameters:
    history_files (list of str): List of file paths to JSON files containing the training
                                 and validation metrics for each model. Each JSON file is 
                                 expected to include:
                                 - "train_loss"
                                 - "val_loss"
                                 - "train_accuracy"
                                 - "val_accuracy"
                                 - "train_precision"
                                 - "val_precision"
                                 - "train_recall"
                                 - "val_recall"
                                 - "train_f1"
                                 - "val_f1"

    This function generates plots for the following metrics:
    - Loss
    - Accuracy
    - Precision
    - F1 Score

    Metrics are plotted for both training and validation data.
    Separate figures are created for training and validation metrics.
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    fig_train, axes_train = plt.subplots(2, 2, figsize=(15, 10))
    fig_val, axes_val = plt.subplots(2, 2, figsize=(15, 10))

    for i, history_file in enumerate(history_files):
        with open(history_file, "r") as f:
            history = json.load(f)

        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        train_accuracy = history["train_accuracy"]
        val_accuracy = history["val_accuracy"]
        train_precision = history["train_precision"]
        val_precision = history["val_precision"]
        train_f1 = history["train_f1"]
        val_f1 = history["val_f1"]
        epochs = range(1, len(train_loss) + 1)

        color = colors[i % len(colors)]

        axes_train[0, 0].plot(epochs, train_loss, label=f"Train Loss (Model {i+1})", color=color)
        axes_train[0, 1].plot(epochs, train_accuracy, label=f"Train Accuracy (Model {i+1})", color=color)
        axes_train[1, 0].plot(epochs, train_precision, label=f"Train Precision (Model {i+1})", color=color)
        axes_train[1, 1].plot(epochs, train_f1, label=f"Train F1 Score (Model {i+1})", color=color)

        axes_val[0, 0].plot(epochs, val_loss, label=f"Val Loss (Model {i+1})", color=color)
        axes_val[0, 1].plot(epochs, val_accuracy, label=f"Val Accuracy (Model {i+1})", color=color)
        axes_val[1, 0].plot(epochs, val_precision, label=f"Val Precision (Model {i+1})", color=color)
        axes_val[1, 1].plot(epochs, val_f1, label=f"Val F1 Score (Model {i+1})", color=color)

    axes_train[0, 0].set_xlabel("Epochs")
    axes_train[0, 0].set_ylabel("Loss")
    axes_train[0, 0].set_title("Train Loss History")
    axes_train[0, 0].legend()

    axes_train[0, 1].set_xlabel("Epochs")
    axes_train[0, 1].set_ylabel("Accuracy")
    axes_train[0, 1].set_title("Train Accuracy History")
    axes_train[0, 1].legend()

    axes_train[1, 0].set_xlabel("Epochs")
    axes_train[1, 0].set_ylabel("Precision")
    axes_train[1, 0].set_title("Train Precision History")
    axes_train[1, 0].legend()

    axes_train[1, 1].set_xlabel("Epochs")
    axes_train[1, 1].set_ylabel("F1 Score")
    axes_train[1, 1].set_title("Train F1 Score History")
    axes_train[1, 1].legend()

    axes_val[0, 0].set_xlabel("Epochs")
    axes_val[0, 0].set_ylabel("Loss")
    axes_val[0, 0].set_title("Validation Loss History")
    axes_val[0, 0].legend()

    axes_val[0, 1].set_xlabel("Epochs")
    axes_val[0, 1].set_ylabel("Accuracy")
    axes_val[0, 1].set_title("Validation Accuracy History")
    axes_val[0, 1].legend()

    axes_val[1, 0].set_xlabel("Epochs")
    axes_val[1, 0].set_ylabel("Precision")
    axes_val[1, 0].set_title("Validation Precision History")
    axes_val[1, 0].legend()

    axes_val[1, 1].set_xlabel("Epochs")
    axes_val[1, 1].set_ylabel("F1 Score")
    axes_val[1, 1].set_title("Validation F1 Score History")
    axes_val[1, 1].legend()

    plt.tight_layout()
    plt.figure(fig_train.number)
    plt.show()
    plt.figure(fig_val.number)
    plt.show()



def display_predictions(model, X_validation, Y_validation, labels_map, n_samples=5):
    """
    Displays n_samples randomly selected images from the validation set 
    with their true and predicted labels.

    Parameters:
    model (nn.Module): Trained PyTorch model.
    X_validation (torch.Tensor): Validation images.
    Y_validation (torch.Tensor): True labels for validation images.
    labels_map (dict): Mapping of label indices to class names.
    n_samples (int): Number of samples to display.
    """
    model.eval()
    indices = random.sample(range(len(X_validation)), n_samples)  # Randomly select n_samples indices
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))

    for i, idx in enumerate(indices):
        img = X_validation[idx].permute(1, 2, 0).numpy()
        true_label = labels_map[Y_validation[idx].item()]

        with torch.no_grad():
            output = model(X_validation[idx].unsqueeze(0).cuda())
            predicted_label = labels_map[torch.argmax(output).item()]

        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_label}\nPred: {predicted_label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_version = input("""
    Which model do you want to analyze?
    all: 0
    v1:  1
    v2:  2
    v3:  3
    """)

    if model_version not in ["0", "1", "2", "3"]:
        sys.exit()

    # Load validation data
    file_path = "input_data.npz"
    data = np.load(file_path)
    X_validation = torch.tensor(data['X_validation'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_validation = torch.tensor(data['Y_validation'], dtype=torch.long)
    labels_map = {0: "Ship", 1: "Iceberg"}

    models = {
        '1': SimpleCNN_v1().cuda(),
        '2': SimpleCNN_v2().cuda(),
        '3': SimpleCNN_v3().cuda()
    }

    if model_version == '0':
        history_files = [
            "training_history_model_v1.json",
            "training_history_model_v2.json",
            "training_history_model_v3.json",
        ]
        for version, model in models.items():
            model.load_state_dict(torch.load(f"simple_cnn_v{version}.pth"))
            print(f"Displaying predictions for Model v{version}:")
            display_predictions(model, X_validation, Y_validation, labels_map, n_samples=5)

    elif model_version in models:
        history_files = [f"training_history_model_v{model_version}.json"]
        model = models[model_version]
        model.load_state_dict(torch.load(f"simple_cnn_v{model_version}.pth"))
        print(f"Displaying predictions for Model v{model_version}:")
        display_predictions(model, X_validation, Y_validation, labels_map, n_samples=5)

    plot_history(history_files)
