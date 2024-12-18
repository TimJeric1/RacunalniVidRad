import sys
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import SimpleCNN_v1, SimpleCNN_v2, SimpleCNN_v3, SimpleCNN_v4, ResNet, EfficientNet  # Import all models
import random


def evaluate_on_test_data(model, X_test, Y_test, labels_map):
    """
    Evaluate the model on the test data and return metrics.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for idx in range(len(X_test)):
            output = model(X_test[idx].unsqueeze(0).cuda())
            predicted_label = torch.argmax(output).item()
            y_pred.append(predicted_label)
            y_true.append(Y_test[idx].item())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1


def plot_test_metrics(models, X_test, Y_test, labels_map):
    """
    Plot bar graphs for test metrics (Accuracy, Precision, Recall, F1 Score) for each model,
    with the actual values displayed on top of the bars.
    """
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for model in models.values():
        accuracy, precision, recall, f1 = evaluate_on_test_data(model, X_test, Y_test, labels_map)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracies, precisions, recalls, f1_scores]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Added one more color for the new model

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(models.keys(), values[i], color=colors[:len(models)])
        ax.set_title(f"{metric} of Test Data")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model Version")
        ax.set_ylim(0, 1)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_history(history_files):
    """
    Plots the training and validation metrics for one or more models.
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


def display_predictions(models, X_test, Y_test, labels_map, n_samples=5):
    """
    Displays n_samples randomly selected images from the test set 
    with their true and predicted labels.
    """
    model_names = list(models.keys())
    indices = random.sample(range(len(X_test)), n_samples)

    fig, axes = plt.subplots(len(models), n_samples, figsize=(5 * n_samples, 4 * len(models)))

    if n_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for j, (model_name, model) in enumerate(models.items()):
        for i, idx in enumerate(indices):
            img = X_test[idx].permute(1, 2, 0).numpy()
            true_label = labels_map[Y_test[idx].item()]

            with torch.no_grad():
                output = model(X_test[idx].unsqueeze(0).cuda())
                predicted_label = labels_map[torch.argmax(output).item()]
            text_color = 'green' if predicted_label == true_label else 'red'
            
            # Display the image with a border
            axes[j, i].imshow(img)
            axes[j, i].axis('off')  # Hide the axis

            # Add a border around each image
            for spine in axes[j, i].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            
            # Title with improved font size, and contrast for better readability
            axes[j, i].set_title(
                f"Model: {model_name}\nTrue: {true_label}\nPred: {predicted_label}",
                fontsize=12,
                color=text_color,
                fontweight='bold'
            )
        
    # Increase spacing between subplots for better clarity
    fig.suptitle(f"Predictions for {n_samples} Test Images Across Models", fontsize=18, fontweight='bold')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ensure title doesn't overlap
    plt.show()



if __name__ == "__main__":
    model_version = input("""
    Which model do you want to analyze?
    all: 0
    v1:  1
    v2:  2
    v3:  3
    v4:  4
    v5:  5  # ResNet-based model
    v6:  6  # EfficientNet-B0 model
    """)

    if model_version not in ["0", "1", "2", "3", "4", "5", "6"]:
        sys.exit()

    file_path = "updated_data.npz"
    data = np.load(file_path)
    X_validation = torch.tensor(data['X_validation'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_validation = torch.tensor(data['Y_validation'], dtype=torch.long)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_test = torch.tensor(data['Y_test'], dtype=torch.long)
    labels_map = {0: "Ship", 1: "Iceberg"}

    # Define models, including ResNet and EfficientNet-B0
    models = {
        '1': SimpleCNN_v1().cuda(),
        '2': SimpleCNN_v2().cuda(),
        '3': SimpleCNN_v3().cuda(),
        '4': SimpleCNN_v4().cuda(),
        '5': ResNet().cuda(),
        '6': EfficientNet().cuda()  # EfficientNet-B0 added
    }

    if model_version == '0':
        history_files = [
            "training_history_model_v1.json",
            "training_history_model_v2.json",
            "training_history_model_v3.json",
            "training_history_model_v4.json",
            "training_history_model_v5.json",
            "training_history_model_v6.json"  # EfficientNet-B0 history
        ]
        for version, model in models.items():
            model.load_state_dict(torch.load(f"simple_cnn_v{version}.pth"))
            print(f"Displaying predictions for Model v{version}:")
        
        plot_test_metrics(models, X_test, Y_test, labels_map)
        display_predictions(models, X_test, Y_test, labels_map, n_samples=5)

    elif model_version in models:
        history_files = [f"training_history_model_v{model_version}.json"]
        model = models[model_version]
        model.load_state_dict(torch.load(f"simple_cnn_v{model_version}.pth"))
        print(f"Displaying predictions for Model v{model_version}:")
        
        plot_test_metrics({model_version: model}, X_test, Y_test, labels_map)
        display_predictions({model_version: model}, X_test, Y_test, labels_map, n_samples=5)

    plot_history(history_files)
