import sys
import json
import matplotlib.pyplot as plt

def plot_history(history_files):
    # Define a consistent set of colors for the models
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # You can extend this list if needed

    # Create separate figures for train and validation
    fig_train, axes_train = plt.subplots(2, 2, figsize=(15, 10))
    fig_val, axes_val = plt.subplots(2, 2, figsize=(15, 10))

    # Iterate through each model's history file and plot
    for i, history_file in enumerate(history_files):
        with open(history_file, "r") as f:
            history = json.load(f)

        # Extract metrics from the history
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        train_accuracy = history["train_accuracy"]
        val_accuracy = history["val_accuracy"]
        train_precision = history["train_precision"]
        val_precision = history["val_precision"]
        train_recall = history["train_recall"]
        val_recall = history["val_recall"]
        train_f1 = history["train_f1"]
        val_f1 = history["val_f1"]
        epochs = range(1, len(train_loss) + 1)

        # Use the same color for each model across the two figures
        color = colors[i % len(colors)]

        # Plot Train Metrics
        axes_train[0, 0].plot(epochs, train_loss, label=f"Train Loss (Model {i+1})", color=color)
        axes_train[0, 1].plot(epochs, train_accuracy, label=f"Train Accuracy (Model {i+1})", color=color)
        axes_train[1, 0].plot(epochs, train_precision, label=f"Train Precision (Model {i+1})", color=color)
        axes_train[1, 1].plot(epochs, train_f1, label=f"Train F1 Score (Model {i+1})", color=color)

        # Plot Validation Metrics
        axes_val[0, 0].plot(epochs, val_loss, label=f"Val Loss (Model {i+1})", color=color)
        axes_val[0, 1].plot(epochs, val_accuracy, label=f"Val Accuracy (Model {i+1})", color=color)
        axes_val[1, 0].plot(epochs, val_precision, label=f"Val Precision (Model {i+1})", color=color)
        axes_val[1, 1].plot(epochs, val_f1, label=f"Val F1 Score (Model {i+1})", color=color)

    # Set the titles, labels, and legends for the train plots
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

    # Set the titles, labels, and legends for the validation plots
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

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.figure(fig_train.number)
    plt.show()
    plt.figure(fig_val.number)
    plt.show()

# Main script
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

    if model_version == '0':  # If the user chooses "all"
        history_files = [
            "training_history_model_v1.json",  # History for model v1
            "training_history_model_v2.json",  # History for model v2
            "training_history_model_v3.json",  # History for model v3
        ]
    elif model_version == '1':
        history_files = ["training_history_model_v1.json"]  # Only model v1
    elif model_version == '2':
        history_files = ["training_history_model_v2.json"]  # Only model v2
    elif model_version == '3':
        history_files = ["training_history_model_v3.json"]  # Only model v3

    plot_history(history_files)
