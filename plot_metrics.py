import json
import matplotlib.pyplot as plt

def plot_history(history_file):
    # Load history from JSON file
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

    plt.figure(figsize=(15, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History")
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_precision, label="Train Precision")
    plt.plot(epochs, val_precision, label="Validation Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.title("Precision History")
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_f1, label="Train F1 Score")
    plt.plot(epochs, val_f1, label="Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score History")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    history_file = "training_history.json"  # Path to the saved history file
    plot_history(history_file)
