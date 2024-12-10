import sys
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from model import SimpleCNN_v1, SimpleCNN_v2, SimpleCNN_v3, SimpleCNN_v4  # Import the new model v4

def train_and_validate(train_loader, val_loader, model, criterion, optimizer, patience=5):
    """
    Train and validate the model with early stopping based on validation loss.

    Parameters:
    train_loader (DataLoader): DataLoader for training data.
    val_loader (DataLoader): DataLoader for validation data.
    model (nn.Module): PyTorch model to be trained.
    criterion (nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer.
    patience (int): Number of epochs with no improvement in validation loss to trigger early stopping.

    Returns:
    dict: Training and validation metrics history including loss, accuracy, precision, recall, and F1 score.
    """
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": []
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1000):
        model.train()
        epoch_loss, correct_train, total_train = 0, 0, 0
        y_true_train, y_pred_train = [], []

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == Y_batch).sum().item()
            total_train += Y_batch.size(0)
            y_true_train.extend(Y_batch.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_precision = precision_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        train_recall = recall_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        train_f1 = f1_score(y_true_train, y_pred_train, average='weighted')

        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == Y_batch).sum().item()
                total_val += Y_batch.size(0)
                y_true_val.extend(Y_batch.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
        val_f1 = f1_score(y_true_val, y_pred_val, average='weighted')

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_precision"].append(train_precision)
        history["val_precision"].append(val_precision)
        history["train_recall"].append(train_recall)
        history["val_recall"].append(val_recall)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return history

if __name__ == "__main__":
    model_version = input("""
    Which model do you want to train?
    all: 0
    v1:  1
    v2:  2
    v3:  3
    v4:  4  # Add v4 as an option
    """)

    if model_version not in ["0", "1", "2", "3", "4"]:
        sys.exit()

    file_path = "updated_data.npz"
    data = np.load(file_path)
    X_train_tensor = torch.tensor(data['X_train'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_train_tensor = torch.tensor(data['Y_train'], dtype=torch.long)
    X_val_tensor = torch.tensor(data['X_validation'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_val_tensor = torch.tensor(data['Y_validation'], dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    models = {
        '1': SimpleCNN_v1().cuda(),
        '2': SimpleCNN_v2().cuda(),
        '3': SimpleCNN_v3().cuda(),
        '4': SimpleCNN_v4().cuda()  # Add v4 model
    }

    if model_version == '0':
        for model_key, model in models.items():
            print(f"Training Model v{model_key}...")
            criterion = CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=0.001)
            history = train_and_validate(train_loader, val_loader, model, criterion, optimizer, patience=3)

            with open(f"training_history_model_v{model_key}.json", "w") as f:
                json.dump(history, f)
            torch.save(model.state_dict(), f"simple_cnn_v{model_key}.pth")
    else:
        model = models[model_version]
        print(f"Training Model v{model_version}...")
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        history = train_and_validate(train_loader, val_loader, model, criterion, optimizer, patience=3)

        with open(f"training_history_model_v{model_version}.json", "w") as f:
            json.dump(history, f)
        torch.save(model.state_dict(), f"simple_cnn_v{model_version}.pth")
