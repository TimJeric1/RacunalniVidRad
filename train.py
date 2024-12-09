import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from model import SimpleCNN

# Function for training and validation with early stopping
def train_and_validate(train_loader, val_loader, model, criterion, optimizer, patience=5):
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": []
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1000):  # Arbitrarily large number of epochs; we stop early when needed
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        y_true_train = []
        y_pred_train = []

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == Y_batch).sum().item()
            total_train += Y_batch.size(0)

            # Collect true and predicted values for metrics
            y_true_train.extend(Y_batch.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Calculate precision, recall, and F1 for training
        train_precision = precision_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        train_recall = recall_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        train_f1 = f1_score(y_true_train, y_pred_train, average='weighted')

        # Validation Phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():  # No need to track gradients during validation
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.cuda(), Y_batch.cuda()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == Y_batch).sum().item()
                total_val += Y_batch.size(0)

                # Collect true and predicted values for metrics
                y_true_val.extend(Y_batch.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        # Calculate precision, recall, and F1 for validation
        val_precision = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
        val_recall = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
        val_f1 = f1_score(y_true_val, y_pred_val, average='weighted')

        # Store metrics in history
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

        print(f"Epoch {epoch+1}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset counter if we improve
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break  # Stop training if validation loss hasn't improved for 'patience' epochs

    return history

# Main script
if __name__ == "__main__":
    # Load dataset (Assume `load_data` properly loads and preprocesses the data)
    # file_path = "input_data.npz"
    file_path = "/home/timjeric/Downloads/racvid/RacunalniVid_Rad/DrugiDataset/input_data.npz"
    data = np.load(file_path)  # Use this if loading from a .pth or change to your data
    X_train_tensor = torch.tensor(data['X_train'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_train_tensor = torch.tensor(data['Y_train'], dtype=torch.long)
    X_val_tensor = torch.tensor(data['X_validation'], dtype=torch.float32).permute(0, 3, 1, 2)
    Y_val_tensor = torch.tensor(data['Y_validation'], dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, criterion, and optimizer
    model = SimpleCNN().cuda()  # Use GPU if available
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model and get the history with early stopping
    history = train_and_validate(train_loader, val_loader, model, criterion, optimizer, patience=5)

    # Save history to a JSON file
    with open("training_history.json", "w") as f:
        json.dump(history, f)

    # Save the model
    torch.save(model.state_dict(), "simple_cnn.pth")
