import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze_data(file_path):
    """
    Load and analyze a dataset containing image data and corresponding labels.

    Parameters:
    file_path (str): Path to the dataset file in `.npz` format. The file is expected 
                     to contain the following arrays:
                     - X_train: Training images
                     - Y_train: Training labels
                     - X_validation: Validation images
                     - Y_validation: Validation labels

    This function performs the following tasks:
    1. Loads the dataset and extracts the data arrays.
    2. Prints basic dataset information including the shapes of arrays, 
       unique classes, and class distributions.
    3. Displays information about image dimensions and calculates the 
       number of pixels per image.
    4. Checks for any missing values in the dataset.
    5. Visualizes a subset of sample images with their labels.

    Returns:
    None
    """
    data = np.load(file_path)

    X_train = data['X_train']
    Y_train = data['Y_train']
    X_validation = data['X_validation']
    Y_validation = data['Y_validation']

    print("Dataset Information:")
    print("----------------------")
    
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_validation: {X_validation.shape}")

    unique_train_classes = np.unique(Y_train)
    unique_val_classes = np.unique(Y_validation)
    
    print(f"Unique classes in Y_train: {unique_train_classes}")
    print(f"Unique classes in Y_validation: {unique_val_classes}")
    
    train_class_counts = {class_id: np.sum(Y_train == class_id) for class_id in unique_train_classes}
    val_class_counts = {class_id: np.sum(Y_validation == class_id) for class_id in unique_val_classes}
    
    print(f"Class distribution in training data: {train_class_counts}")
    print(f"Class distribution in validation data: {val_class_counts}")

    print(f"Image dimensions: {X_train.shape[1:]} (Height, Width, Channels)")
    
    image_size = X_train.shape[1] * X_train.shape[2]
    print(f"Number of pixels per image: {image_size}")
    
    print(f"Missing values in X_train: {np.isnan(X_train).sum()} NaN values")
    print(f"Missing values in Y_train: {np.isnan(Y_train).sum()} NaN values")
    print(f"Missing values in X_validation: {np.isnan(X_validation).sum()} NaN values")
    print(f"Missing values in Y_validation: {np.isnan(Y_validation).sum()} NaN values")
    
    print("Sample data points:")
    labels_map = {0: "Ship", 1: "Iceberg"}
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i in range(5):
        ax = axes[i]
        ax.imshow(X_train[i])
        ax.set_title(f"Label: {labels_map[Y_train[i]]}")
        ax.axis('off')

    plt.show()

if __name__ == "__main__":
    file_path = "input_data.npz"
    load_and_analyze_data(file_path)
