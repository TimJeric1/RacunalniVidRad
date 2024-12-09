import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze_data(file_path):
    # Load dataset
    data = np.load(file_path)

    # Extract the data arrays
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_validation = data['X_validation']
    Y_validation = data['Y_validation']

    # Basic information
    print("Dataset Information:")
    print("----------------------")
    
    # Shape of the images
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_validation: {X_validation.shape}")

    # Number of classes in Y_train and Y_validation
    unique_train_classes = np.unique(Y_train)
    unique_val_classes = np.unique(Y_validation)
    
    print(f"Unique classes in Y_train: {unique_train_classes}")
    print(f"Unique classes in Y_validation: {unique_val_classes}")
    
    # Class distribution in the training and validation sets
    train_class_counts = {class_id: np.sum(Y_train == class_id) for class_id in unique_train_classes}
    val_class_counts = {class_id: np.sum(Y_validation == class_id) for class_id in unique_val_classes}
    
    print(f"Class distribution in training data: {train_class_counts}")
    print(f"Class distribution in validation data: {val_class_counts}")

    # Size of images
    print(f"Image dimensions: {X_train.shape[1:]} (Height, Width, Channels)")
    
    # Calculate number of pixels per image
    image_size = X_train.shape[1] * X_train.shape[2]  # Height * Width
    print(f"Number of pixels per image: {image_size}")
    
    # Check for any NaNs or missing values in the dataset
    print(f"Missing values in X_train: {np.isnan(X_train).sum()} NaN values")
    print(f"Missing values in Y_train: {np.isnan(Y_train).sum()} NaN values")
    print(f"Missing values in X_validation: {np.isnan(X_validation).sum()} NaN values")
    print(f"Missing values in Y_validation: {np.isnan(Y_validation).sum()} NaN values")
    
    # Display a few sample images and their labels
    print("Sample data points:")
    labels_map = {0: "Ship", 1: "Iceberg"}
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # Create a 1x5 grid for 5 images

    for i in range(5):  # Show 5 samples
        ax = axes[i]
        ax.imshow(X_train[i])  # Show the image
        ax.set_title(f"Label: {labels_map[Y_train[i]]}")  # Convert label to text
        ax.axis('off')  # Turn off axis

    plt.show()

# Main script
if __name__ == "__main__":
    # Path to the dataset
    file_path = "/home/timjeric/Downloads/racvid/RacunalniVid_Rad/DrugiDataset/input_data.npz"  # Update path if necessary
    
    # Load and analyze the data
    load_and_analyze_data(file_path)
