import numpy as np
import matplotlib.pyplot as plt

def load_and_create_test_data(file_path, output_file):
    """
    Load the dataset, create a new test dataset, and save train, test, and validation data to a file.

    Parameters:
    file_path (str): Path to the input dataset file in `.npz` format.
    output_file (str): Path to save the updated dataset file.
    """
    data = np.load(file_path)

    # Load datasets
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_validation = data['X_validation']
    Y_validation = data['Y_validation']

    # Create new test dataset
    print("Creating new test dataset...")
    X_train, Y_train, X_test, Y_test = create_test_data(X_train, Y_train, samples_per_class=500)

    # Save the updated datasets
    np.savez(output_file, X_train=X_train, Y_train=Y_train,
             X_test=X_test, Y_test=Y_test,
             X_validation=X_validation, Y_validation=Y_validation)

    print(f"Datasets saved to {output_file}")

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation

def create_test_data(X_train, Y_train, samples_per_class):
    """
    Create a new test dataset with equal samples from each class.

    Parameters:
    X_train (ndarray): Training data.
    Y_train (ndarray): Training labels.
    samples_per_class (int): Number of samples per class to include in the test dataset.

    Returns:
    tuple: Updated training data and labels, and the newly created test data and labels.
    """
    # Initialize lists for test data
    X_test_list = []
    Y_test_list = []

    for label in [0, 1]:
        # Get indices of the current label
        indices = np.where(Y_train == label)[0]
        selected_indices = indices[:samples_per_class]

        # Append selected samples to test data
        X_test_list.append(X_train[selected_indices])
        Y_test_list.append(Y_train[selected_indices])

        # Remove selected samples from training data
        X_train = np.delete(X_train, selected_indices, axis=0)
        Y_train = np.delete(Y_train, selected_indices, axis=0)

    # Combine test data
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)

    return X_train, Y_train, X_test, Y_test

def analyze_data(X_train, Y_train, X_test, Y_test, X_validation, Y_validation):
    """
    Analyze the dataset, including training, validation, and test data.

    Parameters:
    X_train, Y_train, X_test, Y_test, X_validation, Y_validation: Dataset splits.
    """
    print("\nDataset Analysis:")
    print("----------------------")

    # Print shapes of all datasets
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_validation: {X_validation.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    # Analyze class distributions
    train_class_counts = {class_id: np.sum(Y_train == class_id) for class_id in np.unique(Y_train)}
    val_class_counts = {class_id: np.sum(Y_validation == class_id) for class_id in np.unique(Y_validation)}
    test_class_counts = {class_id: np.sum(Y_test == class_id) for class_id in np.unique(Y_test)}

    print(f"Class distribution in training data: {train_class_counts}")
    print(f"Class distribution in validation data: {val_class_counts}")
    print(f"Class distribution in test data: {test_class_counts}")

    # Analyze image dimensions
    print(f"Image dimensions (Height, Width, Channels): {X_train.shape[1:]}")
    image_size = X_train.shape[1] * X_train.shape[2]
    print(f"Number of pixels per image: {image_size}")

    # Check for missing values
    print("\nMissing Values:")
    print(f"X_train: {np.isnan(X_train).sum()} NaN values")
    print(f"Y_train: {np.isnan(Y_train).sum()} NaN values")
    print(f"X_validation: {np.isnan(X_validation).sum()} NaN values")
    print(f"Y_validation: {np.isnan(Y_validation).sum()} NaN values")
    print(f"X_test: {np.isnan(X_test).sum()} NaN values")
    print(f"Y_test: {np.isnan(Y_test).sum()} NaN values")

    # Visualize sample images from test data
    print("\nSample data points from test data:")
    labels_map = {0: "Ship", 1: "Iceberg"}
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i in range(5):
        ax = axes[i]
        ax.imshow(X_test[i])
        ax.set_title(f"Label: {labels_map[Y_test[i]]}")
        ax.axis('off')

    plt.show()

if __name__ == "__main__":
    input_file_path = "input_data.npz"
    output_file_path = "updated_data.npz"

    # Load, create test dataset, and save all splits
    X_train, Y_train, X_test, Y_test, X_validation, Y_validation = load_and_create_test_data(input_file_path, output_file_path)

    # Analyze the data
    analyze_data(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)
