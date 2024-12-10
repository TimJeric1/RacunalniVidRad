import torch.nn as nn
import torch

class SimpleCNN_v1(nn.Module):
    """
    A simple convolutional neural network with one convolutional layer and 
    one fully connected layer.

    Layers:
    - Convolutional Layer: 8 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Fully Connected Layer: Flattens the input and connects to 32 neurons.
    - Output Layer: Fully connected layer with 2 neurons for binary classification.

    Input Shape: (batch, 3, 75, 75)
    Output Shape: (batch, 2)
    """
    def __init__(self):
        super(SimpleCNN_v1, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 37 * 37, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SimpleCNN_v2(nn.Module):
    """
    A simple convolutional neural network with two convolutional layers and 
    two fully connected layers.

    Layers:
    - First Convolutional Layer: 32 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Second Convolutional Layer: 64 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Fully Connected Layer: Flattens the input and connects to 128 neurons.
    - Output Layer: Fully connected layer with 2 neurons for binary classification.

    Input Shape: (batch, 3, 75, 75)
    Output Shape: (batch, 2)
    """
    def __init__(self):
        super(SimpleCNN_v2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SimpleCNN_v3(nn.Module):
    """
    A simple convolutional neural network with three convolutional layers and 
    two fully connected layers.

    Layers:
    - First Convolutional Layer: 32 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Second Convolutional Layer: 64 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Third Convolutional Layer: 128 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Fully Connected Layer: Flattens the input and connects to 128 neurons.
    - Output Layer: Fully connected layer with 2 neurons for binary classification.

    Input Shape: (batch, 3, 75, 75)
    Output Shape: (batch, 2)
    """
    def __init__(self):
        super(SimpleCNN_v3, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SimpleCNN_v4(nn.Module):
    """
    A simple convolutional neural network with three convolutional layers and 
    two fully connected layers, with dropout added for regularization.

    Layers:
    - First Convolutional Layer: 32 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Second Convolutional Layer: 64 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Third Convolutional Layer: 128 filters, kernel size 3x3, stride 1, padding 1.
    - Activation: ReLU.
    - MaxPooling: Kernel size 2x2, stride 2.
    - Fully Connected Layer: Flattens the input and connects to 128 neurons.
    - Dropout: Dropout layer with probability 0.3 for regularization.
    - Output Layer: Fully connected layer with 2 neurons for binary classification.

    Input Shape: (batch, 3, 75, 75)
    Output Shape: (batch, 2)
    """
    def __init__(self):
        super(SimpleCNN_v4, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout with 30% probability
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    model_v1 = SimpleCNN_v1()
    model_v2 = SimpleCNN_v2()
    model_v3 = SimpleCNN_v3()
    model_v4 = SimpleCNN_v4()
