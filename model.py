import torch.nn as nn
import torch

class SimpleCNN_v1(nn.Module):
    def __init__(self):
        super(SimpleCNN_v1, self).__init__()

        # Only one convolutional layer with very few filters
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # Input: (batch, 3, 75, 75) -> (batch, 8, 75, 75)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 8, 37, 37)
        )

        # Only one fully connected layer with very few neurons
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the input from (batch, 8, 37, 37)
            nn.Linear(8 * 37 * 37, 32),  # Only 32 units in the fully connected layer
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: (batch, 2), for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    

class SimpleCNN_v2(nn.Module):
    def __init__(self):
        super(SimpleCNN_v2, self).__init__()

        # Reduced the number of convolutional layers (removed one layer)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: (batch, 3, 75, 75)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 32, 37, 37)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 64, 18, 18)
        )

        # Keep the fully connected layers the same as before
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Output: (batch, 64 * 18 * 18)
            nn.Linear(64 * 18 * 18, 128),  # Keep the fully connected layer size the same
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: (batch, 2), for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class SimpleCNN_v3(nn.Module):
    def __init__(self):
        super(SimpleCNN_v3, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: (batch, 3, 75, 75)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 32, 37, 37)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 64, 18, 18)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (batch, 128, 9, 9)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Output: (batch, 128 * 9 * 9)
            nn.Linear(128 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: (batch, 2), for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    

if __name__ == "__main__":
    model_v1 = SimpleCNN_v1()
    model_v2 = SimpleCNN_v2()
    model_v3 = SimpleCNN_v3()
