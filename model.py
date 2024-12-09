import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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
    model = SimpleCNN()
    print(model)
