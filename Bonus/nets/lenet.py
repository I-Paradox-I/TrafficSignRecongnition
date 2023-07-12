import torch
import torch.nn as nn

# Define the LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16 * 56 * 56, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        # 224x224x3 -> 224x224x6
        x = nn.functional.relu(self.conv1(x))
        # 224x224x6 -> 112x112x6
        x = nn.functional.max_pool2d(x, 2)
        # 112x112x6 -> 112x112x16
        x = nn.functional.relu(self.conv2(x))
        # 112x112x16 -> 56x56x16
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
