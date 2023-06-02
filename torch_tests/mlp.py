import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 784  # Input size for MNIST (28x28 images)
hidden_size = 256
num_classes = 10  # Output classes for MNIST (0-9 digits)
model = MLP(784, 256, 10)

# Compile the model with inductor backend
model = torch.compile(model, backend="torchmhlo")

# Evaluate the model
x = torch.randn(input_size)
output = model(x)
print(output)
