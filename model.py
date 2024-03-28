import torch.nn as nn


# this defines our neural network model
class MlpModel(nn.Module):
    def __init__(self, num_input_features=387, num_classes=10, hidden_size_1=100, hidden_size_2=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_input_features, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, num_classes),
        )

    def forward(self, x):
        return self.network(x)
