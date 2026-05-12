import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x):
        return x + self.block(x)
    
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_blocks=3, dropout=0.1):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return self.output_layer(x)