import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.encoder = Encoder(
            self.input_dim, self.hidden_dim
        )
        self.out = nn.Linear(self.hidden_dim, self.z_dim)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        return self.out(hidden)




