import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, latent_channels, hidden_channels):
        super(Predictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        self.reset_parameters()

    def forward(self, x):
        return self.pred(x)
    
    def reset_parameters(self):
        for p in self.pred:
            if isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)
                nn.init.zeros_(p.bias)