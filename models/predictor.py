import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, latent_channels, hidden_channels, z_dim=128):
        super(Predictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(latent_channels + z_dim + 4, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, latent_channels + 4)
        )
        self.z = torch.nn.Parameter(torch.randn(z_dim))
        
        self.reset_parameters()

    def forward(self, x):
        x = torch.cat([x, self.z.expand(x.size(0), -1)], dim=1)
        return self.pred(x)
    
    def reset_parameters(self):
        for p in self.pred:
            if isinstance(p, nn.Linear):
                nn.init.xavier_normal_(p.weight)
                nn.init.zeros_(p.bias)