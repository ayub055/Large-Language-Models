import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        t = x + 0.044715*(torch.pow(x, 3))
        t = torch.sqrt(torch.tensor(2.0/torch.pi)) * t
        t = 1 + torch.tanh(t)
        t = 0.5 * x* t
        return t


class LayerNorm(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(1, embed_size))
        self.shift = nn.Parameter(torch.zeros(1, embed_size))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out_norm = (x - mean) / (torch.sqrt(var + self.eps))
        return self.scale*out_norm + self.shift


class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)
    

if __name__ == "__main__":
    print("Nothing")