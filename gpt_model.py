# GPT_CONFIG_124M = {
#         "vocab_size": 50257,     # Vocabulary size
#         "context_length": 1024,  # Context length
#         "emb_dim": 768,          # Embedding dimension
#         "n_heads": 12,           # Number of attention heads
#         "n_layers": 12,          # Number of layers
#         "drop_rate": 0.1,        # Dropout rate
#         "qkv_bias": False        # Query-Key-Value bias
#     }

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(1, emb_dim))
        self.shift = nn.Parameter(torch.zeros(1, emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * out + self.shift

class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        t = x + 0.044715 * torch.pow(x, 3)
        t = torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * t)
        return 0.5 * x * t
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout, context_length, qkv_bias=False) -> None:
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads
        self.dropout = dropout
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.last_proj = nn.Linear(d_out, d_out)


    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention_score = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_score = attention_score.masked_fill(mask_bool, -torch.inf)

        attention_wts = torch.softmax(attention_score/keys.shape[-1] ** 0.5, dim=-1)
        context_vectors = attention_wts @ values
        context_vectors = context_vectors.transpose(1, 2)
        context_vectors = context_vectors.reshape(b, num_tokens, self.d_out)
        return self.last_proj(context_vectors)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=cfg['emb_dim'], out_features=4*cfg['emb_dim']),
            GELU(),
            nn.Linear(in_features=4*cfg['emb_dim'], out_features=cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)\

class TransformerBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ffn = FeedForwardNetwork(cfg)
        self.ln1 = LayerNorm(cfg['emb_dim'])
        self.ln2 = LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.drop(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        self.pos_emb = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_ln = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_embds = self.tok_emb(x)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embds + pos_embs
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_ln(x)
        
        logits = self.out_head(x)
        return logits



