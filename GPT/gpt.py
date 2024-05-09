import torch
import torch.nn as nn
from attention import MultiHeadAttention
from gpt_config import GPT_CONFIG_124M as cfg
from transformer import GELU, LayerNorm, FeedForwardNetwork

# Transformer Block 
# ln -->> msa -->> drop -->> ln1 --> ffn --> drop
class TransformerBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            bias=cfg['qkv_bias']
        )
        self.ln1 = LayerNorm(embed_size=cfg['emb_dim'])
        self.ln2 = LayerNorm(embed_size=cfg['emb_dim'])
        self.ffn = FeedForwardNetwork(cfg)
        self.drop = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.drop(self.attention(self.ln1(x)))
        x = x + shortcut

        shortcut = x
        x = self.drop(self.ffn(self.ln2(x)))
        x = x + shortcut

        return x


# GPT model
class GPTModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        
        self.final_ln = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, inp_idx):
        batch_size, seq_len = inp_idx.shape
        tok_embs = self.tok_emb(inp_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=inp_idx.device))
        x = tok_embs + pos_embs
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_ln(x)

        logits = self.out_head(x)
        return logits
    


if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    print(batch)
  
    model = GPTModel(cfg=cfg)
    out = model(batch)

    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
