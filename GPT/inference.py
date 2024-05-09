import torch
import torch.nn as nn
from gpt import GPTModel

def generate_text(model, idx, max_tokens, context_size):
    for _ in range(max_tokens):

        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx