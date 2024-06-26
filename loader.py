import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTDataset(nn.Module):
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        tokenizer_ids = tokenizer.encode(txt)

        for i in range(0, len(tokenizer_ids) - max_length, stride):
            inputs = tokenizer_ids[i : i + max_length]
            targets = tokenizer_ids[i+1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(inputs))
            self.target_ids.append(torch.tensor(targets))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_loader(txt, batch_size, max_length, stride, shuffle, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

