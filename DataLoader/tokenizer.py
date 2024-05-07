import re
import torch
import tiktoken
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def vocabulary(data = 'the-verdict.txt'):
    with open(data, "r", encoding = "utf-8") as f:
        raw_text = f.read()
    
    preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed_text = [item for item in preprocessed_text if item.strip()]
    unique_words = sorted(set(preprocessed_text))    
    unique_words.extend(["<EOS>", "<UNK>"])
    vocab = {word:id for id, word in enumerate(unique_words)}
    # print(len(unique_words), len(preprocessed_text), len(vocab))
    return vocab

def create_dataloader(txt, batch_size, context_size, stride):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, context_size, stride)
    dataloader = DataLoader(dataset, batch_size, num_workers=0, shuffle=False)
    return dataloader
    
class SimpleTokenizer:
    def __init__(self, vocab):
        self.token_to_ids = vocab
        self.ids_to_token = {idx : token for token, idx in vocab.items()}
    
    def encode(self, text):
        preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_text = [token for token in preprocessed_text if token.strip()]
        preprocessed_text = [token if token in self.token_to_ids else "<UNK>" for token in preprocessed_text]
        token_ids = [self.token_to_ids[token] for token in preprocessed_text]
        return token_ids
        
    def decode(self, token_ids) :
        text = " ".join([self.ids_to_token[idx] for idx in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, context, stride):
        self.input_idxs = []
        self.target_idxs = []
        
        # Tokenize the entire dataset
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        num_token = len(token_ids)
        for i in range(0, num_token - context, stride):
            input_chunk = token_ids[i:i+context]
            output_chunk = token_ids[i+1:i+context+1]
            
            self.input_idxs.append(torch.tensor(input_chunk))
            self.target_idxs.append(torch.tensor(output_chunk))
    
    def __len__(self):
        return len(self.input_idxs)
    
    def __getitem__(self, idx):
        return self.input_idxs[idx], self.target_idxs[idx]
        

def token_emebedding(dataloader, vocab_size, context_size, output_dim):
    token_embd_layer = nn.Embedding(vocab_size, output_dim)
    pos_embd_layer = nn.Embedding(context_size, output_dim)
    
    for batch in dataloader:
        X, y = batch
        token_emb = token_embd_layer(X)
        pos_emb = pos_embd_layer(torch.arange(context_size))
        
        inp_emb = token_emb + pos_emb
        return inp_emb
        
    
    
if __name__ == "__main__": 
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    vocab_size = 50257
    output_dim = 256
    context_size = 1024
    
    dataloader = create_dataloader(raw_text, batch_size=8, context_size=4, stride=1)
    embeddings = token_emebedding(dataloader=dataloader, vocab_size=vocab_size,
                                  context_size=4, output_dim=output_dim)
    print(embeddings.shape)
    
    
    
    
    
    
    