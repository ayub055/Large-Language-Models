import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, bias:bool):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # self.mask = torch.tril(torch.ones(context_length, context_length))

    def forward(self, x):
        b, num_of_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # print(queries.shape, keys.shape, values.shape)
        score = queries @ keys.transpose(1, 2)
        mask = torch.tril(torch.ones(num_of_tokens, num_of_tokens))
        score = score.masked_fill(mask == 0, -torch.inf)
        attention_weights = torch.softmax(score / keys.shape[-1]*0.5 , dim=-1) # batch, num_token, num_token
        attention_weights = self.dropout(attention_weights)
        
        context_vectors = attention_weights @ values
        return context_vectors     

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_out, d_out)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Shape : b, n_head, n_token, head_dim
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # print(queries.shape, keys.shape, values.shape)

        score = queries @ keys.transpose(2, 3)
        mask = torch.tril(torch.ones(num_tokens, num_tokens))
        score = score.masked_fill(mask==0, -torch.inf)
        attention_wts = torch.softmax(score / keys.shape[-1]**0.5 , dim=-1)
        attention_wts = self.dropout(attention_wts)

        # print(score.shape)
        context_vectors = attention_wts @ values
        context_vectors = context_vectors.transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)

        return self.proj(context_vectors)

        
if __name__ == "__main__":

    model = MultiHeadAttention(d_in=4, d_out=8, context_length=4, dropout=0.5, num_heads=4, bias=False)
    inputs = torch.randn(2, 6, 4)
    op = model(inputs)
    print(op.shape)


