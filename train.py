import torch 
import tiktoken
import torch.nn as nn
from gpt_model import GPTModel
from loader import create_loader
import matplotlib.pyplot as plt
from util import generate_print, calculate_batch_loss, evaluate_model, plot_losses

def train(model, train_loader, val_loader, optimizer, device, num_epochs, 
          eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_batch_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): | " 
                      f"Train loss {train_loss:.3f} | Val loss {val_loss:.3f}")
        
        generate_print(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_token_seen


def main(gpt_config, hyprams, seed, device):
    torch.manual_seed(seed)

    with open('the-verdict.txt','r', encoding='utf-8') as f:
        raw_text = f.read()

    model = GPTModel(gpt_config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyprams['learning_rate'], weight_decay=hyprams['weight_decay'])

    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))

    train_loader = create_loader(
        raw_text[:split_idx],
        batch_size=hyprams["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_loader(
        raw_text[split_idx:],
        batch_size=hyprams["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    tokenizer = tiktoken.get_encoding('gpt2')
    train_losses, val_losses, tokens_seen = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=hyprams["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    HYPER_PARAMETERS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    seed = 123
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, HYPER_PARAMETERS, seed, device)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, HYPER_PARAMETERS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, torch.tensor(tokens_seen).cpu(), torch.tensor(train_losses).cpu(), torch.tensor(val_losses).cpu())
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "gpt_124M.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("gpt_124M.pth"))
    