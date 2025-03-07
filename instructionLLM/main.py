import json
import os
from urllib import request
import torch
from torch.utils.data import Dataset
import tiktoken
from functools import partial
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
import torch.nn as nn
import logging
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import site
site.addsitedir('/Users/traiano/Desktop/tllm/trai-llm/.venv/lib/python3.12/site-packages')

from transformers import GPT2Tokenizer
from huggingface_hub import login
import requests
from pathlib import Path

print("Python path:", sys.executable)
print("PYTHONPATH:", os.environ.get('PYTHONPATH', ''))
print("sys.path:", sys.path)

logger = logging.getLogger(__name__)

def download_tokenizer_files():
    # Create cache directory
    cache_dir = Path(__file__).parent / "cache" / "gpt2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Files needed for GPT-2 tokenizer
    files = {
        'vocab.json': 'https://huggingface.co/gpt2/raw/main/vocab.json',
        'merges.txt': 'https://huggingface.co/gpt2/raw/main/merges.txt',
        'tokenizer.json': 'https://huggingface.co/gpt2/raw/main/tokenizer.json'
    }
    
    # Download each file
    for filename, url in files.items():
        filepath = cache_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url)
            response.raise_for_status()
            filepath.write_bytes(response.content)
            print(f"Downloaded {filename}")
    
    return str(cache_dir)

# Download files and get cache directory
try:
    cache_dir = download_tokenizer_files()
    print(f"Using cache directory: {cache_dir}")
    
    # Load tokenizer from local files
    tokenizer = GPT2Tokenizer.from_pretrained(
        cache_dir,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully!")

except Exception as e:
    print(f"Error during tokenizer setup: {e}")
    raise

# Configuration
cfg = {
    'max_length': 91,  # Based on max length seen in your data
    'batch_size': 4,   # From your data loader
    'vocab_size': tokenizer.vocab_size,
    # ... other config parameters ...
}

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# 1. Create a Config class to handle configuration
class TrainingConfig:
    def __init__(self, config_dict):
        # Model parameters
        self.n_embd = config_dict.get('n_embd', 768)
        self.n_head = config_dict.get('n_head', 12)
        self.n_layer = config_dict.get('n_layer', 12)
        self.vocab_size = config_dict.get('vocab_size', 50257)
        
        # Training parameters
        self.num_epochs = config_dict.get('num_epochs', 3)
        self.batch_size = config_dict.get('batch_size', 4)
        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        self.max_seq_length = config_dict.get('max_seq_length', 1024)
        self.dropout = config_dict.get('dropout', 0.1)
        
        # Optimizer parameters
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.beta1 = config_dict.get('beta1', 0.9)
        self.beta2 = config_dict.get('beta2', 0.999)
        
        # Device configuration
        self.device = config_dict.get('device', 'mps')

# 2. Update the training function to use the config object
def train_model_simple(model, train_loader, val_loader, optimizer, scheduler, cfg):
    train_losses = []
    val_losses = []
    tokens_seen = 0
    
    # Default accumulation steps if not in config
    gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', 4)
    optimizer.zero_grad()
    
    print(f"Starting training for {cfg['num_epochs']} epochs...")
    
    for epoch in range(cfg['num_epochs']):
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(cfg['device'])
            targets = targets.to(cfg['device'])
            
            # Forward pass with scaled loss
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Only optimize after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            tokens_seen += inputs.numel()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{cfg['num_epochs']}, Batch {batch_idx}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        # Validation step
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs, val_targets = val_batch
                val_inputs = val_inputs.to(cfg['device'])
                val_targets = val_targets.to(cfg['device'])
                val_loss = model(val_inputs)
                val_epoch_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} complete. Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()
    
    return train_losses, val_losses, tokens_seen

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.weight = assign(
            gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_key.weight = assign(
            gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_value.weight = assign(
            gpt.trf_blocks[b].attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.bias = assign(
            gpt.trf_blocks[b].attn.W_query.bias, q_b)
        gpt.trf_blocks[b].attn.W_key.bias = assign(
            gpt.trf_blocks[b].attn.W_key.bias, k_b)
        gpt.trf_blocks[b].attn.W_value.bias = assign(
            gpt.trf_blocks[b].attn.W_value.bias, v_b)

        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.weight = assign(
            gpt.trf_blocks[b].norm1.weight,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.bias = assign(
            gpt.trf_blocks[b].norm1.bias,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.weight = assign(
            gpt.trf_blocks[b].norm2.weight,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.bias = assign(
            gpt.trf_blocks[b].norm2.bias,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.weight = assign(gpt.final_norm.weight, params["g"])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.emb_dim = cfg["emb_dim"]
        self.context_length = cfg["context_length"]
        self.dropout = cfg["drop_rate"]
        assert self.emb_dim % self.n_heads == 0, "d_out must be divisible by n_heads"

        self.W_query = nn.Linear(self.emb_dim, self.emb_dim, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(self.emb_dim, self.emb_dim, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(self.emb_dim, self.emb_dim, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)  # Linear layer to combine head outputs
        self.register_buffer('mask', torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1))
        self.attn_dropout = nn.Dropout(cfg["drop_rate"])
        self.resid_dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.n_heads, self.emb_dim // self.n_heads)
        values = values.view(b, num_tokens, self.n_heads, self.emb_dim // self.n_heads)
        queries = queries.view(b, num_tokens, self.n_heads, self.emb_dim // self.n_heads)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.emb_dim)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

# class TransformerBlock(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.attn = MultiHeadAttention(cfg)
#         self.mlp = nn.Sequential(
#             nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
#             nn.GELU(),
#             nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
#             nn.Dropout(cfg["drop_rate"]),
#         )
#         self.norm1 = LayerNorm(cfg["emb_dim"])
#         self.norm2 = LayerNorm(cfg["emb_dim"])

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# The GPT model architecture implementation
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        self.is_classification = False
        self.debug = True
        logger.info(f"Initialized GPTModel with config: {cfg}")
        self.use_checkpointing = False
        
    def gradient_checkpointing_enable(self):
        self.use_checkpointing = True
    
    def forward(self, batch):
        # Check if we're getting a tuple (during training) or single tensor (during generation)
        if isinstance(batch, tuple):
            input_ids, target_ids = batch
        else:
            input_ids = batch
            target_ids = None  # During generation, we don't have targets
        
        # Move input_ids to the same device as the model
        input_ids = input_ids.to(self.tok_emb.weight.device)
        if target_ids is not None:
            target_ids = target_ids.to(self.tok_emb.weight.device)
        
        # Get embeddings
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(input_ids.shape[1], device=input_ids.device))
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.trf_blocks:
            x = block(x)
        
        # Get logits
        logits = self.out_head(x)
        
        if target_ids is not None:
            # Training mode - return loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target_ids = target_ids.view(B*T)
            loss = F.cross_entropy(logits, target_ids)
            return loss
        else:
            # Generation mode - return logits
            return logits


# Custom collate function to handle variable-length sequences and thus padd each sequence to the longest sequence in the batch
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] # Pads and prepares inputs
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # Removes extra padded token added earlier
        inputs_lst.append(inputs)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data: #Pretokenizes texts
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)


# generates the target token IDs from the input token IDs:
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
    ):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
        # Truncates the last token for inputs
        # Shifts +1 to the right for targets
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# final draft custom collate function
def custom_collate_fn(batch, **kwargs):
    if isinstance(batch[0], list) and isinstance(batch[0][0], (int, torch.Tensor)):
        # Convert to tensors but leave on CPU
        sequences = [torch.tensor(seq) if isinstance(seq, list) else seq 
                    for seq in batch]
        
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [torch.cat([seq, 
                                     torch.full((max_len - len(seq),), 
                                              tokenizer.eos_token_id)]) 
                          for seq in sequences]
        
        padded_batch = torch.stack(padded_sequences)
        inputs = padded_batch[:, :-1]
        targets = padded_batch[:, 1:]
        
        return inputs, targets

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.show()


# Alpaca prompt formatting
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
)
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
print("Example entry:\n", data[560])

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

# Partitioning the data into training, validation, and test sets
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)  # Use 10% for testing
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# Testing the custom collate function on a batch of 3 sequences
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

print("Using custom_collate_draft_1(batch):", custom_collate_draft_1(batch))

inputs, targets = custom_collate_draft_2(batch)
print("Using custom_collate_draft_2(batch) - inputs:", inputs)
print("Using custom_collate_draft_2(batch) - targets:", targets)

inputs, targets = custom_collate_fn(batch)
print("Using custom_collate_fn(batch) - inputs:", inputs)
print("Using custom_collate_fn(batch) - targets:", targets)

# Select the optimal device for this host machine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print("Device:", device)

# Redefine the custom collate function with the device argument
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# Initializing the data loaders
num_workers = 0
batch_size = 4

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# examine the dimensions of the input and target batches generated by the training loader:
print("Train loader:")
for inputs, targets in train_loader:
    print("Inputs shape: ", inputs.shape, "Targets shape: ", targets.shape)

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True,         # Query-key-value bias
    "batch_size": 4,  # Reduce from 8 to 4 or smaller
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

# assess the pretrained LLM's performance on one of the validation tasks by comparing its output to the expected response
# baseline understanding of how well the model performs on an instruction-following task right out of the box, prior to fine-tuning

torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)

generated_text = token_ids_to_text(token_ids, tokenizer)

# To isolate the model's response text, we need to subtract the length of the input instruction from the start of the generated_text:
response_text = generated_text[len(input_text):].strip()
# At this point, we can compare the response_text to the expected response. We can't expect the model to perform well at this point as it has not been trained yet.
print(response_text)

# Next steps: Fine Tuning the LLM model on instruction response data.
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
)
print("Before training: Training loss:", train_loss)
print("Before training: Validation loss:", val_loss)

# Now train the model: Instruction fine-tune a pretrained LLM

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2

# Define the configuration dictionary with all necessary parameters
cfg = {
    'batch_size': 4,  # From your data loader
    'context_length': 91,  # Maximum sequence length in your data
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'warmup_steps': 1000,
    'vocab_size': 50257,  # GPT-2 vocab size
    'model_dim': 768,
    'num_heads': 12,
    'num_layers': 12,
    'dropout': 0.1,
    'device': 'mps'  # From your logs
}

# Modify the function call to only pass the required positional arguments
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Example scheduler
device = torch.device('mps')  # Your device

train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,  # Make sure this is actually a scheduler object
    cfg=cfg
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


# plot the training and validation losses over the course of the training process
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# 3. Update the main training setup
def main():
    # Define configuration
    config_dict = {
        'n_embd': 768,
        'n_head': 12,
        'n_layer': 12,
        'vocab_size': 50257,
        'num_epochs': 3,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'max_seq_length': 1024,
        'dropout': 0.1,
        'weight_decay': 0.01,
        'device': 'mps'
    }
    
    # Create config object
    cfg = TrainingConfig(config_dict)
    
    # Model setup
    model = GPTModel(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Training
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg
    )

if __name__ == "__main__":
    main()