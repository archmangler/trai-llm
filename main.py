import urllib.request
import re
from importlib.metadata import version
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import torch.nn as nn
from gpt_download import download_and_load_gpt2
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F
import time
import logging
from datetime import datetime

# Set up logging
log_filename = f"spam_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log system information
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python path: {sys.path}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

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
        self.n_heads = cfg.get("n_heads", 12)
        self.emb_dim = cfg["emb_dim"]
        assert self.emb_dim % self.n_heads == 0
        self.head_dim = self.emb_dim // self.n_heads
        self.attn_dropout = nn.Dropout(cfg["drop_rate"])
        self.resid_dropout = nn.Dropout(cfg["drop_rate"])
        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=False)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert label to tensor
        label_tensor = torch.tensor(1 if label == 'spam' else 0, dtype=torch.long)
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with zeros if needed
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens), label_tensor

def plot_values( epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # Plots training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    # Creates a second x-axis for examples seen
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    # Invisible plot for aligning ticks
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
    # Adjusts layout to make room

def calc_loss_loader(data_loader, model, device, num_batches=None):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    criterion = nn.CrossEntropyLoss()
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)
                if model.is_classification:
                    loss = criterion(logits, target_batch)
                else:
                    # For language modeling, we need to reshape the output
                    loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            total_loss += loss.item() * input_batch.size(0)
            total_examples += input_batch.size(0)
        else:
            break
    return total_loss / total_examples

# Calculating the classification accuracy
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)
                # Handle both classification and language modeling outputs
                if model.is_classification:
                    # Classification case: logits shape is (batch_size, num_classes)
                    predicted_labels = torch.argmax(logits, dim=-1)
                else:
                    # Language modeling case: logits shape is (batch_size, seq_len, vocab_size)
                    logits = logits[:, -1, :]
                    predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                    (predicted_labels == target_batch).sum().item())
        else: 
            break
    return correct_predictions / num_examples

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # Dropout is disabled during evaluation for stable, reproducible results.
    with torch.no_grad(): # Disables gradient tracking, which is not required during evaluation, to reduce the computational overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

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
        
    def forward(self, in_idx):
        if self.debug:
            logger.debug(f"\nGPTModel forward pass:")
            logger.debug(f"Input shape: {in_idx.shape}")
            
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        
        if self.debug:
            logger.debug(f"Token embeddings shape: {tok_embeds.shape}")
            logger.debug(f"Position embeddings shape: {pos_embeds.shape}")
            
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        if self.is_classification:
            if self.debug:
                logger.debug("Classification mode: using last token representation")
                logger.debug(f"Pre-classification shape: {x.shape}")
            x = x[:, -1, :]
            if self.debug:
                logger.debug(f"Post-classification shape: {x.shape}")
        
        logits = self.out_head(x)
        if self.debug:
            logger.debug(f"Final output shape: {logits.shape}")
            if self.is_classification:
                logger.debug(f"Classification logits: {logits}")
        
        return logits

# Load and preprocess the spam dataset
def load_spam_dataset(file_path):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            data.append(text)
            labels.append(label)
    return data, labels

def split_dataset(data, labels, train_ratio=0.8, val_ratio=0.1):
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return {
        'train': (
            [data[i] for i in train_indices],
            [labels[i] for i in train_indices]
        ),
        'val': (
            [data[i] for i in val_indices],
            [labels[i] for i in val_indices]
        ),
        'test': (
            [data[i] for i in test_indices],
            [labels[i] for i in test_indices]
        )
    }
# Plot training history
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def classify_review(
        text, model, tokenizer, device, max_length: int = 128,
        pad_token_id: int = 50256, debug: bool = True
        ):
        if debug:
            logger.debug("\nDebug Information:")
            logger.debug(f"Input text length: {len(text)}")
            logger.debug(f"Model classification mode: {model.is_classification}")
        
        model.eval()
        input_ids = tokenizer.encode(text)
        if debug:
            logger.debug(f"Tokenized length: {len(input_ids)}")
        
        supported_context_length = model.pos_emb.weight.shape[1]
        effective_max_length = max_length if max_length is not None else supported_context_length
        input_ids = input_ids[:min(effective_max_length, supported_context_length)]
        
        if debug:
            logger.debug(f"Truncated length: {len(input_ids)}")
        
        padding_length = effective_max_length - len(input_ids)
        input_ids += [pad_token_id] * padding_length
        
        if debug:
            logger.debug(f"Final input length with padding: {len(input_ids)}")
        
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
        if debug:
            logger.debug(f"Input tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            logits = model(input_tensor)
            if debug:
                logger.debug(f"Output logits shape: {logits.shape}")
                logger.debug(f"Logits values: {logits}")
                softmax_probs = torch.softmax(logits, dim=-1)
                logger.debug(f"Softmax probabilities: {softmax_probs}")
            
            predicted_label = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_label].item()
            
            if debug:
                logger.debug(f"Predicted label: {predicted_label}")
                logger.debug(f"Confidence: {confidence:.4f}")
        
        result = "spam" if predicted_label == 1 else "not spam"
        logger.info(f"Classification result: {result} (confidence: {confidence:.4f})")
        return result, confidence

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    if model.debug:
        logger.debug(f"\nCalc Loss Batch Debug:")
        logger.debug(f"Input batch shape: {input_batch.shape}")
        logger.debug(f"Target batch shape: {target_batch.shape}")
    
    logits = model(input_batch)
    
    if model.debug:
        logger.debug(f"Output logits shape: {logits.shape}")
    
    if model.is_classification:
        # Classification case: logits shape is (batch_size, num_classes)
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
    else:
        # Language modeling case: logits shape is (batch_size, seq_len, vocab_size)
        logits = logits[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
    
    if model.debug:
        logger.debug(f"Loss value: {loss.item():.4f}")
    
    return loss

# train classifier simple
def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                          num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    
    logger.info("Starting training...")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Evaluation frequency: {eval_freq} steps")
    logger.info(f"Evaluation iterations: {eval_iter}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                logger.info(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
                
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        logger.info(
            f"Training accuracy: {train_accuracy*100:.2f}% | "
            f"Validation accuracy: {val_accuracy*100:.2f}%"
        )
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    logger.info("Training completed!")
    return train_losses, val_losses, train_accs, val_accs, examples_seen


# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with the base configuration
BASE_CONFIG = {
    "vocab_size": 50257,  # GPT-2's vocabulary size
    "emb_dim": 768,      # Embedding dimension
    "context_length": 1024,  # Maximum sequence length
    "n_layers": 12,      # Number of transformer blocks
    "drop_rate": 0.1,    # Dropout rate
    "n_heads": 12       # Number of attention heads
}

model = GPTModel(BASE_CONFIG)
model.to(device)

# Freeze the model in selected layers
for param in model.parameters():
    param.requires_grad = False


# Replace the out head:
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

model.is_classification = True  # Set model to classification mode

# Selectively freeze the layers: Final Transformer block and normalisation layer:
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

# Example of how to use the model for classification
# Note: You'll need to replace this with your actual data loading code
example_input = torch.randint(0, BASE_CONFIG["vocab_size"], (1, 10))  # Batch size 1, sequence length 10
example_input = example_input.to(device)

with torch.no_grad():
    outputs = model(example_input)

print("Example outputs shape:", outputs.shape)
print("Example classification logits:", outputs)

# Note: To use the model for training, you'll need to:
# 1. Create your SpamDataset class
# 2. Create train_loader and val_loader
# 3. Define your loss function and optimizer
# 4. Implement the training loop

# The following code should be uncommented and used once you have your data loaders ready:
"""
# Calculate and print training accuracy
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)
print("Training accuracy:", train_accuracy)
""" 

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load the dataset
data_file_path = Path("sms_spam_collection/SMSSpamCollection.tsv")
texts, labels = load_spam_dataset(data_file_path)

# Split the dataset
np.random.seed(42)
splits = split_dataset(texts, labels)

# Create datasets
train_dataset = SpamDataset(splits['train'][0], splits['train'][1], tokenizer)
val_dataset = SpamDataset(splits['val'][0], splits['val'][1], tokenizer)
test_dataset = SpamDataset(splits['test'][0], splits['test'][1], tokenizer)

# Create data loaders
batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# Training configuration
num_epochs = 5
learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=learning_rate
)

# Training loop
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            log_msg = f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            logger.info(log_msg)
    
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Average training loss: {avg_loss:.4f}")
    return avg_loss

# Training and evaluation
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Training
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = calc_loss_loader(val_loader, model, device)
        val_accuracy = calc_accuracy_loader(val_loader, model, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Plot training history
# plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

# Final evaluation on test set  
model.eval()
with torch.no_grad():
    test_loss = calc_loss_loader(test_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}") 

# Save the model
torch.save(model.state_dict(), "gpt2_spam_classifier.pth")

# Load the model
model = GPTModel(BASE_CONFIG)

# Configure model for classification before loading state dict
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
model.is_classification = True

# Now load the state dict
model.load_state_dict(torch.load("gpt2_spam_classifier.pth"))
model.to(device)

# Test the model with example texts
text_1 = (
    "You are a winner you have been specifically"
    "selected to receive $1000 cash or a $2000 award."
)

print("\nTesting spam detection:")
print("Text 1 (spam example):", text_1)
prediction, confidence = classify_review(
    text_1, model, tokenizer, device,
    max_length=train_dataset.max_length
)
print(f"Classification: {prediction} (confidence: {confidence:.4f})")

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print("\nText 2 (non-spam example):", text_2)
prediction, confidence = classify_review(
    text_2, model, tokenizer, device,
    max_length=train_dataset.max_length
)
print(f"Classification: {prediction} (confidence: {confidence:.4f})")

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1) 
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# test the model again:

# Test the model with example texts
text_1 = (
  "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune."
  "To disable this service, reply with the word 'STOP' in the message. For more information on callertunes, visit http://www.vodafone.in/callertune"
)

print("\nTesting spam detection:")
print("Text 1 (spam example):", text_1)
prediction, confidence = classify_review(
    text_1, model, tokenizer, device,
    max_length=train_dataset.max_length
)
print(f"Classification: {prediction} (confidence: {confidence:.4f})")

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print("\nText 2 (non-spam example):", text_2)
prediction, confidence = classify_review(
    text_2, model, tokenizer, device,
    max_length=train_dataset.max_length
)
print(f"Classification: {prediction} (confidence: {confidence:.4f})")

# At the end of the script, log the final test results
logger.info("\nFinal Test Results:")
logger.info(f"Test Loss: {test_loss:.4f}")
logger.info(f"Test Accuracy: {test_accuracy:.4f}")
logger.info(f"Training completed in {execution_time_minutes:.2f} minutes")

# Log the example classifications
logger.info("\nExample Classifications:")
logger.info(f"Spam example: {text_1}")
prediction1, confidence1 = classify_review(
    text_1, model, tokenizer, device,
    max_length=train_dataset.max_length
)
logger.info(f"Classification: {prediction1} (confidence: {confidence1:.4f})")

logger.info(f"\nNon-spam example: {text_2}")
prediction2, confidence2 = classify_review(
    text_2, model, tokenizer, device,
    max_length=train_dataset.max_length
)
logger.info(f"Classification: {prediction2} (confidence: {confidence2:.4f})")

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# If there is a sharp downward slope in the plot for both curves, the model is learning well from the training data, 
# and there is little to no indication of overfitting; that is, there is no noticeable gap between the 
# training and validation set losses.

# save the model
torch.save(model.state_dict(), "review_classifier.pth")

# plot the classification accuracies
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(
    epochs_tensor, examples_seen_tensor, train_accs, val_accs,
    label="accuracy"
)

train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

