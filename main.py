import urllib.request
import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
# from chapter03 import MultiHeadAttention

print(sys.executable)
print(sys.path)



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
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device) # The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# The transformer block component of GPT
# This code defines a TransformerBlock class in PyTorch that includes a multi-head attention mechanism (MultiHeadAttention) and a feed forward network (Feed- Forward), both configured based on a provided configuration dictionary (cfg), such as GPT_CONFIG_124M.
# Layer normalization (LayerNorm) is applied before each of these two components, and dropout is applied after them to regularize the model and prevent overfitting. This is also known as Pre-LayerNorm.
# Older architectures, such as the original transformer model, applied layer normalization after the self-attention and feed forward networks instead, known as Post-LayerNorm, which often leads to worse training dynamics.
class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut                # add the original input back
        shortcut = x                    # shortcut connection for feedforward bloc
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut                # add the original input back
        return x



# A neural network to illustrate shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                          GELU())
                          ])
    def forward(self, x):
        for layer in self.layers: #Check if shortcut can be applied
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# A feed forward neural network module
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
    
# An implementation of the GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
            ))
            
# A layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# A placeholder GPT model architecture class
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
              )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# A simple placeholder class that will be replaced by a real TransformerBlock later
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x

# A simple placeholder class that will be replaced by a real LayerNorm later
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5): # The parameters here are just to mimic the LayerNorm interface.
        super().__init__()
    def forward(self, x): # The forward method describes the data flow through the model: it computes token and positional embeddings for the input indices, applies dropout, processes the data through the transformer blocks, applies normalization, and finally produces logits with the linear output layer.
        return x

# An efficient multi-head attention class:
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),diagonal=1)
            )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # reshaping (.view) and transposing (.transpose) of tensors inside the MultiHeadAttention class
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim 
        )
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
            )
        context_vec = self.out_proj(context_vec)
        return context_vec
    
# A wrapper class to implement multi-head attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
             d_in, d_out, context_length, dropout, qkv_bias
             )
             for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# A compact causal attention class
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # Compared to the previous SelfAttention_v1 class, we added a dropout layer.
        self.register_buffer( 'mask', #The register_buffer call is also a new addition (more information is provided in the following text).
                             torch.triu(torch.ones(context_length, context_length),
                                        diagonal=1)
                                        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_( #We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

# A compact, self-attention Python class
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec    
         
# Naive softmax for normalisation
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

#before the introduction of unknown token handling:
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
            ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

#After introduction of unknown token handling:
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# A data loader to generate batches with input-target pairs
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
     tokenizer = tiktoken.get_encoding("gpt2") #            Initializes the tokenizer
     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # Creates dataset
     dataloader = DataLoader(
         dataset,
         batch_size=batch_size,
         shuffle=shuffle,
         drop_last=drop_last, #drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
         num_workers=num_workers #The number of CPU processes to use for preprocessing
         )
     return dataloader 

#Get/"Ingest" raw data
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

#"Load" the data
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

#Next up: Tokenize (to words and characters), so we can turn it into word embeddings.
#First, a SAMPLE test: Can the text be split using Python regular expressions? If so, we
#should be ok to do it with an LLM!
text = raw_text[:99]
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

#Should be:
#Input: I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
#['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no']

#NOTES: 
# 1. Don't blindly lower case input! LLMs read capitalisation as context!
# 2. Whitespace can sometimes be important e.g in text that uses whitespace to convey meaning. We;ve removed it here.

#Second: A FULL text test:
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

#Coingratulations, you now have TOKENS!
print(preprocessed[:30])

#Third: Assign these tokens an ID using some scheme. This is called a "Vocabulary" for some bizarre reason.
#Which means build a mapping from token to ID, which means building some form of dictionary/lookup table
#which allows one to map from token to ID and vice versa.
#NOTE: We need this to eventually create embedding vectors. 
#This means deduplication must be performed at this stage.

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("Vocabulary size after deduplication: ", vocab_size)

#Print out for the human for illustrative purposes:
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

#instantiate a tokenizer object: First a small test
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Token Ids resulting from sample tokenization: \n",ids)

#Can decode?
print("Decoding token ids: \n", tokenizer.decode(ids))

#Will it recognise tokens not in the vocabulary?
#text = "This is more like Science Fiction."
#print("Test: recognisign words not in the vocabulary: \n", tokenizer.encode(text))
#This should fail with: KeyError: 'Science'

#Modify the vocabulary to include special tokens to handle unknown tokens and separate unrelated text sources:
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print("Extended vocabulary size: \n", len(vocab.items()))

#A basic test for the end-of-text token;
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print("debug: ",text)

# Now test SimpleTokenizerV2
tokenizer = SimpleTokenizerV2(vocab)
print("Tokenize: ", tokenizer.encode(text))

#Detokenise to check:
print("Detokenize: ", tokenizer.decode(tokenizer.encode(text)))

#Byte-PAir-Encoding. Yes, it's a thing and aparently we have to do  it.
#Also, it's really complex, so the python library is just a wrapper around RUST.
#tiktoken's the name and BPE's the game: pip install tiktoken
#Weird, but true. Anyways ...
print("Using tiktoken version:", version("tiktoken"))

#instantiate a BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

#Use the new tokenizer:
text = (
            "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
             "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Tokenize with BPE (Byte Pair Encoding): ",integers)

#Convert token IDs back into text:
strings = tokenizer.decode(integers)
print("De-tokenize with BPE: ",strings)

#Test: Test BPE on an unknown word:
#"Cykuz lhu"
text = (
            "Hello, do you like Cykuz lhu? <|endoftext|> In the sunlit terraces of Cykuz lhu"
             "of someunknownPlace in Cykuz lhu."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Tokenize with BPE (Byte Pair Encoding): ",integers)
for item in integers:
    print(" -> ",item)

for item in integers:
    decoded_item = tokenizer.decode([item])
    print(item," -> ",decoded_item)

#Convert token IDs back into text:
strings = tokenizer.decode(integers)
print("De-tokenize with BPE: ",strings)

#Next: Implement a data loader that fetches the input–target pairs in figure 2.12 from the training dataset using a sliding window approach.
#But why???
#Firstly, encode using BPE
input_data_file="the-verdict.txt"
with open(input_data_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print("BPE encoded corpus text: ",enc_text)
print("BPE encoded length of corpus: ",len(enc_text))

#take arbitrarily , say 45, tokens from the encoded text
enc_sample = enc_text[45:]

#Create input-target pairs for the word prediction test:
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

# create the next-word prediction tasks
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

#Check if gpu available.
print("Can access NVIDIA GPU? -> ", torch.cuda.is_available())
print("Can access the Apple Silicon GPU? ", torch.backends.mps.is_available())

# Testing the data loader task
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

#To understand the meaning of stride=1, let's fetch another batch from this dataset: 
second_batch = next(data_iter)
print(second_batch)

# The second batch has the following contents: [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
# If we compare the first and second batches, we can see that the second batch's token IDs are shifted by one position 
# (for example, the second ID in the first batch's input is 367, which is the first ID of the second batch's input). 
# The stride setting dictates the number of positions the inputs shift across batches, emulating a sliding window approach

# small batch sizes require less memory during training but lead to more noisy model updates. Just like in regular deep learning, the batch size is a tradeoff and a hyperparameter to experiment with when training LLMs.

# use the data loader to sample with a batch size greater than 1:
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# last step in preparing the input text for LLM training is to convert the token IDs into embedding vectors,
# 1. Suppose we have the following four input tokens with IDs 2, 3, 5, and 1:
input_ids = torch.tensor([2, 3, 5, 1])

# 3. suppose we have a small vocabulary of only 6 words (instead of the 50,257 words in the BPE tokenizer vocabulary), and we want to create embed- dings of size 3 (in GPT-3, the embedding size is 12,288 dimensions):
vocab_size = 6
output_dim = 3

# 4. instantiate an embedding layer in PyTorch, setting the random seed to 123 for reproducibility purposes:
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight) # prints the embedding layer's underlying weight matrix

# 5. apply it to a token ID to obtain the embedding vector:
print(embedding_layer(torch.tensor([3])))

# 6. now apply that to all four input IDs (torch.tensor([2, 3, 5, 1])):
print("The embedding matrix: ",embedding_layer(input_ids))


#Next: A bigger sample:
# we assume that the token IDs were created by the BPE tokenizer we implemented earlier, which has a vocabulary size of 50,257:


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)


# now use the embedding layer to embed these token IDs into 256-dimensional vectors:
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# GPT model's approach:  create another embedding layer that has the same embedding dimension as the token_embedding_ layer:
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) 
print(pos_embeddings.shape)

# add these directly to the token embeddings, where PyTorch will add the 4 × 256–dimensional pos_embeddings tensor to each 4 × 256–dimensional token embedding tensor in each of the eight batches:
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

# Checkpoint: The data is now in a position to be processed by an LLM

# Next: The Attention Mechanism.
#1. Why attention mechanisms in neural networks
#2. Basic self-attention framework, progressing to an enhanced self-attention mechanism
#3. A causal attention module that allows LLMs to generate one token at a time
#4. Masking randomly selected attention weights with dropout to reduce overfitting
#5. Stacking multiple causal attention modules into a multi-head attention module

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

# Calculate the intermediate attention scores between the query token and each input token. 
# Determine these scores by computing the dot product of the query, x(2), with every other input token:
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("Preliminary attention scores: ", attn_scores_2)

# normalize each of the attention scores we computed previously. The main goal behind the normalization is to obtain attention weights that sum up to 1.
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Use the softmax function for normal- ization. This approach is better at managing extreme values
# softmax function ensures that the attention weights are always posi- tive. 
# This makes the output interpretable as probabilities or relative importance, where higher weights indicate greater importance.
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# Better to use the pytorch implementation of Softmax:
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# We have now computed the normalized attention weights
# It's all about the Context Vector!
# The final step, after calculating and normalizing the attention scores to obtain the attention weights for a query, is to compute the context vector. 
# This context vector is a combination of all input vectors x(1) to x(T) weighted by the the attention weights.
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

# Computing attention weights for all input tokens
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

#The normalised attention score will be printed.
print("With for-loops", attn_scores)

# tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])

# Matrix multiplication is faster than for loops:
attn_scores = inputs @ inputs.T
print("With Matrix Multiplication: ",attn_scores)

#Recap:
# 1. Compute the attention scores as dot products between the inputs.
# 2. The attention weights are a normalized version of the attention scores.
# 3. The context vectors are computed as a weighted sum over the inputs.

# normalize each row so that the values in each row sum to 1:
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Per-row normalization: ",attn_weights)

# verify that the rows all sum to 1:
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# compute all context vectors via matrix multiplication:
all_context_vecs = attn_weights @ inputs
print("Computing all context vectors: ", all_context_vecs)
#compare:
print("Compare with previous 2nd context vector:", context_vec_2)

# checkpoint: This concludes the code walkthrough of a simple self-attention mechanism. 

# Next: add trainable weights, enabling the LLM to learn from data. Implementing self-attention with trainable weights.
# "scaled dot-product attention."
# Objective: we want to compute context vectors as weighted sums over the input vectors specific to a certain input element

# Example;
# The second input element
x_2 = inputs[1]
d_in = inputs.shape[1] # The input embedding size, d=3
d_out = 2 # The output embedding size, d_out=2

# Initialize the three weight matrices Wq, Wk, and Wv shown in figure 3.14: torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# compute the query, key, and value vectors:
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("Resulting query vector: ", query_2)

# Note: In the weight matrices W, the term "weight" is short for "weight parameters," the values of a neural network that are optimized during training. 
# This is not to be confused with the attention weights. As we already saw, attention weights determine the extent to which a context vector depends on 
# the different parts of the input (i.e., to what extent the network focuses on different parts of the input). In summary, weight parameters are the fundamental, 
# learned coefficients that define the network's connections, while attention weights are dynamic, context-specific values.

# we still require the key and value vectors for all input elements as they are involved in com- puting the attention weights with respect to the query 
# obtain all keys and values via matrix multiplication:

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# At this point we have successfully projected the six input tokens from a three-dimensional onto a two-dimensional embedding space.

# Next: compute the attention scores

# Test: compute one of the attention scores:
# compute the attention score ω22:
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("Attention score 22: ", attn_score_22)

# Calcualte all the attention scores for a given query:
attn_scores_2 = query_2 @ keys.T
print("All attention scores for the latest query: ", attn_scores_2)

# Next: go from the attention scores to the attention weights.
# We compute the attention weights by scaling the attention scores and using the softmax function.

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("The attention weights: ", attn_weights_2)

# The last step is multiplying each value vector with its respective attention weight and then summing them to obtain the context vector
# we can use matrix multiplication to obtain the output in one step:
context_vec_2 = attn_weights_2 @ values
print("Resulting context vector: ", context_vec_2)

# Checkpoint: we've only computed a single context vector. 
# Next, we will generalize the code to compute all context vectors in the input sequence

# Using a compact self-attention Python class
torch.manual_seed(123)
sa_v1 = SelfAttention_v2(d_in, d_out)
print("Context Vectors V2: ", sa_v1(inputs))

# Our next step is to implement the causal attention mask in code. 
# To implement the steps to apply a causal attention mask to obtain the masked attention weights,
# 1. Attention scores (unnormalized)
# 2. Attention weights (normalized)
# 3. Masked attention scores (unnormalized)
# 4. Masked Attention weights Masked attention scores (normalized)
# One way to obtain the masked attention weight matrix in causal attention is to apply the softmax function to the attention scores, zeroing out the elements above the diagonal and normalizing the resulting matrix.

# use the SelfAttention_v2 similar to SelfAttention_v1:
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# Compute the attention weights using the softmax function as we have done previously:
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("Attention weights using softmax function: ", attn_weights)

# Next step: Use PyTorch's tril function to create a mask where the values above the diagonal in the tensor are zero:
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("Mask for causal attention masking: ", mask_simple)

# multiply this mask with the attention weights to zero-out the values above the diagonal:
masked_simple = attn_weights*mask_simple
print("Zeroing values above the diagonal: ", masked_simple)

# third step is to renormalize the attention weights to sum up to 1 again in each row. We can achieve this by dividing each element in each row by the sum in each row (wtf for?)
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("Re-normalised attention weights: ", masked_simple_norm)

# More efficient masking with Matrix and softmax magic:

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("Alternative Masking Method: ", masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print("Softmaxed, Masked results: ",attn_weights)

# Checkpoint: Hiding future words with causal attention
#Exploring Dropout to prevent overfitting
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) #choose a 50% dropout rate
example = torch.ones(6, 6) #Create a matrix of ones
print("Droput applied to a Matrix of ones: ", dropout(example))


#  apply dropout to the attention weight matrix itself:
torch.manual_seed(123)
print("Dropout applied to the attention weight matrix: ", dropout(attn_weights))

# Implementing a compact causal attention class
batch = torch.stack((inputs, inputs), dim=0)
print("Shape of a given batch: ", batch.shape)

# Use the CausalAttention class
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)

# Use the multi-head attention class
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print("Multihead COntext Vector", context_vecs) # A tensor representing context vectors
print("context_vecs.shape:", context_vecs.shape)

# Note: However, these are processed sequentially via [head(x) for head in self.heads] in the forward method
# We can do this in parrallel with a better class implementation.
# One way to achieve this is by com- puting the outputs for all attention heads simultaneously via matrix multiplication.

# Next: Implementing multi-head attention with weight splits

# MultiHeadAttention class can be used similar to the SelfAttention and CausalAttention classes we implemented earlier
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# Checkpoint: We have now implemented the MultiHeadAttention class that we will use when we implement and train the LLM.

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

# Test: Tokenize a batch of text:
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# initialize a new 124-million-parameter DummyGPTModel instance and feed it the tokenized batch:
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print("Logits output: ", logits)    

# The embedding has 50,257 dimensions because each of these dimensions refers to a unique token in the vocabulary. 
# When we implement the postprocessing code, we will convert these 50,257-dimensional vectors back into token IDs, which we can then decode into words.


# Next: Implement a neural network layer with five inputs and six outputs that we apply to two input examples:
# The neural network layer we have coded consists of a Linear layer followed by a non- linear activation function, 
# ReLU (short for rectified linear unit), which is a standard activation function in neural networks. 
# If you are unfamiliar with ReLU, it simply thresholds negative inputs to 0, ensuring that a layer outputs only positive values, 
# which explains why the resulting layer output does not contain any negative values.

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print("Neural Network Layer: ", out)

# Before we apply layer normalization to these outputs, let's examine the mean and variance of each row:
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)


# apply layer normalization to the layer outputs we obtained earlier. The operation consists of subtracting the mean and dividing by the square root of the vari- ance (also known as the standard deviation):

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

# turn off the scientific notation when printing tensor values by setting sci_mode to False:
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# try the LayerNorm module in practice and apply it to the batch input:
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# Note: The results *should* show that the layer normalization code works as expected and normalizes the values of each of the two inputs such that they have a mean of 0 and a variance of 1

# Next: Implement and test the GELU activation func- tion, which is one of the activation functions used in LLMs, instead of the traditional ReLU function we used previously.

# first: check we have th GeLU activation function
# The smoothness of GELU can lead to better optimization properties during training, as it allows for more nuanced adjustments to the model's parameters. 
# ReLU has a sharp corner at zero (figure 4.18, right), which can sometimes make opti- mization harder, especially in networks that are very deep or have complex architec- tures

gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
#plt.show()

# initialize a new FeedForward module with a token embedding size of 768 and feed it a batch input with two samples and three tokens each:
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print("Output of feedforward NN: ", out.shape)

# The FeedForward module plays a crucial role in enhancing the model's ability to learn from and generalize the data. 
# Although the input and output dimensions of this module are the same, it internally expands the embedding dimension into a higher-dimensional space through the first linear layer, 
# This expansion is followed by a nonlinear GELU activation and then a contraction back to the original dimension with the second linear transformation. 
# Such a design allows for the exploration of a richer representation space. (But wtf for???)

# The Vanishing Gradient problem: The vanishing gradient problem refers to the issue where gradients (which guide weight updates during training) become progressively smaller as they 
# propagate backward through the layers, making it difficult to effectively train earlier layers.
# Shortcut Connections: Originally, shortcut connections were proposed for deep networks in computer vision (specifically, in residual networks) to mitigate the challenge of van- ishing gradients. 
# A shortcut connection creates an alternative, shorter path for the gradient to flow through the network by skipping one or more layers, which is achieved by adding the output of one layer to the output of a later layer. 
# This is why these connections are also known as skip connections. They play a crucial role in pre- serving the flow of gradients during the backward pass in training.
    
#  A neural network to illustrate shortcut connections
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

# Implement a function that computes the gradients in the model's back- ward pass:
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# use the print_gradients function and apply it to the model without skip connections:
# output of the print_gradients function shows, the gradients become smaller as we progress from the last layer (layers.4) to the first layer (layers.0), which is a phenomenon called the vanishing gradient problem.
print_gradients(model_without_shortcut, sample_input)

#  now instantiate a model with skip connections and see how it compares:
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)

print("\nPrinting with model now using shortcuts\n")

print_gradients(model_with_shortcut, sample_input)

# The last layer (layers.4) still has a larger gradient than the other layers. 
# However, the gradient value stabilizes as we progress toward the first layer (layers.0) 
# and doesn't shrink to a vanishingly small value.

# Next: Create a transformer module and connect all the preceding modules to complete the GPT architecture.

# Theory: The self-attention mechanism in the multi-head attention block identifies and analyzes relationships between elements in the input sequence. 
# The feed forward network modifies the data individually at each position of the input sequence. This combination enables an "understanding" and capacity to process input for complex data patterns ("draw complex relationships").
# 

# Instantiate a transformer block and input sample data:
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("[Transformer Bloc] Input shape:", x.shape)
print("[Transformer Bloc] Output shape:", output.shape) 
# Note: the transformer architecture should process sequences of data without altering their shape throughout the network. So the input dimension should be same as output dimension.
# Each output vector must directly correspond to an input vector, maintaining a one-to-one relationship
# The output is a context vector that encapsulates information from the entire input sequence (see chapter 3). 
# This means that while the physical dimensions of the sequence (length and feature size) remain unchanged as it passes through the trans- former block, the content of each output 
# vector is re-encoded to integrate contextual information from across the entire input sequence.

# Recap: The transformer block combines layer normalization, the feed forward network, GELU activa- tions, and shortcut connections. It makes up the main component of the GPT architecture.

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("[RealGPTModel] Input batch:\n", batch)
print("\n[RealGPTModel] Output shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# compute the memory requirements of the 163 million parameters (compute the memory requirements of the 163 million parameters)
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

# Next:  write the code to convert these output tensors into text. 
# A function for the GPT model to generate text:

def generate_text_simple(model, idx, max_new_tokens, context_size): #idx is a (batch, n_tokens) array of indices in the current context.
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond) #Focuses only on the last time step, so that(batch,n_token,vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


# Test text generation output: encode the input context into token IDs:
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval() #Disables dropout since we are not training the model
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
    )
print("Output:", out)
print("Output length:", len(out[0]))

#At this point we'll get the untrained output prediction
#which will not be very good. 
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("Decoded text with prediction: ",decoded_text)

#We need to now train our model before we an get anything better.

# Recap: We have implemented the GPT architecture and initialized a GPT model instance with initial random weights.