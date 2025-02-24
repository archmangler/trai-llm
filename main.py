import urllib.request
import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import sys

print(sys.executable)
print(sys.path)

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

#To understand the meaning of stride=1, let’s fetch another batch from this dataset: 
second_batch = next(data_iter)
print(second_batch)

# The second batch has the following contents: [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
# If we compare the first and second batches, we can see that the second batch’s token IDs are shifted by one position 
# (for example, the second ID in the first batch’s input is 367, which is the first ID of the second batch’s input). 
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
print(embedding_layer.weight) # prints the embedding layer’s underlying weight matrix

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





