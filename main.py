import urllib.request
import re
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import sys

print(sys.executable)
print(sys.path)

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
print("Can access GPU? -> ", torch.cuda.is_available())

