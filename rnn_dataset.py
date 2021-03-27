import os
import pandas as pd
import numpy as np
import torch
import spacy # for tokenization

from torch.nn.utils.rnn import pad_sequence # padding of every batch
from torch.utils.data import Dataset, DataLoader

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenizer_en(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentences):
        frequencies = {} # store the frequency of each word encountered
        idx = 4 # 0, 1, 2 and 3 are already set
        
        for sentence in sentences:
            for word in self.tokenizer_en(sentence):
                if word not in frequencies: frequencies[word] = 1
                else: frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        
    def stoi(self, word): 
        if word in self.stoi: return self.stoi[word]
        else: return self.stoi["<UNK>"]
    
    def __getitem__(self, idx): 
        if idx in self.itos: return self.itos[idx]
        else: return -1
    
    def encode(self, sentence):
        tokens = self.tokenizer_en(sentence)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokens 
        ]


class SentimentDataset(Dataset):
    def __init__(self, root_dir, filename, freq_threshold=1):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir,filename))
        
        self.sentiments = self.df["sentiment"]
        self.texts = self.df["text"]
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.texts.tolist())
        
        self.vocab_size = len(self.vocab)
        
    def __len__(self): return len(self.df)
    
    def one_hot_tensor(self, idx):
        tensor = np.zeros(self.vocab_size)
        tensor[idx] = 1
        return tensor
    
    def __getitem__(self, idx):
        encoded_text = [self.one_hot_tensor(self.vocab.stoi["<SOS>"])]
        encoded_text += [self.one_hot_tensor(encoded_token) 
                         for encoded_token in self.vocab.encode(self.texts[idx])]
        encoded_text.append(self.one_hot_tensor(self.vocab.stoi["<EOS>"]))
        
        return { "text": torch.tensor(encoded_text).float(), "sentiment": torch.tensor(self.sentiments[idx]) }
    

class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        sentiments = [item["sentiment"] for item in batch]
        texts = [item["text"] for item in batch]
        texts = pad_sequence(texts, batch_first=False, padding_value=self.pad_idx)
        
        return texts, torch.tensor(sentiments)

    
def get_loader(root_dir, filename, batch_size=10, num_workers=1, shuffle=True, pin_memory=True):
    dataset = SentimentDataset(root_dir, filename)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, 
                        shuffle=shuffle, pin_memory=pin_memory, collate_fn=CollateBatch(pad_idx=pad_idx))
    return loader, dataset