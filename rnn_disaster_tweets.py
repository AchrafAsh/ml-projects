import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence  # padding of every batch
from nlp.rnn_dataset import Vocabulary
from rnn import RNNClassifier, classify


class DisasterDataset(Dataset):
    def __init__(self, root_dir, filename, freq_threshold=1):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, filename))
        self.df = self.df.loc[0:100]

        self.targets = self.df["target"]
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

        return {"text": torch.tensor(encoded_text).float(), "target": torch.tensor(self.targets[idx])}


class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        targets = [item["target"] for item in batch]
        texts = [item["text"] for item in batch]
        texts = pad_sequence(texts, batch_first=False,
                             padding_value=self.pad_idx)

        return texts, torch.tensor(targets)


def get_loader(root_dir, filename, batch_size=10, num_workers=1, shuffle=True, pin_memory=True):
    dataset = DisasterDataset(root_dir, filename)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory, collate_fn=CollateBatch(pad_idx=pad_idx))
    return loader, dataset


if __name__ == "__main__":
    BATCH_SIZE = 1
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 2
    LR = 0.05
    NUM_EPOCHS = 100

    dataloader, dataset = get_loader(root_dir="../kaggle/disaster-tweets/data",
                                     filename="train.csv",
                                     batch_size=BATCH_SIZE)

    classifier = RNNClassifier(input_size=dataset.vocab_size, hidden_size=HIDDEN_SIZE,
                               output_size=OUTPUT_SIZE)

    classifier.fit(dataloader, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    print(classify(classifier, vocab=dataset.vocab,
                   sentence="Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"))
    print(classify(classifier, vocab=dataset.vocab,
                   sentence="Summer is lovely"))
