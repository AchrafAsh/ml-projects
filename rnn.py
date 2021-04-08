import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.whh = nn.Linear(hidden_size, hidden_size)
        self.wxh = nn.Linear(input_size, hidden_size)

    def forward(self, x, hidden_state):  # x (batch_size, input_size)
        return torch.sigmoid(self.whh(hidden_state) + self.wxh(x))


class RNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=2):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = RNNCell(input_size, hidden_size)
        self.rnn2 = RNNCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
#         self.whh = nn.Linear(hidden_size, hidden_size)
#         self.wxh = nn.Linear(input_size, hidden_size)

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, x, hidden_state):  # x (seq_len, batch_size, input_size)
        h1 = hidden_state
        h2 = hidden_state

        for i in range(x.shape[0]):
            h1 = self.rnn1(x[i], h1)
            h2 = self.rnn2(h1, h2)

        output = F.softmax(self.fc(h2), dim=1)
        return output, hidden_state

    def fit(self, dataset, batch_size, epochs, lr=0.001):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()  # ignore_index=pad_idx?

        for epoch in range(epochs):
            total_loss = 0
            for _, (texts, sentiments) in enumerate(tqdm(dataset)):
                hidden_state = self.init_hidden_state(batch_size=batch_size)

                # forward
                for i in range(texts.shape[0]):
                    output, hidden_state = self.forward(texts[i], hidden_state)

                loss = criterion(output, sentiments)

                # backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

                # gradient descent or Adam step
                optimizer.step()

                total_loss += loss

            if (epoch % 10 == 0):
                print(
                    f"epoch [{epoch+1} / {epochs}] | total loss: {total_loss}")


def classify(classifier, vocab, sentence):
    vocab_size = len(vocab)

    def one_hot_tensor(idx):
        tensor = [0] * vocab_size
        tensor[idx] = 1
        return tensor

    encoded_text = []
    encoded_text.append([one_hot_tensor(vocab.stoi["<SOS>"])])
    encoded_text += [[one_hot_tensor(encoded_token)]
                     for encoded_token in vocab.encode(sentence)]
    encoded_text.append([one_hot_tensor(vocab.stoi["<EOS>"])])

    encoded_text = torch.tensor(encoded_text).float()
    print(encoded_text.shape)
    h0 = classifier.init_hidden_state(batch_size=1)

    for i in range(encoded_text.shape[0]):
        output, h0 = classifier(encoded_text[i], h0)
    return output
