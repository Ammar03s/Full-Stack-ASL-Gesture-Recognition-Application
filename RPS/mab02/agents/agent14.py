# Originally: lstm_sequence (LstmSequenceAgent)

import random
import torch
import torch.nn as nn
import numpy as np

class Agent14:
    def __init__(self):
        self.name = "Agent14"
        self.model = LSTMModel()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def get_move(self, history):
        if len(history) < 5:
            return random.choice(['r', 'p', 's'])

        mapping = {'r': 0, 'p': 1, 's': 2}
        reverse = {0: 'r', 1: 'p', 2: 's'}

        # Prepare training data
        sequences = []
        targets = []
        for i in range(3, len(history)):
            seq = [mapping[history[j]['player']] for j in range(i-3, i)]
            target = mapping[history[i]['player']]
            sequences.append(seq)
            targets.append(target)

        X = torch.tensor(sequences, dtype=torch.long)
        y = torch.tensor(targets, dtype=torch.long)

        self.model.train()
        for _ in range(5):  # mini-train
            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()

        # Predict
        last_seq = [mapping[entry['player']] for entry in history[-3:]]
        input_seq = torch.tensor([last_seq], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_seq).argmax(dim=1).item()

        counter = {'r': 'p', 'p': 's', 's': 'r'}
        return counter[reverse[pred]]


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(3, 8)
        self.lstm = nn.LSTM(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
