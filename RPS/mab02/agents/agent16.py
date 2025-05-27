# Originally: transformer_agent (TransformerAgentAgent)

import random
import torch
import torch.nn as nn

class Agent16:
    def __init__(self):
        self.name = "Agent16"
        self.model = TransformerModel()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_move(self, history):
        if len(history) < 6:
            return random.choice(['r', 'p', 's'])

        mapping = {'r': 0, 'p': 1, 's': 2}
        reverse = {0: 'r', 1: 'p', 2: 's'}

        X, y = [], []
        for i in range(4, len(history)):
            seq = [mapping[history[j]['player']] for j in range(i-4, i)]
            target = mapping[history[i]['player']]
            X.append(seq)
            y.append(target)

        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        self.model.train()
        for _ in range(5):
            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()

        last_seq = [mapping[entry['player']] for entry in history[-4:]]
        input_seq = torch.tensor([last_seq], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_seq).argmax(dim=1).item()

        counter = {'r': 'p', 'p': 's', 's': 'r'}
        return counter[reverse[pred]]


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(3, 16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])
