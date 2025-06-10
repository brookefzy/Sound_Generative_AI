# models/autoregressive.py
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self, input_length, input_dim=1, hidden_dim=128, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):  # x: [B, 1, T]
        x_seq = x.transpose(1,2)  # [B, T, 1]
        out, _ = self.rnn(x_seq)
        return torch.tanh(self.fc(out)).transpose(1,2)