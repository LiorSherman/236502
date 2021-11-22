import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, mapping_path, device='cpu', num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.mapping_path = mapping_path
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.drop = nn.Dropout()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # hidden state of the lstm
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # internal state of the lstm
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        output, _ = self.lstm(x, (h0, c0))
        output = self.drop(output)
        output = self.fc(output[:, -1, :])
        return output
