import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim,
                 layer_num):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        # input(batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=layer_num,
                            batch_first=True)  # [1, hidden] --> [batch_size, 1, hidden]
        self.fc = nn.Linear(hidden_dim, output_dim)



    def forward(self, x):

        # input(layer_num, batch_size, hidden)
        h0 = torch.randn(self.layer_num, x.shape[0], self.hidden_dim)
        c0 = torch.randn(self.layer_num, x.shape[0], self.hidden_dim)

        # print(x.shape)

        x = x.unsqueeze(2)  # [batch_size, seq_len, 1]

        # print(x.shape)
        out, _ = self.lstm(x, (h0, c0))  # output(batch_size, seq_len, hidden_size)
        out = self.fc(out)  # output(batch_size, seq_len, 1)
        out = out[:, -1, :]
        return out

class Model(nn.Module):
    def __init__(self, hidden_dim, output_dim,
                 layer_num):
        super(Model, self).__init__()

        self.lstm = LSTM(hidden_dim, output_dim, layer_num)

    def forward(self, x):
        output = self.lstm(x)

        return output