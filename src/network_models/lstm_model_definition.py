import torch
import torch.nn as nn

# Model parameters
# hidden_dim = 128
# num_layers = 3
#output_dim = 1
# hidden layer is [3 x 128]

class LSTMModel(nn.Module):
    def __init__(
            self, 
            input_dim: int = 1, 
            hidden_dim: int = 16, 
            num_layers: int = 1, 
            output_dim: int = 1
        ):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.do = nn.Dropout()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) # Initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) # Initial cell state
        out, _ = self.lstm(x, (h0, c0))
        # out = self.do(out)
        out = self.fc(out[:, -1, :])   # Take the last point
        return out