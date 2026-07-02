import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):
    def __init__(self, latent_dim, seq_length=50):
        super(LSTMGenerator, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # Two stacked LSTM layers for temporal dynamics
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=64, num_layers=2, batch_first=True)
        # Output layer maps hidden state to a single capacity value
        self.linear = nn.Linear(64, 1)

    def forward(self, z):
        # z shape: [batch_size, latent_dim]
        # Repeat the latent vector for every time step in the sequence
        z_seq = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        lstm_out, _ = self.lstm(z_seq)
        out = self.linear(lstm_out)
        return out.squeeze(2)