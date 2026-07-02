import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, seq_length=50):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * (seq_length // 4)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Upsample + Conv1d prevents jagged artifacts
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.seq_length // 4)
        out = self.conv_blocks(out)
        out = nn.functional.interpolate(out, size=self.seq_length)
        return out.squeeze(1) 

class Discriminator(nn.Module):
    def __init__(self, seq_length=50):
        super(Discriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Flatten and output probability
        self.fc = nn.Sequential(
            nn.Linear(128 * (seq_length // 4), 1),
            
        )

    def forward(self, sequence):
        # Conv1D expects shape [batch_size, channels, length]
        out = sequence.unsqueeze(1)
        out = self.conv_blocks(out)
        out = out.view(out.size(0), -1) # Flatten
        return self.fc(out)