import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse

from src.models import Generator, Discriminator
from src.physics_loss import calculate_physics_loss, count_monotonicity_violations

# --- Use Argparse to make running experiments easy ---
parser = argparse.ArgumentParser()
parser.add_argument('--lambda_weight', type=float, default=0.0, help='Weight of the physics loss')
args = parser.parse_args()

# 1. Hyperparameters
latent_dim = 100
seq_length = 50 
num_epochs = 3000
batch_size = 32
learning_rate = 0.0002
lambda_physics = args.lambda_weight # Controlled via command line

# 2. Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Starting Experiment with Lambda = {lambda_physics} on {device} ---")

# 3. Data Loading
data_path = "data/processed/battery_windows.pt"
training_tensor = torch.load(data_path, weights_only=True).to(torch.float32)
dataset = TensorDataset(training_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 4. Initialize Models
generator = Generator(latent_dim, seq_length).to(device)
discriminator = Discriminator(seq_length).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

os.makedirs(f"saved_models/lambda_{lambda_physics}", exist_ok=True)

# 5. The Training Loop
for epoch in range(num_epochs):
    epoch_d_loss = 0.0
    epoch_g_adv_loss = 0.0
    epoch_g_phys_loss = 0.0
    epoch_violations = 0
    
    for i, (real_batch,) in enumerate(dataloader):
        real_batch = real_batch.to(device)
        current_batch_size = real_batch.size(0)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        output_real = discriminator(real_batch)
        d_loss_real = criterion(output_real, real_labels)
        
        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_batch = generator(z)
        output_fake = discriminator(fake_batch.detach())
        d_loss_fake = criterion(output_fake, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        
        output_fake_for_G = discriminator(fake_batch)
        g_loss_adv = criterion(output_fake_for_G, real_labels)
        
        # --- SEPARATE AND MEASURE THE LOSSES ---
        g_loss_physics_raw = calculate_physics_loss(fake_batch)
        
        g_loss = g_loss_adv + (lambda_physics * g_loss_physics_raw)
        
        g_loss.backward()
        optimizer_G.step()
        
        # Tracking metrics
        epoch_d_loss += d_loss.item()
        epoch_g_adv_loss += g_loss_adv.item()
        epoch_g_phys_loss += g_loss_physics_raw.item() # Track the RAW physics loss, not multiplied
        epoch_violations += count_monotonicity_violations(fake_batch.detach())

    # 6. Logging
    if epoch % 100 == 0:
        avg_d = epoch_d_loss / len(dataloader)
        avg_g_adv = epoch_g_adv_loss / len(dataloader)
        avg_g_phys = epoch_g_phys_loss / len(dataloader)
        
        print(f"Ep [{epoch}/{num_epochs}] | D: {avg_d:.4f} | G_Adv: {avg_g_adv:.4f} | G_Phys(Raw): {avg_g_phys:.6f} | Violations: {epoch_violations}")

# 7. Save Final Models
torch.save(generator.state_dict(), f"saved_models/lambda_{lambda_physics}/generator_final.pth")
print(f"Experiment Lambda={lambda_physics} complete.")