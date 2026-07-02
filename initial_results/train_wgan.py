import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import numpy as np

from src.models import Generator, Discriminator
from src.physics_loss import calculate_tv_loss, calculate_monotonicity_loss, calculate_all_physics_loss, count_monotonicity_violations

parser = argparse.ArgumentParser()
parser.add_argument('--lambda_physics', type=float, default=50.0)
parser.add_argument('--loss_type', type=str, required=True, choices=['none', 'tv', 'monotonicity', 'all'])
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--seed', type=int, default=42, help='Random seed for statistical significance')
args = parser.parse_args()

# Set the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

latent_dim = 100
seq_length = 50 
num_epochs = 3000
batch_size = 32
learning_rate = 0.0001
n_critic = 5
lambda_gp = 10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Starting {args.exp_name} (Loss: {args.loss_type}, Lambda: {args.lambda_physics}) ---")

# LOAD ONLY TRAINING DATA
training_tensor = torch.load("data/processed/train_windows.pt", weights_only=True).to(torch.float32)
dataset = TensorDataset(training_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

generator = Generator(latent_dim, seq_length).to(device)
critic = Discriminator(seq_length).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))

save_dir = f"saved_models/{args.exp_name}"
os.makedirs(save_dir, exist_ok=True)

def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

for epoch in range(num_epochs):
    epoch_c_loss, epoch_g_loss, epoch_phys_loss, epoch_violations = 0.0, 0.0, 0.0, 0
    
    for i, (real_batch,) in enumerate(dataloader):
        real_batch = real_batch.to(device)
        current_batch_size = real_batch.size(0)

        # Train Critic
        optimizer_C.zero_grad()
        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_batch = generator(z)
        
        real_validity = critic(real_batch)
        fake_validity = critic(fake_batch.detach())
        gp = compute_gradient_penalty(critic, real_batch.data, fake_batch.data)
        
        c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
        c_loss.backward()
        optimizer_C.step()
        
        # Train Generator
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z = torch.randn(current_batch_size, latent_dim).to(device)
            fake_batch = generator(z)
            g_adv_loss = -torch.mean(critic(fake_batch))
            
            # --- THE ROUTER: Selects the correct loss based on arguments ---
            if args.loss_type == 'tv':
                g_phys_loss_raw = calculate_tv_loss(fake_batch)
            elif args.loss_type == 'monotonicity':
                g_phys_loss_raw = calculate_monotonicity_loss(fake_batch)
            elif args.loss_type == 'all':
                g_phys_loss_raw = calculate_all_physics_loss(fake_batch)
            else:
                g_phys_loss_raw = torch.tensor(0.0).to(device)
            
            g_loss = g_adv_loss + (args.lambda_physics * g_phys_loss_raw)
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss += g_adv_loss.item()
            epoch_phys_loss += g_phys_loss_raw.item()
            
        epoch_c_loss += c_loss.item()
        epoch_violations += count_monotonicity_violations(fake_batch.detach())

    if epoch % 500 == 0:
        print(f"Ep [{epoch}/{num_epochs}] | Critic: {epoch_c_loss/len(dataloader):.4f} | G_Adv: {epoch_g_loss/(len(dataloader)/n_critic):.4f} | Violations: {epoch_violations}")

torch.save(generator.state_dict(), f"{save_dir}/generator_final.pth")
print(f"Finished {args.exp_name}.")