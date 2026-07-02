import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from src.models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim, seq_length = 100, 50

# 1. Load Data and Models
test_data = torch.load("data/processed/test_windows.pt", weights_only=True).numpy()

g_baseline = Generator(latent_dim, seq_length).to(device)
g_baseline.load_state_dict(torch.load("saved_models/baseline_seed_10/generator_final.pth", weights_only=True))
g_baseline.eval()

g_tv = Generator(latent_dim, seq_length).to(device)
g_tv.load_state_dict(torch.load("saved_models/tv_seed_10/generator_final.pth", weights_only=True))
g_tv.eval()

g_phys = Generator(latent_dim, seq_length).to(device)
g_phys.load_state_dict(torch.load("saved_models/all_priors_seed_10/generator_final.pth", weights_only=True))
g_phys.eval()

# ---------------------------------------------------------
# FIGURE 1: The TV vs Physics Comparison
# ---------------------------------------------------------
real_sample = test_data[42] # Static index for reproducibility

with torch.no_grad():
    z_single = torch.randn(1, latent_dim).to(device)
    tv_curve = g_tv(z_single).cpu().numpy().flatten()
    phys_curve = g_phys(z_single).cpu().numpy().flatten()

fig1, axs1 = plt.subplots(1, 3, figsize=(15, 4))
axs1[0].plot(real_sample, color='black', marker='o', markersize=3)
axs1[0].set_title('Real Degradation Window')
axs1[1].plot(tv_curve, color='blue', marker='o', markersize=3)
axs1[1].set_title('Generic Smoothing (TV Loss)')
axs1[2].plot(phys_curve, color='green', marker='o', markersize=3)
axs1[2].set_title('Physics-Informed (All Priors)')

for ax in axs1:
    ax.grid(True)
plt.tight_layout()
plt.savefig('fig1_tv_comparison.png', dpi=300)

# ---------------------------------------------------------
# FIGURE 2: The 12-Curve Random Grid (No Cherry-Picking)
# ---------------------------------------------------------
# Select 4 random real curves
random.seed(99) # Set seed so the "random" selection is reproducible for the paper
random_indices = random.sample(range(len(test_data)), 4)
real_4 = [test_data[i] for i in random_indices]

# Generate 4 fake curves from Baseline and 4 from Physics
with torch.no_grad():
    z_batch = torch.randn(4, latent_dim).to(device)
    base_4 = g_baseline(z_batch).cpu().numpy()
    phys_4 = g_phys(z_batch).cpu().numpy()

fig2, axs2 = plt.subplots(3, 4, figsize=(16, 10))
fig2.suptitle('Qualitative Comparison: Real vs. Baseline GAN vs. Physics-Informed GAN', fontsize=16)

for i in range(4):
    # Row 1: Real
    axs2[0, i].plot(real_4[i], color='black', marker='o', markersize=2)
    axs2[0, i].set_title(f'Real Sample {i+1}')
    # Row 2: Baseline
    axs2[1, i].plot(base_4[i], color='red', marker='o', markersize=2)
    axs2[1, i].set_title(f'Baseline WGAN {i+1}')
    # Row 3: Physics
    axs2[2, i].plot(phys_4[i], color='green', marker='o', markersize=2)
    axs2[2, i].set_title(f'Physics-Informed {i+1}')

for ax in axs2.flat:
    ax.grid(True)
    ax.set_ylim([-1.1, 1.1]) # Uniform y-axis for fair comparison

plt.tight_layout()
plt.savefig('fig2_12_curve_grid.png', dpi=300)

print("Figures successfully generated and saved as PNGs.")