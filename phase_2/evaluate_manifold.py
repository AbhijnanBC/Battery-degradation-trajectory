import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 2000

print("Loading Real Data...")
real_data = torch.load("data/processed/calce_test_windows.pt", weights_only=True).squeeze(-1).numpy()

def get_start_end(curves):
    return curves[:, 0], curves[:, -1]

real_start, real_end = get_start_end(real_data)

def generate_and_extract(model_path):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping.")
        return None, None
    
    gen = Generator(latent_dim=100, seq_length=50).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    gen.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, 100).to(device)
        fake_curves = gen(z).cpu().numpy()
        
    return get_start_end(fake_curves)

print("Loading Baseline Seed 10...")
base_start, base_end = generate_and_extract("saved_models_calce/baseline_seed_10/generator_final.pth")

print("Loading TV Seed 10...")
tv_start, tv_end = generate_and_extract("saved_models_calce/tv_seed_10/generator_final.pth")

print("Plotting Manifold Coverage Histograms...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Starting Capacities
axes[0].hist(real_start, bins=50, alpha=0.5, density=True, label='Real (CALCE)', color='black')
if base_start is not None:
    axes[0].hist(base_start, bins=50, alpha=0.5, density=True, label='Baseline', color='red')
if tv_start is not None:
    axes[0].hist(tv_start, bins=50, alpha=0.5, density=True, label='TV', color='blue')
axes[0].set_title('Distribution of Starting Capacities (Window index 0)')
axes[0].set_xlabel('Normalized Capacity')
axes[0].legend()

# Plot Ending Capacities
axes[1].hist(real_end, bins=50, alpha=0.5, density=True, label='Real (CALCE)', color='black')
if base_end is not None:
    axes[1].hist(base_end, bins=50, alpha=0.5, density=True, label='Baseline', color='red')
if tv_end is not None:
    axes[1].hist(tv_end, bins=50, alpha=0.5, density=True, label='TV', color='blue')
axes[1].set_title('Distribution of Ending Capacities (Window index 49)')
axes[1].set_xlabel('Normalized Capacity')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_manifold_coverage.png', dpi=300)
print("Saved plot to 'fig_manifold_coverage.png'")