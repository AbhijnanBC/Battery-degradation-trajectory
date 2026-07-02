import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 2000

# Function to compute linear MMD
def compute_mmd(X, Y):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    xx = torch.mm(X, X.t()).mean()
    yy = torch.mm(Y, Y.t()).mean()
    xy = torch.mm(X, Y.t()).mean()
    return (xx + yy - 2 * xy).item()

print("Loading Real Data...")
real_data = torch.load("data/processed/calce_test_windows.pt", weights_only=True).squeeze(-1).numpy()

def load_generator(model_path):
    if not os.path.exists(model_path): return None
    gen = Generator(latent_dim=100, seq_length=50).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    gen.eval()
    return gen

def generate_curves(gen, n=num_samples):
    if gen is None: return None
    with torch.no_grad():
        z = torch.randn(n, 100).to(device)
        return gen(z).cpu().numpy()

gen_base = load_generator("saved_models_calce/baseline_seed_10/generator_final.pth")
gen_tv = load_generator("saved_models_calce/tv_seed_10/generator_final.pth")
gen_phys = load_generator("saved_models_calce/all_priors_seed_10/generator_final.pth")

print("\n--- Generating Data ---")
base_data = generate_curves(gen_base)
tv_data = generate_curves(gen_tv)
phys_data = generate_curves(gen_phys)

print("\n--- Maximum Mean Discrepancy (MMD) to Real Data ---")
# To keep computation fast, we calculate MMD on a random subset
indices = np.random.choice(real_data.shape[0], min(1000, real_data.shape[0]), replace=False)
real_subset = real_data[indices]

if base_data is not None:
    mmd_base = compute_mmd(real_subset, base_data[:1000])
    print(f"Baseline MMD: {mmd_base:.6f}")

if tv_data is not None:
    mmd_tv = compute_mmd(real_subset, tv_data[:1000])
    print(f"TV MMD:       {mmd_tv:.6f}")

if phys_data is not None:
    mmd_phys = compute_mmd(real_subset, phys_data[:1000])
    print(f"Physics MMD:  {mmd_phys:.6f} <-- (Lower is better)")