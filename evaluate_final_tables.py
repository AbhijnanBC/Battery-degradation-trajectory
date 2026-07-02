import torch
import numpy as np
import os
import sys
from scipy.stats import wasserstein_distance

sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 1000

# --- Helper Functions ---
def compute_mmd(X, Y):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    xx = torch.mm(X, X.t()).mean()
    yy = torch.mm(Y, Y.t()).mean()
    xy = torch.mm(X, Y.t()).mean()
    return (xx + yy - 2 * xy).item()

def get_derivatives(data):
    return np.diff(data, axis=1).flatten()

def load_generator(model_path):
    if not os.path.exists(model_path): return None
    gen = Generator(latent_dim=100, seq_length=50).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    gen.eval()
    return gen

def generate_data(gen, n=num_samples):
    if gen is None: return None
    with torch.no_grad():
        z = torch.randn(n, 100).to(device)
        return gen(z).cpu().numpy()

# --- Load Real Data ---
print("Loading Real Data...")
nasa_real = torch.load("data/processed/test_windows.pt", weights_only=True).numpy()
calce_real = torch.load("data/processed/calce_test_windows.pt", weights_only=True).squeeze(-1).numpy()

nasa_deriv_real = get_derivatives(nasa_real)
calce_deriv_real = get_derivatives(calce_real)

# --- Define Model Paths ---
nasa_models = {
    "Baseline": "saved_models/exp_1_baseline/generator_final.pth",
    "TV": "saved_models/exp_2_tv/generator_final.pth",
    "Monotonicity": "saved_models/exp_3_monotonicity/generator_final.pth",
    "All Priors": "saved_models/exp_4_all/generator_final.pth"
}

calce_models = {
    "Baseline": "saved_models_calce/baseline_seed_10/generator_final.pth",
    "TV": "saved_models_calce/tv_seed_10/generator_final.pth",
    "Monotonicity": "saved_models_calce/monotonicity_seed_10/generator_final.pth",
    "All Priors": "saved_models_calce/all_priors_seed_10/generator_final.pth"
}

# --- Evaluate NASA MMD ---
print("\n" + "="*40)
print(" MISSING NASA MMD METRICS (TABLE 3)")
print("="*40)
nasa_subset = nasa_real[np.random.choice(nasa_real.shape[0], min(1000, nasa_real.shape[0]), replace=False)]
for name, path in nasa_models.items():
    gen = load_generator(path)
    fake_data = generate_data(gen, len(nasa_subset))
    if fake_data is not None:
        mmd = compute_mmd(nasa_subset, fake_data)
        print(f"{name:15} | MMD: {mmd:.6f}")

# --- Evaluate CALCE Derivative WD & MMD ---
print("\n" + "="*40)
print(" MISSING CALCE METRICS (TABLE 4)")
print("="*40)
calce_subset = calce_real[np.random.choice(calce_real.shape[0], min(1000, calce_real.shape[0]), replace=False)]
for name, path in calce_models.items():
    gen = load_generator(path)
    fake_data = generate_data(gen, 1000)
    if fake_data is not None:
        fake_deriv = get_derivatives(fake_data)
        wd = wasserstein_distance(calce_deriv_real, fake_deriv)
        mmd = compute_mmd(calce_subset, fake_data)
        print(f"{name:15} | Deriv WD: {wd:.6f} | MMD: {mmd:.6f}")
print("="*40)