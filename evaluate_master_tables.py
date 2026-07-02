import torch
import numpy as np
import os
import sys
from scipy.stats import wasserstein_distance

# Attempt to load DTW, fallback to Euclidean if tslearn isn't installed
try:
    from tslearn.metrics import dtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("Warning: tslearn not found. DTW will be approximated using Euclidean distance for speed.")

sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = [10, 20, 30, 40, 50]
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
    return np.diff(data, axis=1)

def compute_violation_area(data):
    # Sum of all positive increments (monotonicity violations)
    derivs = get_derivatives(data)
    violations = np.maximum(0, derivs)
    # Return average violation area per curve * 1000 (for table scaling)
    return np.mean(np.sum(violations, axis=1)) * 1000

def compute_dtw_score(real, fake, n_samples=100):
    # Calculate pairwise DTW on a random subset for speed
    idx_real = np.random.choice(len(real), min(len(real), n_samples), replace=False)
    idx_fake = np.random.choice(len(fake), n_samples, replace=False)
    scores = []
    for r, f in zip(real[idx_real], fake[idx_fake]):
        if HAS_DTW:
            scores.append(dtw(r, f))
        else:
            scores.append(np.linalg.norm(r - f))
    return np.mean(scores)

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

nasa_deriv_real = get_derivatives(nasa_real).flatten()
calce_deriv_real = get_derivatives(calce_real).flatten()

# --- Define Model Configurations ---
experiments = ["baseline", "tv", "monotonicity", "all_priors"]

def evaluate_dataset(dataset_name, real_data, real_deriv_flat, base_dir):
    print(f"\n{'='*80}")
    print(f" EVALUATING {dataset_name.upper()} DATASET (5 SEEDS)")
    print(f"{'='*80}")
    
    # Pre-select real subset for MMD to keep comparisons consistent
    real_subset = real_data[np.random.choice(real_data.shape[0], min(1000, real_data.shape[0]), replace=False)]
    
    for exp in experiments:
        dtw_list, deriv_wd_list, viol_area_list, mmd_list = [], [], [], []
        
        for seed in seeds:
            model_path = os.path.join(base_dir, f"{exp}_seed_{seed}", "generator_final.pth")
            gen = load_generator(model_path)
            
            if gen is None:
                continue
                
            fake_data = generate_data(gen, num_samples)
            fake_deriv = get_derivatives(fake_data)
            
            # Compute Metrics
            dtw_list.append(compute_dtw_score(real_data, fake_data))
            deriv_wd_list.append(wasserstein_distance(real_deriv_flat, fake_deriv.flatten()))
            viol_area_list.append(compute_violation_area(fake_data))
            mmd_list.append(compute_mmd(real_subset, fake_data))
            
        if len(dtw_list) > 0:
            print(f"--- {exp.upper()} ---")
            print(f" Mean DTW      : {np.mean(dtw_list):.4f} ± {np.std(dtw_list):.4f}")
            print(f" Deriv WD      : {np.mean(deriv_wd_list):.6f} ± {np.std(deriv_wd_list):.6f}")
            print(f" Viol Area x10³: {np.mean(viol_area_list):.3f} ± {np.std(viol_area_list):.3f}")
            print(f" MMD           : {np.mean(mmd_list):.6f} ± {np.std(mmd_list):.6f}\n")
        else:
            print(f"--- {exp.upper()} --- Models not found.\n")

# Run Evaluation
evaluate_dataset("NASA", nasa_real, nasa_deriv_real, "saved_models")
evaluate_dataset("CALCE", calce_real, calce_deriv_real, "saved_models_calce")