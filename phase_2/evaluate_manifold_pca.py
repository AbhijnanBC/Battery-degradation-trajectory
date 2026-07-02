import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 2000

print("Loading Real CALCE Data...")
real_data = torch.load("data/processed/calce_test_windows.pt", weights_only=True).squeeze(-1).numpy()

def load_generator(model_path):
    if not os.path.exists(model_path):
        return None
    gen = Generator(latent_dim=100, seq_length=50).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    gen.eval()
    return gen

def generate_curves(gen, n=num_samples):
    if gen is None: return None
    with torch.no_grad():
        z = torch.randn(n, 100).to(device)
        return gen(z).cpu().numpy()

def compute_physical_stats(curves):
    if curves is None: return None, None, None
    means = np.mean(curves, axis=1)
    variances = np.var(curves, axis=1)
    slopes = curves[:, -1] - curves[:, 0]  # End - Start
    return means, variances, slopes

print("Loading Models and Generating Data...")
gen_base = load_generator("saved_models_calce/baseline_seed_10/generator_final.pth")
gen_tv = load_generator("saved_models_calce/tv_seed_10/generator_final.pth")
gen_phys = load_generator("saved_models_calce/all_priors_seed_10/generator_final.pth")

base_data = generate_curves(gen_base)
tv_data = generate_curves(gen_tv)
phys_data = generate_curves(gen_phys)

# --- 1. PHYSICAL SUMMARY STATISTICS ---
print("\n--- Physical Summary Statistics (Mean of the metric across all windows) ---")
real_means, real_vars, real_slopes = compute_physical_stats(real_data)
print(f"REAL     | Mean Cap: {np.mean(real_means):.4f} | Variance: {np.mean(real_vars):.6f} | Avg Slope: {np.mean(real_slopes):.4f}")

if base_data is not None:
    b_means, b_vars, b_slopes = compute_physical_stats(base_data)
    print(f"BASELINE | Mean Cap: {np.mean(b_means):.4f} | Variance: {np.mean(b_vars):.6f} | Avg Slope: {np.mean(b_slopes):.4f}")

if tv_data is not None:
    t_means, t_vars, t_slopes = compute_physical_stats(tv_data)
    print(f"TV       | Mean Cap: {np.mean(t_means):.4f} | Variance: {np.mean(t_vars):.6f} | Avg Slope: {np.mean(t_slopes):.4f}")

if phys_data is not None:
    p_means, p_vars, p_slopes = compute_physical_stats(phys_data)
    print(f"PHYSICS  | Mean Cap: {np.mean(p_means):.4f} | Variance: {np.mean(p_vars):.6f} | Avg Slope: {np.mean(p_slopes):.4f}")

# --- 2. PCA MANIFOLD VISUALIZATION ---
print("\nFitting PCA on Real Data...")
# We fit PCA on the real data to define the 'true' physical space, then project the fakes into it
pca = PCA(n_components=2)
pca.fit(real_data)

real_pca = pca.transform(real_data)

plt.figure(figsize=(10, 8))
plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, label='Real (CALCE)', color='black', s=10)

if base_data is not None:
    base_pca = pca.transform(base_data)
    plt.scatter(base_pca[:, 0], base_pca[:, 1], alpha=0.3, label='Baseline', color='red', s=10)

if tv_data is not None:
    tv_pca = pca.transform(tv_data)
    plt.scatter(tv_pca[:, 0], tv_pca[:, 1], alpha=0.3, label='TV Constraint', color='blue', s=10)

if phys_data is not None:
    phys_pca = pca.transform(phys_data)
    plt.scatter(phys_pca[:, 0], phys_pca[:, 1], alpha=0.5, label='All Physics Priors', color='green', s=15)

plt.title("PCA Projection of Generative Manifold Coverage (CALCE)")
plt.xlabel("Principal Component 1 (Primary Degradation Trend)")
plt.ylabel("Principal Component 2 (Local Variance/Recovery)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig6_pca_manifold.png", dpi=300)
print("Saved PCA Manifold plot to 'fig6_pca_manifold.png'")