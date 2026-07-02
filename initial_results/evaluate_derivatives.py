import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from src.models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim, seq_length, num_samples = 100, 50, 2000 

print("Loading Real Test Data...")
real_test_data = torch.load("data/processed/test_windows.pt", weights_only=True).numpy()

def calculate_derivatives(curves):
    """Calculates the discrete derivative (slope) at every time step."""
    # diffs shape: [num_samples, seq_length - 1]
    diffs = curves[:, 1:] - curves[:, :-1]
    # Flatten to get a 1D array of all slopes across all samples
    return diffs.flatten()

print("Loading Models...")
g_baseline = Generator(latent_dim, seq_length).to(device)
g_baseline.load_state_dict(torch.load("saved_models/baseline_seed_10/generator_final.pth", weights_only=True))
g_baseline.eval()

g_tv = Generator(latent_dim, seq_length).to(device)
g_tv.load_state_dict(torch.load("saved_models/tv_seed_10/generator_final.pth", weights_only=True))
g_tv.eval()

g_phys = Generator(latent_dim, seq_length).to(device)
g_phys.load_state_dict(torch.load("saved_models/all_priors_seed_10/generator_final.pth", weights_only=True))
g_phys.eval()

print("Generating Data and Calculating Derivatives...")
with torch.no_grad():
    z = torch.randn(num_samples, latent_dim).to(device)
    base_curves = g_baseline(z).cpu().numpy()
    tv_curves = g_tv(z).cpu().numpy()
    phys_curves = g_phys(z).cpu().numpy()

# Calculate the flattened arrays of all slopes
real_derivs = calculate_derivatives(real_test_data)
base_derivs = calculate_derivatives(base_curves)
tv_derivs = calculate_derivatives(tv_curves)
phys_derivs = calculate_derivatives(phys_curves)

# ---------------------------------------------------------
# The Quantitative Proof: Wasserstein Distance of Derivatives
# ---------------------------------------------------------
# How similar is the distribution of generated slopes to the real slopes? (Lower is better)
wd_base = wasserstein_distance(real_derivs, base_derivs)
wd_tv = wasserstein_distance(real_derivs, tv_derivs)
wd_phys = wasserstein_distance(real_derivs, phys_derivs)

print("\n" + "="*60)
print("WASSERSTEIN DISTANCE OF DERIVATIVE DISTRIBUTIONS (vs REAL)")
print("="*60)
print(f"Baseline WGAN:       {wd_base:.6f}")
print(f"Generic Smooth (TV): {wd_tv:.6f}")
print(f"Physics-Inspired:    {wd_phys:.6f}")
print("="*60)
print("Conclusion: If Physics < TV, we mathematically prove TV over-smooths.")

# ---------------------------------------------------------
# The Visual Proof: Histograms of Capacity Changes
# ---------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
fig.suptitle('Distribution of Cycle-to-Cycle Capacity Changes (\u0394 Capacity)', fontsize=16)

# We define strict bins to make the histograms directly comparable
bins = np.linspace(-0.1, 0.1, 100) 

axs[0].hist(real_derivs, bins=bins, color='black', alpha=0.7, density=True)
axs[0].set_title('Real Degradation (NASA)')
axs[0].set_xlabel('\u0394 Capacity')

axs[1].hist(base_derivs, bins=bins, color='red', alpha=0.7, density=True)
axs[1].set_title(f'Baseline WGAN\n(WD: {wd_base:.4f})')
axs[1].set_xlabel('\u0394 Capacity')

axs[2].hist(tv_derivs, bins=bins, color='blue', alpha=0.7, density=True)
axs[2].set_title(f'Generic Smoothing (TV)\n(WD: {wd_tv:.4f})')
axs[2].set_xlabel('\u0394 Capacity')

axs[3].hist(phys_derivs, bins=bins, color='green', alpha=0.7, density=True)
axs[3].set_title(f'Physics-Inspired\n(WD: {wd_phys:.4f})')
axs[3].set_xlabel('\u0394 Capacity')

for ax in axs:
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1) # Mark the 0 line (Monotonic boundary)

plt.tight_layout()
plt.savefig('fig3_derivative_distributions.png', dpi=300)
print("\nSaved histogram comparison as 'fig3_derivative_distributions.png'.")

# --- Add this to evaluate_derivatives.py to generate the CDF ---
plt.figure(figsize=(10, 6))
# Sort the arrays for CDF plotting
plt.plot(np.sort(real_derivs), np.linspace(0, 1, len(real_derivs)), label='Real (NASA)', color='black', linewidth=2)
plt.plot(np.sort(base_derivs), np.linspace(0, 1, len(base_derivs)), label='Baseline WGAN', color='red', linestyle='--')
plt.plot(np.sort(tv_derivs), np.linspace(0, 1, len(tv_derivs)), label='Generic Smoothing (TV)', color='blue', linestyle=':')
plt.plot(np.sort(phys_derivs), np.linspace(0, 1, len(phys_derivs)), label='Physics-Inspired', color='green', linewidth=2)

plt.xlim([-0.05, 0.05]) # Focus on the active region
plt.title('CDF of Cycle-to-Cycle Capacity Changes')
plt.xlabel('\u0394 Capacity')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.savefig('fig4_cdf_derivatives.png', dpi=300)