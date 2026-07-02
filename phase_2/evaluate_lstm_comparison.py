import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import sys
import os

# Import both architectures
from src.models import Generator as CNNGenerator
from baselines.models_lstm import LSTMGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim, seq_length, num_samples = 100, 50, 2000 

print("Loading Real Test Data...")
real_test_data = torch.load("data/processed/test_windows.pt", weights_only=True).numpy()

def calculate_derivatives(curves):
    diffs = curves[:, 1:] - curves[:, :-1]
    return diffs.flatten()

print("Loading Models...")
# 1. Load your best CNN models (Seed 10)
cnn_base = CNNGenerator(latent_dim, seq_length).to(device)
cnn_base.load_state_dict(torch.load("saved_models/baseline_seed_10/generator_final.pth", weights_only=True))
cnn_base.eval()

cnn_phys = CNNGenerator(latent_dim, seq_length).to(device)
cnn_phys.load_state_dict(torch.load("saved_models/all_priors_seed_10/generator_final.pth", weights_only=True))
cnn_phys.eval()

# 2. Load the new LSTM models
lstm_base = LSTMGenerator(latent_dim, seq_length).to(device)
lstm_base.load_state_dict(torch.load("saved_models/lstm_baseline/generator_final.pth", weights_only=True))
lstm_base.eval()

lstm_phys = LSTMGenerator(latent_dim, seq_length).to(device)
lstm_phys.load_state_dict(torch.load("saved_models/lstm_physics/generator_final.pth", weights_only=True))
lstm_phys.eval()

print("Generating Data and Calculating Derivatives...")
with torch.no_grad():
    z = torch.randn(num_samples, latent_dim).to(device)
    cnn_base_curves = cnn_base(z).cpu().numpy()
    cnn_phys_curves = cnn_phys(z).cpu().numpy()
    lstm_base_curves = lstm_base(z).cpu().numpy()
    lstm_phys_curves = lstm_phys(z).cpu().numpy()

real_derivs = calculate_derivatives(real_test_data)
cnn_base_derivs = calculate_derivatives(cnn_base_curves)
cnn_phys_derivs = calculate_derivatives(cnn_phys_curves)
lstm_base_derivs = calculate_derivatives(lstm_base_curves)
lstm_phys_derivs = calculate_derivatives(lstm_phys_curves)

# --- QUANTITATIVE METRICS ---
wd_cnn_base = wasserstein_distance(real_derivs, cnn_base_derivs)
wd_cnn_phys = wasserstein_distance(real_derivs, cnn_phys_derivs)
wd_lstm_base = wasserstein_distance(real_derivs, lstm_base_derivs)
wd_lstm_phys = wasserstein_distance(real_derivs, lstm_phys_derivs)

print("\n" + "="*70)
print("WASSERSTEIN DISTANCE: CNN vs LSTM ARCHITECTURES")
print("="*70)
print(f"CNN Baseline:          {wd_cnn_base:.6f}")
print(f"CNN Physics-Informed:  {wd_cnn_phys:.6f} <-- (Should be best)")
print("-" * 70)
print(f"LSTM Baseline:         {wd_lstm_base:.6f}")
print(f"LSTM Physics-Informed: {wd_lstm_phys:.6f} <-- (Proves LSTM instability)")
print("="*70)

# --- VISUAL PROOF: THE CDF PLOT ---
plt.figure(figsize=(12, 8))

# Sort arrays for CDF plotting
plt.plot(np.sort(real_derivs), np.linspace(0, 1, len(real_derivs)), label='Real (NASA)', color='black', linewidth=3)
plt.plot(np.sort(cnn_base_derivs), np.linspace(0, 1, len(cnn_base_derivs)), label=f'CNN Baseline (WD: {wd_cnn_base:.4f})', color='red', linestyle='--')
plt.plot(np.sort(cnn_phys_derivs), np.linspace(0, 1, len(cnn_phys_derivs)), label=f'CNN Physics (WD: {wd_cnn_phys:.4f})', color='green', linewidth=2)
plt.plot(np.sort(lstm_base_derivs), np.linspace(0, 1, len(lstm_base_derivs)), label=f'LSTM Baseline (WD: {wd_lstm_base:.4f})', color='orange', linestyle=':')
plt.plot(np.sort(lstm_phys_derivs), np.linspace(0, 1, len(lstm_phys_derivs)), label=f'LSTM Physics (WD: {wd_lstm_phys:.4f})', color='purple', linestyle='-.')

plt.xlim([-0.05, 0.05]) # Focus on the active degradation region
plt.title('CDF of Cycle-to-Cycle Capacity Changes: Architecture Comparison', fontsize=14)
plt.xlabel('\u0394 Capacity', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig5_cdf_architecture_comparison.png', dpi=300)
print("\nSaved CDF plot as 'fig5_cdf_architecture_comparison.png'.")