import torch
import numpy as np
import pandas as pd
from fastdtw import fastdtw
import os
import itertools

# --- THIS WAS THE MISSING LINE THAT CAUSED THE CRASH ---
from src.models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
seq_length = 50
num_samples = 1000 
eval_subset_size = 100 

print("\nLoading STRICTLY TEST Data...")
test_data_path = "data/processed/test_windows.pt"
if not os.path.exists(test_data_path):
    raise FileNotFoundError("Missing test_windows.pt. Please run prepare_split.py first.")
real_test_data = torch.load(test_data_path, weights_only=True).numpy()

def calc_advanced_physics_metrics(curves):
    diffs = curves[:, 1:] - curves[:, :-1]
    positive_diffs = np.where(diffs > 0, diffs, 0)
    violation_area = np.mean(np.sum(positive_diffs, axis=1))
    max_violation = np.max(positive_diffs) if positive_diffs.size > 0 else 0
    return violation_area, max_violation

def calc_similarity_and_diversity(generated_curves, real_test_curves, sample_size=100):
    gen_subset = generated_curves[:sample_size]
    dtw_scores = []
    mse_scores = []
    
    for gen_curve in gen_subset:
        best_dtw = float('inf')
        best_mse = float('inf')
        for real_curve in real_test_curves:
            mse = np.mean((gen_curve - real_curve) ** 2)
            if mse < best_mse: 
                best_mse = mse
                
            distance, _ = fastdtw(gen_curve, real_curve, dist=lambda x, y: np.abs(x - y))
            if distance < best_dtw: 
                best_dtw = distance
                
        dtw_scores.append(best_dtw)
        mse_scores.append(best_mse)
        
    pairwise_distances = []
    for idx_1, idx_2 in itertools.combinations(range(len(gen_subset)), 2):
        mse_between_fakes = np.mean((gen_subset[idx_1] - gen_subset[idx_2]) ** 2)
        pairwise_distances.append(mse_between_fakes)
    apd = np.mean(pairwise_distances)
        
    return np.mean(dtw_scores), np.mean(mse_scores), apd

models = ["baseline", "tv", "monotonicity", "all_priors"]
seeds = [10, 20, 30, 40, 50]

all_results = {m: {"area": [], "max": [], "dtw": [], "mse": [], "apd": []} for m in models}

print("Running statistical evaluations across all experimental configurations...")
for m in models:
    for s in seeds:
        path = f"saved_models/{m}_seed_{s}/generator_final.pth"
        if not os.path.exists(path):
            print(f"Warning: Checkpoint missing for {m} at seed {s}. Skipping.")
            continue
            
        g = Generator(latent_dim, seq_length).to(device)
        g.load_state_dict(torch.load(path, weights_only=True))
        g.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim).to(device)
            fake_curves = g(z).cpu().numpy()
            
        area, max_viol = calc_advanced_physics_metrics(fake_curves)
        dtw, mse, apd = calc_similarity_and_diversity(fake_curves, real_test_data, sample_size=eval_subset_size)
        
        all_results[m]["area"].append(area)
        all_results[m]["max"].append(max_viol)
        all_results[m]["dtw"].append(dtw)
        all_results[m]["mse"].append(mse)
        all_results[m]["apd"].append(apd)

final_table = []
for m in models:
    if len(all_results[m]["dtw"]) == 0: 
        continue
    
    final_table.append({
        "Model Configuration": m.upper(),
        "Violation Area (x10^-3)": f"{np.mean(all_results[m]['area'])*1000:.3f} ± {np.std(all_results[m]['area'])*1000:.3f}",
        "Max Single Viol.": f"{np.mean(all_results[m]['max']):.4f} ± {np.std(all_results[m]['max']):.4f}",
        "Mean DTW": f"{np.mean(all_results[m]['dtw']):.4f} ± {np.std(all_results[m]['dtw']):.4f}",
        "Mean MSE": f"{np.mean(all_results[m]['mse']):.4f} ± {np.std(all_results[m]['mse']):.4f}",
        "Diversity (APD)": f"{np.mean(all_results[m]['apd']):.4f} ± {np.std(all_results[m]['apd']):.4f}"
    })

df = pd.DataFrame(final_table)
print("\n" + "="*110)
print("FINAL PUBLICATION MATRIX (5 SEEDS, TEST DATA ONLY)")
print("="*110)
print(df.to_markdown(index=False))
print("="*110)