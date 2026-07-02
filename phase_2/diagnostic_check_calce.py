import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys

# Import your CNN Generator
sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("==================================================")
print("1 & 2. TENSOR SHAPE AND SCALING CHECK")
print("==================================================")
try:
    calce_train = torch.load("data/processed/calce_train_windows.pt", map_location=device, weights_only=True)
    nasa_train = torch.load("data/processed/train_windows.pt", map_location=device, weights_only=True)
    
    print(f"NASA Shape:  {nasa_train.shape}  | Min: {nasa_train.min().item():.4f}  | Max: {nasa_train.max().item():.4f}")
    print(f"CALCE Shape: {calce_train.shape} | Min: {calce_train.min().item():.4f} | Max: {calce_train.max().item():.4f}")
    
    if abs(calce_train.min().item() - nasa_train.min().item()) > 0.1 or abs(calce_train.max().item() - nasa_train.max().item()) > 0.1:
        print(">>> WARNING: Normalization mismatch detected between datasets!")
    else:
        print(">>> SUCCESS: CALCE scaling matches NASA perfectly.")
except Exception as e:
    print(f"Error loading tensors: {e}")
    sys.exit(1)

print("\n==================================================")
print("3. DATALOADER SIZE CHECK")
print("==================================================")
batch_size = 32
calce_loader = DataLoader(TensorDataset(calce_train.to(torch.float32)), batch_size=batch_size, drop_last=True)
nasa_loader = DataLoader(TensorDataset(nasa_train.to(torch.float32)), batch_size=batch_size, drop_last=True)

multiplier = len(calce_loader) / len(nasa_loader)
print(f"NASA Batches per epoch:  {len(nasa_loader)}")
print(f"CALCE Batches per epoch: {len(calce_loader)}")
print(f">>> CONCLUSION: CALCE is {multiplier:.2f}x larger. Violations should naturally be ~{multiplier:.1f}x higher.")

print("\n==================================================")
print("4 & 5. GENERATOR STATISTICAL & VISUAL SANITY CHECK")
print("==================================================")
model_path = "saved_models_calce/baseline_seed_10/generator_final.pth"

if not os.path.exists(model_path):
    print(f"CRITICAL: Model {model_path} not found.")
else:
    print("Loading baseline_seed_10...")
    generator = Generator(latent_dim=100, seq_length=50).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval()

    with torch.no_grad():
        z = torch.randn(10, 100).to(device)
        fake_curves = generator(z).cpu().numpy()

    mean_val = fake_curves.mean()
    std_val = fake_curves.std()
    
    print(f"Generated Data Mean: {mean_val:.4f}")
    print(f"Generated Data Std:  {std_val:.4f}")
    
    if std_val < 0.01:
        print(">>> CRITICAL FAILURE: Standard deviation is near zero. The model suffered severe mode collapse (outputting a flat line).")
    else:
        print(">>> SUCCESS: Standard deviation is healthy. The generator is producing varied data.")

    # Plot 10 curves
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.plot(fake_curves[i], label=f"Sample {i+1}", alpha=0.8, marker='o', markersize=2)
    
    plt.title("Visual Sanity Check: 10 Generated CALCE Curves (Baseline Seed 10)")
    plt.xlabel("Time Step (Window)")
    plt.ylabel("Normalized Capacity")
    plt.grid(True)
    plt.savefig("calce_diagnostic_plot.png", dpi=300)
    print("\nSaved diagnostic plot to 'calce_diagnostic_plot.png'.")