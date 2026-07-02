import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --- 1. DEFINE THE PROGNOSTIC REGRESSOR ---
# A lightweight model to predict the capacity at step 50 given steps 1-49
class CapacityPredictor(nn.Module):
    def __init__(self):
        super(CapacityPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(49, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- 2. PREPARE THE DATA ---
print("Preparing Downstream Task: Next-Cycle Capacity Forecasting...")
real_train = torch.load("data/processed/calce_train_windows.pt", weights_only=True).squeeze(-1).numpy()
real_test = torch.load("data/processed/calce_test_windows.pt", weights_only=True).squeeze(-1).numpy()

# Simulate Data Scarcity: Use only 250 real windows (about 7% of CALCE)
small_real_train = real_train[:250]

# Split into Input (first 49 steps) and Target (last 1 step)
def split_io(data):
    return torch.tensor(data[:, :49], dtype=torch.float32), torch.tensor(data[:, 49:], dtype=torch.float32)

X_test, y_test = split_io(real_test)

# --- 3. GENERATE SYNTHETIC AUGMENTATION DATA ---
def generate_synthetic(model_path, num_samples=2000):
    if not os.path.exists(model_path): return None
    gen = Generator(latent_dim=100, seq_length=50).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    gen.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 100).to(device)
        return gen(z).cpu().numpy()

print("Generating 2000 Synthetic Windows per Model...")
base_fake = generate_synthetic("saved_models_calce/baseline_seed_10/generator_final.pth")
tv_fake = generate_synthetic("saved_models_calce/tv_seed_10/generator_final.pth")
phys_fake = generate_synthetic("saved_models_calce/all_priors_seed_10/generator_final.pth")

# --- 4. TRAINING PROTOCOL ---
def train_and_evaluate(X_train, y_train, name):
    if X_train is None: return "N/A"
    
    model = CapacityPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_train_t = X_train.to(device)
    y_train_t = y_train.to(device)
    X_test_t = X_test.to(device)
    y_test_t = y_test.to(device)
    
    # Train for 200 epochs
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
    # Evaluate on the REAL unseen test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        mse = criterion(test_preds, y_test_t).item()
        rmse = np.sqrt(mse)
        
    return rmse

# --- 5. RUN EXPERIMENTS ---
print("\nTraining Regressors (200 Epochs each)...")

# Experiment A: Real Only
X_real, y_real = split_io(small_real_train)
rmse_real = train_and_evaluate(X_real, y_real, "Real Only")

# Experiment B: Real + Baseline
X_base, y_base = split_io(np.vstack((small_real_train, base_fake)))
rmse_base = train_and_evaluate(X_base, y_base, "Real + Baseline")

# Experiment C: Real + TV
X_tv, y_tv = split_io(np.vstack((small_real_train, tv_fake)))
rmse_tv = train_and_evaluate(X_tv, y_tv, "Real + TV")

# Experiment D: Real + Physics
X_phys, y_phys = split_io(np.vstack((small_real_train, phys_fake)))
rmse_phys = train_and_evaluate(X_phys, y_phys, "Real + Physics")

# --- 6. PRINT RESULTS ---
print("\n" + "="*50)
print(" DOWNSTREAM PROGNOSTIC UTILITY (RMSE ON TEST SET)")
print(" (Lower RMSE is better)")
print("="*50)
print(f" 1. Real Data Only (250 samples)   : {rmse_real:.6f}")
print("-" * 50)
print(f" 2. Augmented (Real + Baseline)    : {rmse_base:.6f}")
print(f" 3. Augmented (Real + TV)          : {rmse_tv:.6f}")
print(f" 4. Augmented (Real + Physics)     : {rmse_phys:.6f} <-- THE ULTIMATE TEST")
print("="*50)