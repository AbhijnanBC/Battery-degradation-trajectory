import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("src"))
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DEFINE THE REGRESSOR ---
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

# --- 2. LOAD REAL DATA ---
print("Preparing Downstream Task: Next-Cycle Capacity Forecasting...")
real_train = torch.load("data/processed/calce_train_windows.pt", weights_only=True).squeeze(-1).numpy()
real_test = torch.load("data/processed/calce_test_windows.pt", weights_only=True).squeeze(-1).numpy()

# The 250-sample baseline
small_real_train = real_train[:250]

def split_io(data):
    return torch.tensor(data[:, :49], dtype=torch.float32), torch.tensor(data[:, 49:], dtype=torch.float32)

X_test, y_test = split_io(real_test)
X_test_t, y_test_t = X_test.to(device), y_test.to(device)

# --- 3. LOAD PHYSICS SYNTHETIC DATA ---
print("Generating 2000 Synthetic Windows from Physics Model...")
model_path = "saved_models_calce/all_priors_seed_10/generator_final.pth"

if not os.path.exists(model_path):
    print("CRITICAL: Physics model not found.")
    sys.exit(1)

gen = Generator(latent_dim=100, seq_length=50).to(device)
gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
gen.eval()

with torch.no_grad():
    z = torch.randn(2000, 100).to(device)
    phys_fake = gen(z).cpu().numpy()

# --- 4. TRAINING PROTOCOL (WITH STRICT SHUFFLING) ---
def train_and_evaluate(real_data, synthetic_data, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Combine data
    if synthetic_data is not None:
        combined_data = np.vstack((real_data, synthetic_data))
    else:
        combined_data = real_data
        
    # STRICT SHUFFLING: Prevent batch memorization
    shuffled_indices = np.random.permutation(len(combined_data))
    combined_data = combined_data[shuffled_indices]
    
    X_train, y_train = split_io(combined_data)
    X_train_t, y_train_t = X_train.to(device), y_train.to(device)
    
    model = CapacityPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        mse = criterion(test_preds, y_test_t).item()
        
    return np.sqrt(mse)

# --- 5. EXECUTE THE ABLATION STUDY ---
print("\nExecuting Augmentation Ratio Ablation Study...")
print("Real Data fixed at 250 samples.")

results = {}

# Baseline: 0 Synthetic
results[0] = train_and_evaluate(small_real_train, None)

# Ablation: 250, 500, 1000, 2000 Synthetic
synthetic_counts = [250, 500, 1000, 2000]
for count in synthetic_counts:
    subset_fake = phys_fake[:count]
    results[count] = train_and_evaluate(small_real_train, subset_fake)

# --- 6. PRINT RESULTS ---
print("\n" + "="*50)
print(" DOWNSTREAM UTILITY: AUGMENTATION RATIO ABLATION")
print("="*50)
print(f" Real (250) +    0 Synthetic : {results[0]:.6f} (Baseline)")
print("-" * 50)
print(f" Real (250) +  250 Synthetic : {results[250]:.6f}")
print(f" Real (250) +  500 Synthetic : {results[500]:.6f}")
print(f" Real (250) + 1000 Synthetic : {results[1000]:.6f}")
print(f" Real (250) + 2000 Synthetic : {results[2000]:.6f}")
print("="*50)