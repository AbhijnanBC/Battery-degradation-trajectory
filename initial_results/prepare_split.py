import torch
import numpy as np

# Load the full dataset
data_path = "data/processed/battery_windows.pt"
all_windows = torch.load(data_path, weights_only=True).numpy()

# Randomly shuffle the data
np.random.seed(42)
np.random.shuffle(all_windows)

# Split 80% Training / 20% Testing
split_idx = int(len(all_windows) * 0.8)
train_windows = all_windows[:split_idx]
test_windows = all_windows[split_idx:]

# Save them separately
torch.save(torch.tensor(train_windows, dtype=torch.float32), "data/processed/train_windows.pt")
torch.save(torch.tensor(test_windows, dtype=torch.float32), "data/processed/test_windows.pt")

print(f"Total: {len(all_windows)}")
print(f"Train: {len(train_windows)} | Test: {len(test_windows)}")