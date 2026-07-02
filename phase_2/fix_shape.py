import torch
import os

PROCESSED_DATA_DIR = "data/processed/"

print("Loading tensors...")
train_tensor = torch.load(os.path.join(PROCESSED_DATA_DIR, "calce_train_windows.pt"), weights_only=True)
test_tensor = torch.load(os.path.join(PROCESSED_DATA_DIR, "calce_test_windows.pt"), weights_only=True)

# Squeeze out the last dimension: [N, 50, 1] -> [N, 50]
train_tensor = train_tensor.squeeze(-1)
test_tensor = test_tensor.squeeze(-1)

# Overwrite the files
torch.save(train_tensor, os.path.join(PROCESSED_DATA_DIR, "calce_train_windows.pt"))
torch.save(test_tensor, os.path.join(PROCESSED_DATA_DIR, "calce_test_windows.pt"))

print("--- SHAPES FIXED ---")
print(f"New Train Shape: {train_tensor.shape}")
print(f"New Test Shape:  {test_tensor.shape}")
print("You can now run 'python run_calce_all.py'")