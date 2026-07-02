import torch
import matplotlib.pyplot as plt
from src.models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
seq_length = 50

# Initialize three generators
g_0 = Generator(latent_dim, seq_length).to(device)
g_5 = Generator(latent_dim, seq_length).to(device)
g_50 = Generator(latent_dim, seq_length).to(device)

# Load the weights (ensure you point to the correct WGAN folders)
g_0.load_state_dict(torch.load("saved_models/wgan_lambda_0.0/generator_final.pth", weights_only=True))
g_5.load_state_dict(torch.load("saved_models/wgan_lambda_5.0/generator_final.pth", weights_only=True))
g_50.load_state_dict(torch.load("saved_models/wgan_lambda_50.0/generator_final.pth", weights_only=True))

g_0.eval()
g_5.eval()
g_50.eval()

# Generate fake data from the same latent noise for a fair comparison
z = torch.randn(1, latent_dim).to(device)

with torch.no_grad():
    curve_0 = g_0(z).cpu().numpy().flatten()
    curve_5 = g_5(z).cpu().numpy().flatten()
    curve_50 = g_50(z).cpu().numpy().flatten()

# Plot them side-by-side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(curve_0, color='red', marker='o', markersize=3)
axs[0].set_title('Baseline (Lambda = 0.0)')
axs[0].grid(True)

axs[1].plot(curve_5, color='blue', marker='o', markersize=3)
axs[1].set_title('Physics-Guided (Lambda = 5.0)')
axs[1].grid(True)

axs[2].plot(curve_50, color='green', marker='o', markersize=3)
axs[2].set_title('Physics-Dominated (Lambda = 50.0)')
axs[2].grid(True)

plt.tight_layout()
plt.show()