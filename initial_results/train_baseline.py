import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Import your architectures
from src.models import Generator, Discriminator

# 1. Hyperparameters
latent_dim = 100
seq_length = 50 
num_epochs = 5000
batch_size = 32
learning_rate = 0.0002

# 2. Hardware Setup (Utilizing the University GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 3. Data Loading
data_path = "data/processed/battery_windows.pt"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing {data_path}. Ensure the extraction notebook was run.")

# Load tensor and ensure it's the right data type
training_tensor = torch.load(data_path).to(torch.float32)
dataset = TensorDataset(training_tensor)

# drop_last=True prevents crashes if dataset size isn't perfectly divisible by batch_size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"Total training windows: {len(dataset)}")
print(f"Batches per epoch: {len(dataloader)}")

# 4. Initialize Models
generator = Generator(latent_dim, seq_length).to(device)
discriminator = Discriminator(seq_length).to(device)

# 5. Loss and Optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 6. The Training Loop
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (real_batch,) in enumerate(dataloader):
        
        real_batch = real_batch.to(device)
        current_batch_size = real_batch.size(0)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Labels
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        # Loss on real data
        output_real = discriminator(real_batch)
        d_loss_real = criterion(output_real, real_labels)
        
        # Loss on fake data
        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_batch = generator(z)
        output_fake = discriminator(fake_batch.detach()) # Detach generator to avoid training it here
        d_loss_fake = criterion(output_fake, fake_labels)
        
        # Total D Loss and backprop
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # We want the generator to fool the discriminator (aim for label 1)
        output_fake_for_G = discriminator(fake_batch)
        g_loss = criterion(output_fake_for_G, real_labels)
        
        g_loss.backward()
        optimizer_G.step()

    # 7. Print Progress
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# 8. Save the Baseline Models Safely
os.makedirs("saved_models", exist_ok=True)
torch.save(generator.state_dict(), "saved_models/generator_baseline.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator_baseline.pth")
print("Training complete. Baseline models saved to 'saved_models/'.")