import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.autograd as autograd
from utils import *
from model import *


# Specify dataset path and device configuration
dataset_path = r"ADBench_datasets/47_yeast.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 4.1: Load and preprocess data
X_all, y_all = load_adbench_data(dataset_path)
input_dim = X_all.shape[1]

# Standardize features using StandardScaler
scaler = StandardScaler()
X_all = torch.tensor(scaler.fit_transform(X_all)).float()

# Split data into train and test sets
D_train_np, D_test_np, y_train_np, y_test_np = train_test_split(
    X_all.numpy(), y_all.numpy(), test_size=0.6, random_state=42, stratify=y_all
)
D_train = torch.tensor(D_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
D_test  = torch.tensor(D_test_np,  dtype=torch.float32)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32)

train_dataset = TensorDataset(D_train, y_train)
dataloader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Training WGAN
n_epochs = 150
batch_size = 64
lr = 0.0002
b1 = 0
b2 = 0.9
latent_dim = 64
input_dim = D_test.shape[1]  # Can be changed for different datasets
n_classes = 1
n_critic = 5
sample_interval = 400
dataset_name = "mnist"
lambda_gp = 10

cuda = True if torch.cuda.is_available() else False

print(D_train.shape)
generator = Generator(latent_dim=latent_dim, n_classes= 1, output_dim=input_dim)
discriminator = Discriminator(input_dim, n_classes=1)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP."""
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Training loop 
batches_done = 0
for epoch in range(150):
    for i, (data, labels) in enumerate(dataloader):
        batch_size = data.shape[0]
        real_data = data.view(batch_size, -1).type(Tensor)
        labels = labels.type(LongTensor)

        # Train Discriminator
        optimizer_D.zero_grad()
        z = Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))
        
        fake_data = generator(z, labels)
        real_validity = discriminator(real_data, labels)
        fake_validity = discriminator(fake_data, labels)
        gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data, labels.data)
        
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty # WASSERSTEIN DISTANCE
        d_loss.backward()
        optimizer_D.step()

        # Train Generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            fake_data = generator(z, labels)
            fake_validity = discriminator(fake_data, labels)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            
            batches_done += n_critic

# Separate minority and majority classes
minority_mask = (y_train == 1)
X_minority = D_train[minority_mask]
majority_mask = (y_train == 0)
X_majority = D_train[majority_mask]

# Calculate the number of synthetic samples to generate
num_generate = len(X_majority) - len(X_minority)

with torch.no_grad():
    # Generate latent variables uniformly within [-2,2]
    z_uniform = (torch.rand(num_generate, latent_dim) * 4.0) - 2.0
    z_uniform = z_uniform.to(device)

    # Assign synthetic labels (e.g., oversample minority class with y=1)
    y_synthetic_c = torch.full((num_generate, 1), 0.9, device=device)

    # Decode synthetic data
    X_synthetic = generator(z_uniform, y_synthetic_c)
    X_synthetic = X_synthetic.cpu()

# Create labels for synthetic samples
y_synthetic_labels = torch.ones(num_generate)

# Combine synthetic data with original training data
D_train_final = torch.cat([D_train, X_synthetic], dim=0)
y_train_final = torch.cat([y_train, y_synthetic_labels], dim=0)

# Step 4.4: Train TransformerDetector on the augmented dataset
train_dataset_final = TensorDataset(D_train_final, y_train_final)
test_dataset = TensorDataset(D_test, y_test)
train_loader_final = DataLoader(train_dataset_final, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print("After oversampling with Generator:")
unique, counts = np.unique(y_train_final.numpy(), return_counts=True)
print("Class distribution in training set:", dict(zip(unique, counts)))

# Initialize Transformer Detector model
model = TransformerDetector(input_size=input_dim).to(device)
optimizer_tf = Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Train the Transformer Detector model
num_epochs_tf = 50
for epoch in range(num_epochs_tf):
    train_loss = train_detector(model, train_loader_final, optimizer_tf, criterion, device)
    print(f"[Transformer] Epoch {epoch+1}/{num_epochs_tf}, Loss={train_loss:.4f}")
    print("-" * 40)

# Ensure the directory for saving models exists
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

# Save the trained Beta-CVAE model
gen_path = os.path.join(save_dir, "gen.pth")
disc_path = os.path.join(save_dir, "disc.pth")
torch.save(generator.state_dict(), gen_path)
print(f"Generator model saved to: {gen_path}")

torch.save(discriminator.state_dict(), disc_path)
print(f"Critic model saved to: {disc_path}")

# Save the trained TransformerDetector model
detector_path = os.path.join(save_dir, "transformer_detector.pth")
torch.save(model.state_dict(), detector_path)
print(f"TransformerDetector model saved to: {detector_path}")    
