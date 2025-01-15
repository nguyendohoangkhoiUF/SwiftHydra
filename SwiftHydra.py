import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
from utils import *
from model import *

# Specify dataset path and device configuration
dataset_path = r"Classical/7_Cardiotocography.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths for saving Beta-CVAE and Transformer Detector models
save_dir = "./saved_models"
vae_path = os.path.join(save_dir, "beta_cvae.pth")
detector_path = os.path.join(save_dir, "transformer_detector.pth")

# Step 4.1: Load data
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

# Create DataLoader for Beta-CVAE
train_dataset = TensorDataset(D_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize Beta-CVAE and Transformer Detector models
loaded_beta_cvae = BetaCVAE(input_dim=input_dim, hidden_dim=512, latent_dim=64, beta=1.0).to(device)
loaded_detector_model = TransformerDetector(input_size=input_dim).to(device)

# Load pre-trained weights for Beta-CVAE and Detector models
loaded_beta_cvae.load_state_dict(torch.load(vae_path, weights_only=True))
loaded_detector_model.load_state_dict(torch.load(detector_path, weights_only=True))

# Set models to evaluation mode
loaded_beta_cvae.eval()
loaded_detector_model.eval()

print("Models loaded successfully.")

# Define training parameters
num_episodes = 200
num_gen_data = 50
batch_size = 128
new_detector = TransformerDetector(input_size=input_dim).to(device)
optimizer_cvae = Adam(loaded_beta_cvae.parameters(), lr=1e-3)
optimizer_detector = Adam(new_detector.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Define directories for logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
beta_cvae_log = os.path.join(log_dir, "beta_cvae.log")
detector_log = os.path.join(log_dir, "detector.log")
adversarial_log = os.path.join(log_dir, "adversarial_samples.log")
synthetic_data = []
start_idx = 0  # Initial start index for adversarial data generation

for ep in range(num_episodes):
    # Create masks for class 1 and class 0 samples
    class1_mask = (y_train == 1)  # Boolean tensor for class 1
    class0_mask = (y_train == 0)  # Boolean tensor for class 0

    # Count the number of samples in each class
    num_class1 = class1_mask.sum().item()
    num_class0 = class0_mask.sum().item()

    # Stop the process if class 1 samples exceed class 0
    if num_class1 > num_class0:
        print(f"Break at Episode {ep + 1}: Class 1 ({num_class1}) exceeds Class 0 ({num_class0}).")
        break

    # Filter class 1 samples from the training data
    D_train_grow_tensor = D_train[class1_mask]
    D_train_grow = [row for row in D_train_grow_tensor]  # List of individual samples

    print(f"===== EPISODE {ep + 1}/{num_episodes} =====")

    # Train Beta-CVAE model
    train_dataset = TensorDataset(D_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    num_epochs_cvae = 10
    for epoch in range(num_epochs_cvae):
        loss_cvae = train_beta_cvae(loaded_beta_cvae, train_loader, optimizer_cvae, device)
        log_to_file(beta_cvae_log, f"Epoch {epoch + 1}/{num_epochs_cvae}, Loss: {loss_cvae:.4f}")

    # Train Transformer Detector model
    detector_epochs = 5
    train_dataset_detector = TensorDataset(D_train, y_train)
    train_loader_detector = DataLoader(train_dataset_detector, batch_size=batch_size, shuffle=True)
    for det_epoch in range(detector_epochs):
        detector_loss = train_detector(new_detector, train_loader_detector, optimizer_detector, criterion, device)

    # Generate adversarial samples
    idx_class1 = (y_train == 1).nonzero(as_tuple=True)[0]
    new_samples, new_labels = [], []

    # Handle wrap-around indexing for adversarial data generation
    end_idx = start_idx + num_gen_data
    indices = idx_class1[start_idx:end_idx]
    if end_idx > len(idx_class1):
        indices += idx_class1[:end_idx - len(idx_class1)]

    start_idx = end_idx % len(idx_class1)

    for syn_data in range(num_gen_data):
        unique, counts = np.unique(y_train.numpy(), return_counts=True)
        print("Class distribution in training set:", dict(zip(unique, counts)))
        print("===== Generated Data {} =====".format(syn_data))
        random_idx = random.choice(idx_class1)
        x_orig = D_train[random_idx]
        x_adv = One_Step_To_Feasible_Action(
            beta_cvae=loaded_beta_cvae,
            detector=loaded_detector_model,
            x_orig=x_orig,
            device=device,
            previously_generated=D_train_grow,
            alpha=1.0,
            lambda_div=0.1,
            lr=0.01,
            steps=20,
            log_file=adversarial_log,
        )
        new_samples.append(x_adv.unsqueeze(0))
        new_labels.append(torch.tensor([1]))
        synthetic_data.append(x_adv)

    # Append newly generated samples to training data
    new_samples = torch.cat(new_samples, dim=0)
    new_labels = torch.cat(new_labels, dim=0)
    D_train = torch.cat([D_train, new_samples], dim=0)
    y_train = torch.cat([y_train, new_labels], dim=0)

# Step 4.4: Visualize generated data using T-SNE
plt.style.use('default')

# Prepare data for T-SNE visualization
X_synthetic = np.asarray(synthetic_data)
X_synthetic = torch.tensor(X_synthetic, dtype=torch.float32)
X_plot = torch.cat([D_train, D_test, X_synthetic], dim=0).numpy()

# Prepare labels for visualization
N_train = len(D_train)
N_test = len(D_test)
N_synthetic = len(X_synthetic)
y_plot = np.concatenate([
    y_train.numpy(),            # Training labels
    y_test.numpy(),             # Test labels
    np.full((N_synthetic,), 2)  # Synthetic labels
], axis=0)

# Normalize data before applying T-SNE
scaler = StandardScaler()
X_plot_scaled = scaler.fit_transform(X_plot)

# Apply T-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
X_embedded = tsne.fit_transform(X_plot_scaled)

# Plot the T-SNE visualization
plt.figure(figsize=(10, 8), facecolor='white')

# Plot training data
idx0_train = (y_plot[:N_train] == 0)
plt.scatter(X_embedded[:N_train][idx0_train, 0], X_embedded[:N_train][idx0_train, 1],
            c='darkred', alpha=0.6, label='Class 0 (Train)')
idx1_train = (y_plot[:N_train] == 1)
plt.scatter(X_embedded[:N_train][idx1_train, 0], X_embedded[:N_train][idx1_train, 1],
            c='darkblue', alpha=0.6, label='Class 1 (Train)')

# Plot test data
idx0_test = (y_plot[N_train:N_train + N_test] == 0)
plt.scatter(X_embedded[N_train:N_train + N_test][idx0_test, 0],
            X_embedded[N_train:N_train + N_test][idx0_test, 1],
            c='orange', alpha=0.6, label='Class 0 (Test)')
idx1_test = (y_plot[N_train:N_train + N_test] == 1)
plt.scatter(X_embedded[N_train:N_train + N_test][idx1_test, 0],
            X_embedded[N_train:N_train + N_test][idx1_test, 1],
            c='skyblue', alpha=0.6, label='Class 1 (Test)')

# Plot synthetic data
idx_syn = (y_plot[N_train + N_test:] == 2)
plt.scatter(X_embedded[N_train + N_test:][idx_syn, 0],
            X_embedded[N_train + N_test:][idx_syn, 1],
            c='green', alpha=0.6, label='Synthetic')

plt.title("T-SNE Visualization: Train, Test, and Synthetic Data")
plt.legend()
plt.show()

# Step 4.5: Train Transformer Detector model on augmented dataset
train_dataset_final = TensorDataset(D_train, y_train)
test_dataset = TensorDataset(D_test, y_test)
train_loader_final = DataLoader(train_dataset_final, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print("After oversampling with Beta-CVAE:")
unique, counts = np.unique(y_train.numpy(), return_counts=True)
print("Class distribution in training set:", dict(zip(unique, counts)))


# Initialize and train the MoE Transformer Detector model
# model = MixtureOfExperts(input_size=input_dim, num_experts=10).to(device)
model = TransformerDetector(input_size=input_dim).to(device)
optimizer_tf = Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
num_epochs_tf = 100
for epoch in range(num_epochs_tf):
    train_loss = train_detector(model, train_loader_final, optimizer_tf, criterion, device)
    print(f"[Transformer] Epoch {epoch + 1}/{num_epochs_tf}, Loss={train_loss:.4f}")

    # Evaluate the model on the test set
    print("Test set evaluation:")
    evaluate_with_classification_report_and_auc(model, test_loader, device, threshold=0.3)
    print("-" * 40)
