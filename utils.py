import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from model import * 
import torch.autograd as autograd

# =========================
# 1. Load & Utility Functions
# =========================

def load_adbench_data(dataset_path):
    """
    Load dataset from a .npz file.
    Assumes the file contains:
    - 'X': Feature matrix (N, d)
    - 'y': Labels (N,)
    """
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def evaluate_with_classification_report_and_auc(model, test_loader, device, threshold=0.5):
    """
    Evaluate a model using classification report and AUC-ROC metric.
    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for test data.
        device: Computation device (CPU/GPU).
        threshold: Threshold for binary classification.
    Returns:
        Classification report and AUC-ROC score.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()  # Predicted scores (B,)
            all_preds.append(y_pred.cpu())
            all_labels.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy()  # Flatten predictions
    labels = torch.cat(all_labels).numpy()  # Flatten labels

    # Convert predictions to binary labels
    binary_preds = (preds > threshold).astype(int)

    # Generate classification report
    report = classification_report(labels, binary_preds, target_names=['Class 0', 'Class 1'])
    print(report)

    # Calculate AUC-ROC if both classes are present
    if len(set(labels)) > 1:
        aucroc = roc_auc_score(labels, preds)
        print(f"AUC-ROC: {aucroc:.4f}")
    else:
        aucroc = None
        print("AUC-ROC: Undefined (only one class present in labels)")

    return report, aucroc

def log_to_file(file_path, message):
    """
    Append a log message to the specified file.
    Args:
        file_path: Path to the log file.
        message: Message to log.
    """
    with open(file_path, "a") as file:
        file.write(message + "\n")

def beta_cvae_loss_fn(x, x_recon, mean, logvar, beta=4.0):
    """
    Compute Beta-CVAE loss (Reconstruction + Beta * KL Divergence).
    Args:
        x: Original input data.
        x_recon: Reconstructed data.
        mean: Mean of latent space distribution.
        logvar: Log variance of latent space distribution.
        beta: Weight for KL divergence.
    Returns:
        Total loss (scalar).
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP."""
    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_conditional_wasserstein_gan(generator, discriminator, data_loader, optimizer_D, optimizer_G, device):
    batch_size = 64
    latent_dim = 64
    n_critic = 5
    lambda_gp = 10

    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    discriminator.train()
    generator.train()
    total_loss = 0
    for i, (data, labels) in enumerate(data_loader):
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
        
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        total_loss+=d_loss
        optimizer_D.step()

        # Train Generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            fake_data = generator(z, labels)
            fake_validity = discriminator(fake_data, labels)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()
     
           
    return total_loss / len(data_loader)


def train_beta_cvae(model, data_loader, optimizer, device):
    """
    Train Beta-CVAE model for one epoch.
    Args:
        model: Beta-CVAE model to train.
        data_loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        device: Computation device (CPU/GPU).
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # Reshape labels (B, 1)

        # Forward pass
        x_recon, mean, logvar = model(x_batch, y_batch)
        loss = beta_cvae_loss_fn(x_batch, x_recon, mean, logvar, beta=model.beta)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def train_detector(model, train_loader, optimizer, criterion, device):
    """
    Train a detector model for one epoch.
    Args:
        model: Detector model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        criterion: Loss function (e.g., BCE Loss).
        device: Computation device (CPU/GPU).
    Returns:
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


# Để tái lập trình ngẫu nhiên cho ví dụ
torch.manual_seed(0)
np.random.seed(0)

# Giả sử ta có một hàm tính reward liên quan đến "độ đa dạng" (diversity)
# Ở đây, tạm thời ta giả lập bằng cách random ra reward để minh hoạ.
def compute_diversity_reward(modified_z):
    # Tùy chỉnh cách tính reward thực tế.
    # Ở đây minh hoạ: reward tỉ lệ với độ lớn L2 norm của z (giả sử).
    return torch.norm(modified_z, p=2, dim=-1, keepdim=True)

# Hàm tiện ích chuyển numpy -> torch
def to_tensor(x, device="cpu", dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)

def One_Step_To_Feasible_Action(
        generator,
        discriminator,
        detector,
        x_orig,
        device,
        previously_generated=None,
        alpha=1.0,
        lambda_div=0.1,
        lr=0.001,
        steps=50,
        log_file=None
):
    """
    Generate adversarial samples by modifying latent space representation.
    Args:
        beta_cvae: Trained Beta-CVAE model.
        detector: Trained detector model.
        x_orig: Original input data.
        device: Computation device (CPU/GPU).
        previously_generated: List of previously generated samples (for diversity).
        alpha: Scaling factor for diversity term.
        lambda_div: Weight for diversity term.
        lr: Learning rate for optimization.
        steps: Number of optimization steps.
        log_file: Path to log file for recording progress.
    Returns:
        Adversarial sample (torch.Tensor).
    """
    generator.eval()
    discriminator.eval()
    detector.eval()

    if previously_generated is None:
        previously_generated = []

    x_orig = x_orig.to(device).unsqueeze(0)  # Reshape to batch format (1, d)
    y_class1 = torch.full((1, 1), 0.8, device=device)  # Target class label (e.g., 0.8)

    # Encode input data into latent space
    with torch.no_grad():
       z_uniform = (torch.rand(1, 64) * 4.0) - 2.0
       z_uniform = z_uniform.to(device)

    # Optimize latent space representation
    optimizer_z = torch.optim.Adam([z_uniform], lr=lr)
    for step in range(steps):
        optimizer_z.zero_grad()

        # Decode latent variable back to data space
        x_synthetic = generator(z_uniform, y_class1)

        # Calculate detector prediction
        prob_class1 = detector(x_synthetic)

        # Diversity term (if previous samples exist)
        if previously_generated:
            x_old_cat = torch.stack(previously_generated, dim=0).to(device)  # Stack previous samples (N, d)
            dist = torch.norm(x_synthetic - x_old_cat, p=2, dim=1)  # Pairwise distances
            diversity_term = torch.exp(-alpha * dist).sum()
        else:
            diversity_term = 0.0

        # Calculate total reward (inverse objective)
        inv_reward = prob_class1.mean() + lambda_div * diversity_term
        inv_reward.backward()
        optimizer_z.step()

    print(f"Deceiving Detector Reward: {1/ (prob_class1.item()+0.0001):.4f}",
          f"Diversity reward: {1/(diversity_term+0.0001):.4f}",
          f"Sample reward: {1/(inv_reward.item()+0.0001):.4f}")

    # Decode optimized latent variable back to data space
    with torch.no_grad():
        x_adv = generator(z_uniform, y_class1).detach().cpu().squeeze(0)
    return x_adv
