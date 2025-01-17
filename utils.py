import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

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
        beta_cvae,
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
    beta_cvae.eval()
    detector.eval()

    if previously_generated is None:
        previously_generated = []

    x_orig = x_orig.to(device).unsqueeze(0)  # Reshape to batch format (1, d)
    y_class1 = torch.full((1, 1), 0.8, device=device)  # Target class label (e.g., 0.8)

    # Encode input data into latent space
    with torch.no_grad():
        mean, logvar = beta_cvae.encode(x_orig, y_class1)
        z = beta_cvae.reparameterize(mean, logvar).detach().clone()

    # Optimize latent space representation
    optimizer_z = torch.optim.Adam([z], lr=lr)
    for step in range(steps):
        optimizer_z.zero_grad()

        # Decode latent variable back to data space
        x_synthetic = beta_cvae.decode(z, y_class1)

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
        x_adv = beta_cvae.decode(z, y_class1).detach().cpu().squeeze(0)
    return x_adv
