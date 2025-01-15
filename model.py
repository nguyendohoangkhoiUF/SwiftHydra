import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =========================
# 2. Beta-CVAE
# =========================
class BetaCVAE(nn.Module):
    """
    Beta Conditional Variational Autoencoder (Beta-CVAE) for binary tasks:
      - Encoder processes (x, y).
      - Decoder processes (z, y).
      - Beta > 1 emphasizes the KL divergence to encourage a more spread-out latent space.
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, beta=4.0):
        super(BetaCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # Scaling factor for KL divergence

        # Encoder layers
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)  # Concatenate y -> +1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Additional hidden layer
        self.fc3_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim + 1, hidden_dim)  # Concatenate y -> +1
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)  # Additional hidden layer
        self.fc6 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        """Encode input (x, y) into latent space with mean and log variance."""
        xy = torch.cat([x, y], dim=1)  # Concatenate features and label (B, input_dim+1)
        h = F.relu(self.fc1(xy))
        h = F.relu(self.fc2(h))  # Additional hidden layer
        mean = self.fc3_mean(h)
        logvar = self.fc3_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Sample from latent space using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, y):
        """Decode latent representation (z, y) back to input space."""
        zy = torch.cat([z, y], dim=1)  # Concatenate latent vector and label (B, latent_dim+1)
        h = F.relu(self.fc4(zy))
        h = F.relu(self.fc5(h))  # Additional hidden layer
        x_recon = self.fc6(h)
        return x_recon

    def forward(self, x, y):
        """Forward pass through the Beta-CVAE."""
        mean, logvar = self.encode(x, y)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mean, logvar


# =========================
# 3. Transformer Detector
# =========================
class PositionalEncoding(nn.Module):
    """
    Add positional encoding to the input for sequential modeling.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        """Add positional encoding to input tensor."""
        L = x.size(1)  # Sequence length
        return x + self.pe[:, :L, :].to(x.device)


class TransformerDetector(nn.Module):
    """
    Transformer-based detector model for anomaly or binary classification tasks.
    """

    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerDetector, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)  # Input embedding layer
        self.positional_encoding = PositionalEncoding(d_model)  # Positional encoding

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability for binary classification
        )

    def forward(self, x):
        """Forward pass through the Transformer Detector."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension (B, 1, input_size)
        x = self.embedding(x)  # Project input to d_model dimensions
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Transformer encoder
        x = x.mean(dim=1)  # Aggregate features by averaging over sequence dimension
        return self.fc(x).squeeze(1)  # Output probabilities

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) architecture where each expert is a Transformer model.
    """
    def __init__(self, input_size, num_experts, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.1, gating_hidden_size=64):
        super(MixtureOfExperts, self).__init__()

        self.num_experts = num_experts

        # Define experts (Transformer models)
        self.experts = nn.ModuleList([
            TransformerDetector(input_size=input_size,
                                d_model=d_model,
                                nhead=nhead,
                                num_layers=num_layers,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout)
            for _ in range(num_experts)
        ])

        # Gating network to decide expert weights
        self.gating_network = nn.Sequential(
            nn.Linear(input_size, gating_hidden_size),
            nn.ReLU(),
            nn.Linear(gating_hidden_size, num_experts),
            nn.Softmax(dim=-1)  # Output weights for each expert
        )

    def forward(self, x):
        """
        Forward pass through Mixture of Experts.
        Args:
            x: Input tensor of shape (batch_size, input_size).
        Returns:
            Output tensor of shape (batch_size).
        """
        # Compute gating weights
        gating_weights = self.gating_network(x)  # (batch_size, num_experts)

        # Collect outputs from experts
        expert_outputs = torch.cat([expert(x).unsqueeze(1) for expert in self.experts], dim=1)  # (batch_size, num_experts)

        # Combine expert outputs based on gating weights
        output = torch.sum(expert_outputs * gating_weights, dim=1)  # (batch_size)

        return output


import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba


class MambaNet(nn.Module):
    def __init__(self, input_size):
        super(MambaNet, self).__init__()

        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_size,  # Model dimension d_model
            d_state=128,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(256, 256)
        # self.dropout2 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(256, 256)
        # self.dropout3 = nn.Dropout(0.2)
        # self.fc4 = nn.Linear(256, 256)
        # self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mamba(x.unsqueeze(1)).squeeze(1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        # x = self.leaky_relu(self.fc2(x))
        # x = self.dropout2(x)
        # x = self.leaky_relu(self.fc3(x))
        # x = self.dropout3(x)
        # x = self.leaky_relu(self.fc4(x))
        # x = self.dropout4(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
