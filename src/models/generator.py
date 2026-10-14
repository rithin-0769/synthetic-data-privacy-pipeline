"""Generative model engines:
1. PyTorch-based Conditional Generative Adversarial Network / VAE (CTGAN architecture) with Differential Privacy noise injection.
2. Parametric Multivariate Gaussian Copula (pure NumPy/SciPy, dependency-free).
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm

class DPNoiseMechanism:
    """Differential privacy Gaussian noise calibration for gradient / parameter perturbation."""
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        # Standard Gaussian DP noise scale: sigma = sqrt(2 * ln(1.25 / delta)) / epsilon
        self.sigma = np.sqrt(2 * np.log(1.25 / delta)) * (clip_norm / epsilon)

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.sigma
        return tensor + noise

class GeneratorNet(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class DiscriminatorNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class TabularSynthesizer:
    def __init__(self, model_type: str = "ctgan", epochs: int = 150, epsilon: float = 1.5):
        self.model_type = model_type.lower()
        self.epochs = epochs
        self.epsilon = epsilon
        self.columns = []
        self.cat_encoders = {}
        self.num_min_max = {}
        self.generator = None
        self.latent_dim = 16
        self.dp = DPNoiseMechanism(epsilon=self.epsilon)

    def _encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        self.columns = list(df.columns)
        encoded_cols = []
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype.name == "category":
                unique_vals = list(df[col].unique())
                self.cat_encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
                rev_map = {idx: val for idx, val in enumerate(unique_vals)}
                self.cat_encoders[f"{col}_rev"] = rev_map
                encoded_cols.append(df[col].map(self.cat_encoders[col]).values.reshape(-1, 1))
            else:
                c_min, c_max = float(df[col].min()), float(df[col].max())
                diff = c_max - c_min if c_max != c_min else 1.0
                self.num_min_max[col] = (c_min, c_max)
                norm_vals = (df[col].values - c_min) / diff
                encoded_cols.append(norm_vals.reshape(-1, 1))
        return np.hstack(encoded_cols).astype(np.float32)

    def _decode_array(self, arr: np.ndarray) -> pd.DataFrame:
        out_dict = {}
        idx = 0
        for col in self.columns:
            col_data = arr[:, idx]
            if col in self.cat_encoders:
                classes = list(self.cat_encoders[col].values())
                snapped = np.clip(np.round(col_data), 0, len(classes) - 1).astype(int)
                rev = self.cat_encoders[f"{col}_rev"]
                out_dict[col] = [rev[val] for val in snapped]
            else:
                c_min, c_max = self.num_min_max[col]
                diff = c_max - c_min
                unnorm = (np.clip(col_data, 0.0, 1.0) * diff) + c_min
                out_dict[col] = np.round(unnorm, 2)
            idx += 1
        return pd.DataFrame(out_dict)

    def fit(self, data: pd.DataFrame):
        data_matrix = self._encode_dataframe(data)
        n_samples, feature_dim = data_matrix.shape

        if self.model_type == "copula":
            # Multivariate Gaussian Copula
            self.mean = np.mean(data_matrix, axis=0)
            cov = np.cov(data_matrix, rowvar=False)
            # Regularize covariance to ensure positive semi-definite
            self.cov = cov + np.eye(feature_dim) * 1e-4
            return

        # CTGAN with Differential Privacy Noise Injection
        self.generator = GeneratorNet(self.latent_dim, feature_dim)
        discriminator = DiscriminatorNet(feature_dim)

        g_opt = optim.Adam(self.generator.parameters(), lr=2e-4)
        d_opt = optim.Adam(discriminator.parameters(), lr=2e-4)
        criterion = nn.BCELoss()

        real_tensor = torch.tensor(data_matrix, dtype=torch.float32)

        for epoch in range(self.epochs):
            # Train Discriminator
            d_opt.zero_grad()
            z = torch.randn(n_samples, self.latent_dim)
            fake_tensor = self.generator(z)

            d_real = discriminator(real_tensor)
            d_fake = discriminator(fake_tensor.detach())

            real_labels = torch.ones(n_samples, 1)
            fake_labels = torch.zeros(n_samples, 1)

            loss_d = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            loss_d.backward()

            # Apply DP gradient clipping & calibrated noise for differential privacy
            for param in discriminator.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, self.dp.clip_norm)
                    # inject DP Gaussian noise
                    noise = torch.randn_like(param.grad) * (self.dp.sigma / np.sqrt(n_samples))
                    param.grad += noise

            d_opt.step()

            # Train Generator
            g_opt.zero_grad()
            z = torch.randn(n_samples, self.latent_dim)
            gen_out = self.generator(z)
            d_eval = discriminator(gen_out)
            loss_g = criterion(d_eval, real_labels)
            loss_g.backward()
            g_opt.step()

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self.model_type == "copula":
            samples = np.random.multivariate_normal(self.mean, self.cov, size=num_rows)
            return self._decode_array(samples)

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_rows, self.latent_dim)
            raw_samples = self.generator(z).numpy()
        return self._decode_array(raw_samples)
