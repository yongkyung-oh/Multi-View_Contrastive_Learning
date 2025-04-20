import torch
import torch.fft as fft
import torchcde
from torch.utils.data import Dataset
from typing import Tuple, Optional


def normalize(X_train: torch.Tensor, X_test: torch.Tensor, epsilon: float = 1e-8):
    # Compute mean and std along the N and L dimensions
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True)
    
    # Add epsilon to std to avoid division by zero
    std = std.clamp(min=epsilon)
    
    # Normalize train and test data
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm, mean, std  # Return mean and std for potential inverse transform


def add_time_feature(X: torch.Tensor):
    # X: [num_samples, sequence_length, num_features]
    num_samples, seq_length, _ = X.shape
    
    # Create a time index vector normalized between 0 and 1
    time_index = torch.linspace(0, 1, steps=seq_length).to(X.device)  # Shape: [sequence_length]
    
    # Expand time index to match the batch size
    time_feature = time_index.unsqueeze(0).unsqueeze(-1).repeat(num_samples, 1, 1)  # Shape: [num_samples, sequence_length, 1]
    
    # Concatenate the time feature to the original data
    X_with_time = torch.cat([time_feature, X], dim=-1)  # New shape: [num_samples, sequence_length, num_features + 1]
    return X_with_time


def get_dx(X: torch.Tensor) -> torch.Tensor:
    N, L, D = X.shape
    t = torch.linspace(0, 1, L, dtype=X.dtype, device=X.device)
    
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
    spline = torchcde.CubicSpline(coeffs, t)
    dx = spline.derivative(t)
    return dx


def get_xf(X: torch.Tensor) -> torch.Tensor:
    return torch.abs(fft.fft(X, dim=1))  # Use dim=1 for the sequence dimension


def preprocess_data(X_train, X_test, time_as_feature=False):
    # Normalize time domain data
    X_train_xt, X_test_xt, mean_xt, std_xt = normalize(X_train, X_test)

    # Compute and normalize derivative
    X_train_dx = get_dx(X_train_xt)
    X_test_dx = get_dx(X_test_xt)
    X_train_dx, X_test_dx, mean_dx, std_dx = normalize(X_train_dx, X_test_dx)
    
    # Compute Fourier transforms
    X_train_xf = get_xf(X_train_xt)
    X_test_xf = get_xf(X_test_xt)
    X_train_xf, X_test_xf, mean_xf, std_xf = normalize(X_train_xf, X_test_xf)

    # Add time as a feature
    if time_as_feature:
        X_train_xt, X_test_xt = add_time_feature(X_train_xt), add_time_feature(X_test_xt)
        X_train_dx, X_test_dx = add_time_feature(X_train_dx), add_time_feature(X_test_dx)
        X_train_xf, X_test_xf = add_time_feature(X_train_xf), add_time_feature(X_test_xf)
    
    return {
        'xt': (X_train_xt.float(), X_test_xt.float(), mean_xt, std_xt),
        'dx': (X_train_dx.float(), X_test_dx.float(), mean_dx, std_dx),
        'xf': (X_train_xf.float(), X_test_xf.float(), mean_xf, std_xf)
    }


class Load_Dataset(Dataset):
    def __init__(self, X: list, X_aug: list, y: torch.Tensor, 
                 mode: str, num_repeats: int = 1):
        super(Load_Dataset, self).__init__()
        
        self.mode = mode
        self.num_repeats = num_repeats
        
        if self.mode == 'pretrain':
            self.setup_pretrain_data(X, X_aug, y)
        else:
            self.setup_finetune_data(X, y)

    def setup_pretrain_data(self, X: list, X_aug: list, y: torch.Tensor):
        self.xt, self.dx, self.xf = X
        self.xt, self.dx, self.xf = self.get_repeats(self.xt), self.get_repeats(self.dx), self.get_repeats(self.xf) 
        self.xt_aug, self.dx_aug, self.xf_aug = X_aug
        self.y = y.long().unsqueeze(-1).repeat(1, self.num_repeats).reshape(-1)

    def setup_finetune_data(self, X: torch.Tensor, y: torch.Tensor):
        self.xt, self.dx, self.xf = X
        self.xt_aug, self.dx_aug, self.xf_aug = X
        self.y = y.long().reshape(-1)

    def get_repeats(self, X: torch.Tensor, num_repeats: int = 10):
        X = X.float().unsqueeze(-1).repeat(1, 1, 1, self.num_repeats)
        return X.permute(0, 3, 1, 2).reshape(-1, X.shape[1], X.shape[2])

    def __len__(self) -> int:
        return self.xt.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.mode == 'pretrain':
            self.xt_aug_noisy = self.data_transform_td(self.xt_aug[idx,...])
            self.dx_aug_noisy = self.data_transform_td(self.dx_aug[idx,...])
            self.xf_aug_noisy = self.data_transform_fd(self.xf_aug[idx,...])
            return (self.xt_aug[idx], self.dx_aug[idx], self.xf_aug[idx],
                    self.xt_aug_noisy, self.dx_aug_noisy, self.xf_aug_noisy, self.y[idx])
        else:
            self.xt_noisy = self.data_transform_td(self.xt[idx,...])
            self.dx_noisy = self.data_transform_td(self.dx[idx,...])
            self.xf_noisy = self.data_transform_fd(self.xf[idx,...])
            return (self.xt[idx], self.dx[idx], self.xf[idx],
                    self.xt_noisy, self.dx_noisy, self.xf_noisy, self.y[idx])

    @staticmethod
    def data_transform_td(sample: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        return sample + torch.normal(mean=0., std=sigma, size=sample.shape, device=sample.device)

    @staticmethod
    def data_transform_fd(sample: torch.Tensor, pertub_ratio: float = 0.05) -> torch.Tensor:
        aug_1 = Load_Dataset.remove_frequency(sample, pertub_ratio)
        aug_2 = Load_Dataset.add_frequency(sample, pertub_ratio)
        return aug_1 + aug_2

    @staticmethod
    def remove_frequency(x: torch.Tensor, pertub_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > pertub_ratio
        return x * mask

    @staticmethod
    def add_frequency(x: torch.Tensor, pertub_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > (1 - pertub_ratio)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape, device=x.device) * (max_amplitude * 0.1)
        pertub_matrix = mask * random_am
        return x + pertub_matrix

