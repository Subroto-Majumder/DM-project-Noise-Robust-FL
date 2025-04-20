# --- START OF FILE data.py ---
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split # Added random_split
from torchvision.datasets import CIFAR10

symmetric_noise_val = 0.6
asymmetric_noise_val = 0.2

def add_label_noise(dataset, symmetric_noise_ratio=0.0, asymmetric_noise_ratio=0.0):
    # --- (Keep existing add_label_noise function) ---
    targets = np.array(dataset.targets)
    num_classes = 10
    num_samples = len(targets)
    indices = np.arange(num_samples)

    if symmetric_noise_ratio > 0:
        num_noisy = int(symmetric_noise_ratio * num_samples)
        noisy_indices = np.random.choice(indices, num_noisy, replace=False)
        for i in noisy_indices:
            new_label = np.random.choice([x for x in range(num_classes) if x != targets[i]])
            targets[i] = new_label

    if asymmetric_noise_ratio > 0:
        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            num_noisy = int(asymmetric_noise_ratio * len(class_indices))
            noisy_indices = np.random.choice(class_indices, num_noisy, replace=False)
            for i in noisy_indices:
                new_label = (targets[i] + 1) % num_classes
                targets[i] = new_label

    dataset.targets = targets.tolist()
    return dataset


def get_client_dataloader(cid, num_clients, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_dataset = CIFAR10(root="./data", train=train, download=True, transform=transform)
    # Apply noise only to the training data loader source if train=True
    if train:
        noisy_dataset = add_label_noise(full_dataset, symmetric_noise_ratio=symmetric_noise_val, asymmetric_noise_ratio=asymmetric_noise_val)
    else:
        noisy_dataset = full_dataset # Test data remains clean unless specifically requested

    num_samples = len(noisy_dataset) // num_clients
    indices = list(range(cid * num_samples, (cid + 1) * num_samples))
    # Ensure indices are within bounds - crucial if dataset size isn't perfectly divisible
    indices = [i for i in indices if i < len(noisy_dataset)]
    if not indices: # Handle case where cid is too high for dataset size
         print(f"Warning: No data for client {cid} with num_clients {num_clients}. Returning empty DataLoader.")
         return DataLoader([], batch_size=batch_size)

    subset = Subset(noisy_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=train)

# --- New Function ---
def get_server_validation_subset(dataset, subset_size=500, seed=42):
    """Creates a fixed, smaller subset of a dataset for server validation."""
    if subset_size >= len(dataset):
        return dataset # Use the whole dataset if requested size is too large
    # Ensure reproducibility
    generator = torch.Generator().manual_seed(seed)
    # Split dataset randomly but deterministically
    validation_subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size], generator=generator)
    return validation_subset

# --- END OF FILE data.py ---