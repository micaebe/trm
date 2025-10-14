import torch
from torch.utils.data import random_split


def create_train_test_datasets(full_dataset, train_ratio=0.8, random_seed=42):
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    return train_dataset, test_dataset