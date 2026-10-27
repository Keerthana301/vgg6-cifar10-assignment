"""CIFAR-10 data loading with a fixed train/validation split."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class Cutout:
    """Mask one randomly positioned square on a tensor image."""
    def __init__(self, length=16):
        self.length = length

    def __call__(self, image):
        _, height, width = image.shape
        cy = random.randrange(height)
        cx = random.randrange(width)
        y0, y1 = max(0, cy - self.length // 2), min(height, cy + self.length // 2)
        x0, x1 = max(0, cx - self.length // 2), min(width, cx + self.length // 2)
        image = image.clone()
        image[:, y0:y1, x0:x1] = 0
        return image


class IndexedDataset(Dataset):
    """Subset that applies its own transform, independent of other subsets."""
    def __init__(self, raw_dataset, indices, transform):
        self.raw_dataset = raw_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.raw_dataset[self.indices[i]]
        return self.transform(image), label


def make_loaders(batch_size=128, seed=42, root='./data', workers=2,
                 val_size=5000, use_cutout=True):
    if not 0 < val_size < 50000:
        raise ValueError('val_size must be between 1 and 49999')
    augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
               transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if use_cutout:
        augment.append(Cutout(16))
    train_transform = transforms.Compose(augment)
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    raw_train = datasets.CIFAR10(root, train=True, download=True)
    raw_test = datasets.CIFAR10(root, train=False, download=True)
    indices = np.random.default_rng(seed).permutation(len(raw_train))
    val_indices, train_indices = indices[:val_size], indices[val_size:]
    train_ds = IndexedDataset(raw_train, train_indices, train_transform)
    val_ds = IndexedDataset(raw_train, val_indices, eval_transform)
    test_ds = IndexedDataset(raw_test, range(len(raw_test)), eval_transform)

    def worker_init(worker_id):
        s = torch.initial_seed() % (2**32)
        random.seed(s)
        np.random.seed(s)

    generator = torch.Generator().manual_seed(seed)
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(),
                  worker_init_fn=worker_init)
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       generator=generator, **common)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train, val, test
