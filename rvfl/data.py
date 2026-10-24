"""Datasets: loading, the client/server partition, and data poisoning."""
import copy
import gc
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def transform_for(model_name: str):
    if model_name == 'ResNet20':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    if 'ResNet' in model_name:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def load_dataset(dataset_name: str, train: bool, model_name: str, dataset_path: str = './dataset'):
    """The torchvision dataset with the model's input transform (downloaded on first use)."""
    transform = transform_for(model_name)
    path = Path(dataset_path)
    if dataset_name == 'MNIST':
        return datasets.MNIST(str(path), train=train, download=True, transform=transform)
    if dataset_name == 'CIFAR10':
        return datasets.CIFAR10(str(path), train=train, download=True, transform=transform)
    if dataset_name == 'ImageNet':
        return datasets.ImageNet(str(path / "ImageNet"), split='train' if train else 'val',
                                 transform=transform)
    raise ValueError(f"unsupported dataset {dataset_name!r}")


def _train_targets(dataset_name: str, dataset_path: str):
    if dataset_name == 'MNIST':
        return datasets.MNIST(dataset_path, train=True, download=True).targets
    if dataset_name == 'CIFAR10':
        return torch.tensor(datasets.CIFAR10(dataset_path, train=True, download=True).targets)
    raise ValueError(f"unsupported dataset {dataset_name!r}")


def partition_and_save(dataset_name: str,
                       num_clients: int,
                       save_path: str = './partitions',
                       is_iid: bool = True,
                       q: float = 0.5,
                       dataset_path: str = './dataset',
                       server_samples: int = 200,
                       seed: Optional[int] = None):
    """Split the training set into the server's root dataset (id -1, class-balanced) and the
    clients' local datasets (ids 0..num_clients-1), as in FLTrust.

    Non-IID (is_iid=False): the clients are put into one group per class; a sample of label l
    goes to group l with probability q and to a uniformly chosen other group otherwise, then to
    a random client of that group. IID uses q = 1 / num_classes.

    The split is drawn from numpy's global generator, seeded with `seed` first when it is
    given. The map is saved under save_path and reused when the same file already exists (the
    seed is part of its name). Returns (path, {id: indices}, mean client size / server size).
    """
    iid_status = "iid" if is_iid else "noniid"
    seed_tag = "" if seed is None else f"_s{seed}"
    full_file_path = Path(save_path) / (f"partitions_{dataset_name}_{num_clients}_{iid_status}_"
                                        f"{q}_{server_samples}{seed_tag}.pkl")

    if full_file_path.exists():
        with open(full_file_path, 'rb') as f:
            client_indices = pickle.load(f)
        avg_client_size = sum(len(client_indices.get(i, [])) for i in range(num_clients))
        ratio = (avg_client_size // num_clients) / len(client_indices.get(-1, []))
        return full_file_path, client_indices, ratio

    if seed is not None:
        np.random.seed(seed % (2 ** 32))
    targets = _train_targets(dataset_name, dataset_path)
    client_indices = {}

    if server_samples > 0:
        labels_all = targets.cpu().numpy()
        num_classes = len(np.unique(labels_all))
        samples_per_class = server_samples // num_classes
        remainder = server_samples % num_classes
        server_idx, remaining_idx_list = [], []
        for k in range(num_classes):
            idx_k = np.where(labels_all == k)[0]
            np.random.shuffle(idx_k)
            take = samples_per_class + (1 if k < remainder else 0)
            server_idx.extend(idx_k[:take])
            remaining_idx_list.extend(idx_k[take:])
        client_indices[-1] = [int(x) for x in server_idx]
        remaining_indices = np.array(remaining_idx_list)
        np.random.shuffle(remaining_indices)
    else:
        remaining_indices = np.arange(len(targets))
        np.random.shuffle(remaining_indices)

    labels = targets[remaining_indices].cpu().numpy()
    num_classes = len(np.unique(labels))
    if is_iid:
        q = 1 / num_classes

    num_virtual_clients = max(num_clients, num_classes)
    virtual_client_list = np.arange(num_virtual_clients)
    np.random.shuffle(virtual_client_list)
    client_groups = np.array_split(virtual_client_list, num_classes)
    virtual_indices = {i: [] for i in range(num_virtual_clients)}

    for pos, label in enumerate(labels):
        if np.random.rand() < q:
            group = label
        else:
            group = np.random.choice([g for g in range(num_classes) if g != label])
        virtual_indices[np.random.choice(client_groups[group])].append(int(remaining_indices[pos]))

    per_client = {i: [] for i in range(num_clients)}
    for v_id, indices in virtual_indices.items():
        per_client[v_id % num_clients].extend(indices)
    for i in range(num_clients):
        client_indices[i] = per_client[i]

    full_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_file_path, 'wb') as f:
        pickle.dump(client_indices, f)
    print(f"[data] partition saved to {full_file_path}: server {len(client_indices.get(-1, []))} "
          f"samples, clients {len(remaining_indices) // num_clients} on average")
    ratio = (len(remaining_indices) // num_clients) / len(client_indices.get(-1, []))
    return full_file_path, client_indices, ratio


class PoisoningConfig:
    """A client-side data-poisoning attack.

      'label_flip_fltrust'  FLTrust label flipping, l -> num_classes - 1 - l
      'backdoor'            stamp a trigger patch and relabel to target_label (Scaling attack)

    poison_rate is the fraction of the client's samples that are poisoned (drawn per sample
    from numpy's global generator).
    """

    def __init__(self,
                 poison_type: str = 'label_flip_fltrust',
                 target_label: Optional[int] = None,
                 poison_rate: float = 1.0,
                 num_classes: int = 10,
                 backdoor_pattern: Optional[Dict] = None):
        self.poison_type = poison_type
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.num_classes = num_classes
        if backdoor_pattern is None and poison_type == 'backdoor':
            self.backdoor_pattern = dict(BACKDOOR_PATTERN)
        else:
            self.backdoor_pattern = backdoor_pattern
        if poison_type == 'backdoor' and self.target_label is None:
            raise ValueError("poison_type='backdoor' requires target_label.")


# The Scaling attack's trigger: a 3x3 patch of value 1.0 (after normalization) in the
# bottom-right corner. pos is taken modulo the image size, so (-3, -3) is the corner.
BACKDOOR_PATTERN = {'pos': (-3, -3), 'size': (3, 3), 'color': 1.0}


def stamp_trigger(data: torch.Tensor, pattern=BACKDOOR_PATTERN) -> torch.Tensor:
    """Stamp the trigger on one image (C, H, W) or a batch (B, C, H, W)."""
    pos_x, pos_y = pattern['pos']
    size_w, size_h = pattern['size']
    H, W = data.shape[-2], data.shape[-1]
    py, px = pos_y % H, pos_x % W
    out = data.clone()
    out[..., py:py + size_h, px:px + size_w] = pattern['color']
    return out


class PoisonedDataset(Dataset):
    def __init__(self, dataset: Dataset, config: PoisoningConfig):
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        target = int(target)
        if np.random.random() < self.config.poison_rate:
            if self.config.poison_type == 'label_flip_fltrust':
                target = self.config.num_classes - 1 - target
            elif self.config.poison_type == 'backdoor':
                if self.config.backdoor_pattern is not None:
                    data = stamp_trigger(data, self.config.backdoor_pattern)
                target = self.config.target_label
        return data, torch.tensor(target, dtype=torch.long)


class ClientDataLoader:
    """The local dataset of one party (client id, or -1 for the server's root dataset)."""

    def __init__(self,
                 dataset_name: str,
                 model_name: str,
                 client_id: int,
                 partition_path: str,
                 dataset_path: str = './dataset',
                 batch_size: int = 32,
                 poisoning_config: Optional[PoisoningConfig] = None):
        self.dataset_name = dataset_name
        self.client_id = client_id
        self.batch_size = batch_size
        self.poisoning_config = poisoning_config

        full_dataset = load_dataset(dataset_name, True, model_name, dataset_path)
        with open(partition_path, 'rb') as f:
            all_indices = pickle.load(f)
        if client_id not in all_indices:
            raise ValueError(f"Client ID {client_id} not found in partition file.")
        self.local_dataset = self._extract_local_data(full_dataset, all_indices[client_id])
        del full_dataset
        gc.collect()

        dataset = self.local_dataset
        if poisoning_config:
            dataset = PoisonedDataset(dataset, poisoning_config)
        self._loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @staticmethod
    def _extract_local_data(full_dataset: Dataset, indices: List[int]) -> Dataset:
        local = copy.copy(full_dataset)
        if hasattr(local, 'data') and hasattr(local, 'targets'):
            if isinstance(local.data, (torch.Tensor, np.ndarray)):
                local.data = local.data[indices]
            else:
                local.data = [local.data[i] for i in indices]
            if isinstance(local.targets, (torch.Tensor, np.ndarray)):
                local.targets = local.targets[indices]
            else:
                local.targets = [local.targets[i] for i in indices]
        elif hasattr(local, 'samples') and hasattr(local, 'targets'):
            local.samples = [local.samples[i] for i in indices]
            local.targets = [local.targets[i] for i in indices]
            if hasattr(local, 'imgs'):
                local.imgs = local.samples
        else:
            raise NotImplementedError("Unsupported dataset structure.")
        return local

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)
