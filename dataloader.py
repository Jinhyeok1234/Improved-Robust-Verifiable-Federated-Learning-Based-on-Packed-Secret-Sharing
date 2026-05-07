from pathlib import Path
import numpy as np
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from typing import List, Dict, Optional, Tuple
import copy
import gc

class PoisoningConfig:
    """
    dummy class for poisoning attack
    """
    def __init__(self, 
                 poison_type: str = 'label_flip', 
                 target_label: Optional[int] = None,
                 source_labels: Optional[List[int]] = None,
                 poison_rate: float = 1.0,
                 backdoor_pattern: Optional[Dict] = None):
        self.poison_type = poison_type
        self.target_label = target_label
        self.source_labels = source_labels
        self.poison_rate = poison_rate
        
        if backdoor_pattern is None and poison_type == 'backdoor':
            self.backdoor_pattern = {'pos': (-3, -3), 'size': (3, 3), 'color': 1.0}
        else:
            self.backdoor_pattern = backdoor_pattern

        if self.target_label is None:
            raise ValueError("공격을 위해서는 target_label 설정이 필수입니다.")

class PoisonedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int], config: PoisoningConfig):
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return len(self.indices)

    def _apply_backdoor(self, data: torch.Tensor) -> torch.Tensor:
        pattern = self.config.backdoor_pattern
        if pattern is None: return data
        
        pos_x, pos_y = pattern['pos']
        size_w, size_h = pattern['size']
        color = pattern['color']
        
        poisoned_data = data.clone()
        poisoned_data[:, pos_y:pos_y+size_h, pos_x:pos_x+size_w] = color
        return poisoned_data

    def __getitem__(self, index):
        data, target = self.dataset[index]

        if np.random.random() < self.config.poison_rate:
            
            if self.config.poison_type == 'label_flip':
                if self.config.source_labels is None or target in self.config.source_labels:
                    target = self.config.target_label

            elif self.config.poison_type == 'backdoor':
                data = self._apply_backdoor(data)
                target = self.config.target_label

        if not isinstance(target, torch.Tensor):
             target = torch.tensor(target, dtype=torch.long)
             
        return data, target

class CustomDataLoader:
    def __init__(self, 
                 dataset_name: str,
                 model_name: str,
                 client_id: int, 
                 partition_path: str,
                 dataset_path: str = './dataset',
                 batch_size: int = 32,
                 poisoning_config: Optional['PoisoningConfig'] = None):
        
        self.dataset_name = dataset_name
        self.client_id = client_id
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.poisoning_config = poisoning_config
        
        full_dataset = self._load_raw_dataset(model_name)
        
        with open(partition_path, 'rb') as f:
            all_indices = pickle.load(f)
        
        if client_id not in all_indices:
            raise ValueError(f"Client ID {client_id} not found in partition file.")
            
        my_indices = all_indices[client_id]
        self.local_dataset = self._extract_local_data(full_dataset, my_indices)
        del full_dataset
        gc.collect()

        self._cached_loader = self._create_loader()

    def _extract_local_data(self, full_dataset: Dataset, indices: List[int]) -> Dataset:
        local_dataset = copy.copy(full_dataset)
        
        if hasattr(local_dataset, 'data') and hasattr(local_dataset, 'targets'):
            if isinstance(local_dataset.data, (torch.Tensor, np.ndarray)):
                local_dataset.data = local_dataset.data[indices]
            else:
                local_dataset.data = [local_dataset.data[i] for i in indices]
                
            if isinstance(local_dataset.targets, (torch.Tensor, np.ndarray)):
                local_dataset.targets = local_dataset.targets[indices]
            else:
                local_dataset.targets = [local_dataset.targets[i] for i in indices]
                
        elif hasattr(local_dataset, 'samples') and hasattr(local_dataset, 'targets'):
            local_dataset.samples = [local_dataset.samples[i] for i in indices]
            local_dataset.targets = [local_dataset.targets[i] for i in indices]
            if hasattr(local_dataset, 'imgs'):
                local_dataset.imgs = local_dataset.samples
        else:
            raise NotImplementedError("Unsupported dataset structure.")
            
        return local_dataset

    def _create_loader(self):
        if self.poisoning_config:
            dataset = PoisonedDataset(self.local_dataset, self.poisoning_config)
        else:
            dataset = self.local_dataset
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _load_raw_dataset(self, model_name):
        if model_name == 'ResNet20':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif 'ResNet' in model_name:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        dataset_path = Path(self.dataset_path)        
        if self.dataset_name == 'MNIST':
            return datasets.MNIST(str(dataset_path), train=True, download=True, transform=transform)
        elif self.dataset_name == 'CIFAR10':
            return datasets.CIFAR10(str(dataset_path), train=True, download=True, transform=transform)
        elif self.dataset_name == 'ImageNet':
            dataset_path = dataset_path / "ImageNet"
            return datasets.ImageNet(str(dataset_path), split='train', transform=transform)
        # ... additional datasets
        # for image classification: CIFAR100, FashionMNIST
        
    def get_loader(self):
        return self._cached_loader
    
    def __iter__(self):
        return iter(self._cached_loader)

    def __len__(self):
        return len(self._cached_loader)