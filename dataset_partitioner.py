import numpy as np
import pickle
import torch
from torchvision import datasets
from pathlib import Path
import collections

def partition_and_save(dataset_name: str, 
                       num_clients: int,  
                       save_path: str = './partitions', 
                       is_iid: bool = True, 
                       q: float = 0.5, 
                       dataset_path: str = './dataset',
                       server_samples: int = 200):
    base_dir = Path(save_path)
    iid_status = "iid" if is_iid else "noniid"
    file_name = f"partitions_{dataset_name}_{num_clients}_{iid_status}_{q}_{server_samples}.pkl"
    full_file_path = base_dir / file_name

    if full_file_path.exists():
        print(f"Loading existing partition map from {full_file_path}...")
        with open(full_file_path, 'rb') as f:
            client_indices = pickle.load(f)
        avg_client_size = 0
        for i in range(num_clients):
            avg_client_size += len(client_indices.get(i, []))
        server_client_ratio = (avg_client_size//num_clients) / len(client_indices.get(-1, []))
        return full_file_path, client_indices, server_client_ratio

    if dataset_name == 'MNIST':
        train_d = datasets.MNIST(dataset_path, train=True, download=True)
        targets = train_d.targets
    elif dataset_name == 'CIFAR10':
        train_d = datasets.CIFAR10(dataset_path, train=True, download=True)
        targets = torch.tensor(train_d.targets)
    else:
        raise ValueError("Unsupported dataset")

    num_samples = len(targets)
    client_indices = {}

    if server_samples > 0:
        labels_all = targets.cpu().numpy()
        num_classes = len(np.unique(labels_all))
        
        samples_per_class = server_samples // num_classes
        remainder = server_samples % num_classes
        
        server_idx = []
        remaining_idx_list = []
        
        for k in range(num_classes):
            idx_k = np.where(labels_all == k)[0]
            np.random.shuffle(idx_k)
            
            take_for_server = samples_per_class + (1 if k < remainder else 0)
            
            server_idx.extend(idx_k[:take_for_server])
            remaining_idx_list.extend(idx_k[take_for_server:])
            
        client_indices[-1] = [int(x) for x in server_idx]
        
        remaining_indices = np.array(remaining_idx_list)
        np.random.shuffle(remaining_indices)
        
    else:
        remaining_indices = np.arange(num_samples)
        np.random.shuffle(remaining_indices)

    remaining_targets = targets[remaining_indices]
    labels = remaining_targets.cpu().numpy()
    
    num_classes = len(np.unique(labels))
    
    if is_iid:
        q = 1 / num_classes
    
    num_virtual_clients = max(num_clients, num_classes)
    
    virtual_client_list = np.arange(num_virtual_clients)
    np.random.shuffle(virtual_client_list)
    
    client_groups = np.array_split(virtual_client_list, num_classes)
    
    virtual_client_indices_list = {i: [] for i in range(num_virtual_clients)}
    
    for idx_in_remaining, label in enumerate(labels):
        actual_idx = remaining_indices[idx_in_remaining]
        
        random_value = np.random.rand()
        
        if random_value < q:
            target_group_idx = label
        else:
            other_groups_indices = []
            for g in range(num_classes):
                if g != label:
                    other_groups_indices.append(g)
            
            target_group_idx = np.random.choice(other_groups_indices)
            
        target_group_clients = client_groups[target_group_idx]
        target_virtual_client = np.random.choice(target_group_clients)
        
        virtual_client_indices_list[target_virtual_client].append(int(actual_idx))

    client_indices_list = {i: [] for i in range(num_clients)}
    for v_client_id, indices in virtual_client_indices_list.items():
        actual_client_id = v_client_id % num_clients
        client_indices_list[actual_client_id].extend(indices)

    for i in range(num_clients):
        client_indices[i] = client_indices_list[i]

    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving partition map to {full_file_path}...")
    print(f"  - Server (-1): {len(client_indices.get(-1, []))} samples")
    print(f"  - Clients (0~{num_clients-1}): Avg {len(remaining_indices)//num_clients} samples")
    
    with open(full_file_path, 'wb') as f:
        pickle.dump(client_indices, f)
    
    server_client_ratio = (len(remaining_indices)//num_clients) / len(client_indices.get(-1, []))
    return full_file_path, client_indices, server_client_ratio


if __name__ == '__main__':
    partition_and_save('MNIST', num_clients=10, is_iid=False, alpha=0.5)