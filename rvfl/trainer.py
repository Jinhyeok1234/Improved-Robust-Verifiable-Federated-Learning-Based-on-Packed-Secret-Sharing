"""Local training and model evaluation."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader


class LocalTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_and_get_update(self, dataloader, epochs: int, lr: float = 0.01) -> torch.Tensor:
        """Train from the model's current parameters and return the update (trained minus
        initial parameters) as a flat CPU vector. The parameters are restored afterwards; the
        batch-norm running statistics keep what the training accumulated."""
        initial = parameters_to_vector(self.model.parameters()).detach().clone()
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(data), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
        trained = parameters_to_vector(self.model.parameters()).detach().clone()
        vector_to_parameters(initial, self.model.parameters())
        optimizer.zero_grad(set_to_none=True)
        return (trained - initial).detach().cpu().clone()


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        correct += (model(data).argmax(dim=1) == target).sum().item()
        total += target.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def attack_success_rate(model: nn.Module, loader: DataLoader, device, target_label: int,
                        stamp) -> float:
    """Fraction of test inputs whose true label is not target_label that the model classifies
    as target_label once the trigger is stamped on them (`stamp`: batch -> batch)."""
    model.eval()
    hit, total = 0, 0
    for data, target in loader:
        keep = target != target_label
        if keep.sum() == 0:
            continue
        data = stamp(data[keep]).to(device)
        hit += (model(data).argmax(dim=1) == target_label).sum().item()
        total += data.size(0)
    return hit / max(total, 1)
