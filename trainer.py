import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import copy

class LocalTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_and_get_update(
            self, 
            dataloader: DataLoader, 
            epochs: int, 
            lr: float = 0.01) -> torch.Tensor:
        initial_weights_vector = parameters_to_vector(self.model.parameters()).detach().clone()
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
        # print(f"[Log] End of Local Training Loss: {loss.item():.4f}")

        trained_weights_vector = parameters_to_vector(self.model.parameters()).detach().clone()

        update_vector = trained_weights_vector - initial_weights_vector

        vector_to_parameters(initial_weights_vector, self.model.parameters())

        optimizer.zero_grad(set_to_none=True)
        del optimizer
        del output
        del loss
        del data
        del target

        return update_vector.detach().cpu().clone()

    def apply_global_update(self, global_update_vector: torch.Tensor):
        current_weights_vector = parameters_to_vector(self.model.parameters()).detach()
        
        if global_update_vector.device != current_weights_vector.device:
            global_update_vector = global_update_vector.to(current_weights_vector.device)
            
        new_weights_vector = current_weights_vector + global_update_vector
        
        vector_to_parameters(new_weights_vector, self.model.parameters())
        
    def get_current_flat_weights(self) -> torch.Tensor:
        return parameters_to_vector(self.model.parameters()).detach().cpu()