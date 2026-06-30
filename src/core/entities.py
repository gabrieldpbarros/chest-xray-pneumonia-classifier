import torch
from dataclasses import dataclass

@dataclass
class NetworkOutput:
    """
    Properties:
        predicted_class (str)
        activation_map (torch.tensor)
        gradient (torch.tensor)
    """
    predicted_class: str
    activation_map: torch.Tensor
    gradient: torch.Tensor

    def to_cpu(self):
        self.activation_map = self.activation_map.detach().cpu()
        self.gradient = self.gradient.detach().cpu()
        return self