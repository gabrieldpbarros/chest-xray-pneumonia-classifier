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