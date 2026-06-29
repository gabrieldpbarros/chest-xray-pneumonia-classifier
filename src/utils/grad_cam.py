import torch
import torch.nn.functional as F
import cv2
import numpy as np
from src.core.entities import NetworkOutput

class GradCAM:
    def __init__(
            self,
            model_output: NetworkOutput,
            target_size: tuple[int, int]
    ):
        self.activation_maps = model_output.activation_map
        self.gradients = model_output.gradient
        self.target_size = target_size
        self.weights = torch.mean(self.gradients, dim=(1,2))
        self.device = self.activation_maps.device

    def computeGradCAM(self):
        cam = torch.zeros(self.activation_maps.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(self.weights):
            cam += w * self.activation_maps[i]

        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)

        cam = cam.cpu().numpy()
        cam_resized = cv2.resize(cam, self.target_size)
        return cam_resized