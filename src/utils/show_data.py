import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from torch.utils.data import DataLoader
from typing import Tuple

def sampleBatch(
    dataloader: DataLoader,
    classes: Tuple[str] = ["NORMAL", "PNEUMONIA"]
) -> None:
    """
    Função de visualização do lote que foi extraído na criação dos DataLoaders

    Args:
        dataloader: DataLoader com imagens dos raios-x
        classes: Labels das imagens (normal e pneumonia)
    """
    # Pega um lote de imagens
    images, labels = next(iter(dataloader))
    
    plt.figure(figsize=(12, 4))
    for idx in range(5):
        _ = plt.subplot(1, 5, idx+1)
        # O PyTorch é (C, H, W), o Matplotlib quer (H, W, C)
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(classes[labels[idx]])
        plt.axis("off")
    plt.show()

def overlayGradCAM(
    img_tensor,
    cam_heatmap,
    alpha=0.5
):
    img = img_tensor.squeeze(0).cpu().numpy()
    if len(img.shape) == 2 or img.shape[0] == 1: 
        img = np.stack((img.squeeze(),)*3, axis=-1)
    elif img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img = np.uint8(255 * img)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    img = np.float32(img) / 255
    overlayed_img = heatmap_colored * alpha + img * (1 - alpha)
    overlayed_img = overlayed_img / np.max(overlayed_img)
    return overlayed_img

def showGradMap(
    image,
    label: str
):
    plt.close('all')
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.title(label=label)
    plt.show()

if __name__ == "__main__":
    from src.utils.load_data import getTrainingDataLoaders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, 'db')

    train_dl, _, _ = getTrainingDataLoaders(db_path)
    sampleBatch(train_dl)