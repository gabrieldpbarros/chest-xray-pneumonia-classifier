import matplotlib.pyplot as plt
import numpy as np

from typing import List
from sklearn.metrics import confusion_matrix

def plotLosses(
        title,
        loss1: List[float],
        loss2: List[float],
        loss1_label: str,
        loss2_label: str,
        epochs: int,
        saving: str
    ) -> None:
    plt.close("all")
    plt.figure()
    plt.plot(range(epochs), loss1, label=loss1_label)
    plt.plot(range(epochs), loss2, label=loss2_label)
    plt.title(title)
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()
    plt.savefig(saving)
    plt.show()

def plotCMatrix(
        labels: np.ndarray,
        predicted: np.ndarray,
        class_names: List[str],
        saving: str
    ) -> None:
    # --- 1. Cálculo da Matriz ---
    cm = confusion_matrix(labels, predicted)
    num_classes = cm.shape[0]

    # --- 2. Plotagem ---
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão")
    plt.colorbar()
    
    # Define os labels dos eixos com os nomes reais (ex: Fire, Water)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right") # Rotação ajuda a ler
    plt.yticks(tick_marks, class_names)
    
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Verdadeira")

    # Escrever os valores em cada célula
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                verticalalignment="center", # Centraliza verticalmente também
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout() 
    plt.savefig(saving)
    plt.show()