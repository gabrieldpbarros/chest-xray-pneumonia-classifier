import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology, measure
from skimage.segmentation import clear_border
import albumentations as A

def le_dataset(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Imagem nao encontrada em: {caminho}"
    return img

def aplica_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe

def aplica_filtro(img):
    img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return img_filtered

def segmenta_pulmao(img):
    # Otsu
    _, img_otsu = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remover bordas conectadas ao limite da imagem
    cleared = clear_border(img_otsu // 255).astype(np.uint8) * 255

    # Operacoes morfologicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(cleared, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Mantem dois maiores componentes (pulmao esq. + dir.)
    labeled  = measure.label(closed)
    props    = measure.regionprops(labeled)
    props    = sorted(props, key=lambda p: p.area, reverse=True)

    mask = np.zeros_like(closed)
    for prop in props[:2]:                  # Os dois maiores
        mask[labeled == prop.label] = 255

    # Aplicar mascara a imagem original
    img_segmented = cv2.bitwise_and(img, img, mask=mask)
    return img_segmented, mask

def redimensiona(img, target_size=(224, 224)):
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    return img_resized

def normaliza(img):
    img_norm = img / 255.0
    return img_norm

def replica_canal(img):
    img_replicada = np.stack([img, img, img], axis=-1)
    # (Replica o canal unico em tres
    # Deixa os dados prontos pro treino de modelos
    return img_replicada

def augmenta(img):
    transform_treino = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.GridDistortion(p=0.2),        # simula variações de posicionamento
        A.ElasticTransform(p=0.2),      # deformação suave dos tecidos
    ])

    # Aplicação
    img_uint8 = (img * 255).astype(np.uint8)
    augmented = transform_treino(image=img_uint8)
    img_aug = augmented["image"]
    return img_aug