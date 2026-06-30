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

def segmenta_pulmao_otsu(img):
    # Otsu
    img_inv = cv2.bitwise_not(img)
    _, img_otsu = cv2.threshold(
        img_inv, 0, 255,
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

def segmenta_pulmao_unet(img, model_unet, device, input_size=512):
    import torch
    import torch.nn.functional as F
    import cv2
    import numpy as np

    orig_h, orig_w = img.shape[:2]

    # Redimensiona para o tamanho de treino
    img_resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LANCZOS4)

    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        out = model_unet(img_tensor)              # (1, 2, H, W)
        probs = F.softmax(out, dim=1)
        mask_pred = torch.argmax(probs, dim=1)     # (1, H, W) -> 0 ou 1
        mask_pred = mask_pred.squeeze().cpu().numpy()

    mask = (mask_pred == 1).astype(np.uint8)
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    img_segmented = cv2.bitwise_and(img, img, mask=mask)
    return img_segmented, mask

def segmenta_pulmao_bbox(
    img, 
    model_unet, 
    device, 
    input_size=512, 
    margin_pct=0.05,
    min_area_pct=0.15,
    min_aspect=0.3,
    max_aspect=2.5
):
    img_segmented_full, mask = segmenta_pulmao_unet(img, model_unet, device, input_size)
    
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img, None  # fallback tratado depois
    
    h, w = img.shape[:2]
    margin_x = int(w * margin_pct)
    margin_y = int(h * margin_pct)
    
    x_min = max(0, xs.min() - margin_x)
    x_max = min(w, xs.max() + margin_x)
    y_min = max(0, ys.min() - margin_y)
    y_max = min(h, ys.max() + margin_y)

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    area_pct = (crop_w * crop_h) / (w * h)
    aspect = crop_w / crop_h
    if area_pct < min_area_pct or not (min_aspect < aspect < max_aspect):
        return img, None
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    desvio_x = abs(center_x - w/2) / w
    desvio_y = abs(center_y - h/2) / h

    if desvio_x > 0.2 or desvio_y > 0.2:
        return img, None

    img_cropped = img[y_min:y_max, x_min:x_max]
    return img_cropped, (y_min, y_max, x_min, x_max)

def redimensiona(img, target_size=224):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    img_padded = cv2.copyMakeBorder(
        img_resized,
        top, 
        bottom, 
        left, 
        right,
        cv2.BORDER_CONSTANT,
        value=0
    )
    return img_padded

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
        A.GaussNoise(std_range=(0.008, 0.02), p=0.3),
        A.GridDistortion(p=0.2),        # simula variações de posicionamento
        A.ElasticTransform(p=0.2),      # deformação suave dos tecidos
    ])

    # Aplicação
    img_uint8 = (img * 255).astype(np.uint8)
    augmented = transform_treino(image=img_uint8)
    img_aug = augmented["image"]
    return img_aug