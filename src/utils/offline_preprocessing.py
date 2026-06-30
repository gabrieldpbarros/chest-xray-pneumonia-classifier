import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from preprocessing import prep_functions as prep
from preprocessing.lung_unet.models import PretrainedUNet
# import segmentation_models_pytorch as smp
from pathlib import Path
import matplotlib.pyplot as plt

def _plotRandomSamples(path):
    import random
    amostra = random.sample(list((SEGMENTED_DB / "train" / "PNEUMONIA").iterdir()), 30)

    fig, axs = plt.subplots(6, 5, figsize=(15, 18))
    for ax, img_path in zip(axs.flat, amostra):
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.set_title(img_path.name, fontsize=6)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def _countEmptyMasks(device, path):
    model_unet = PretrainedUNet(
        in_channels=1,
        out_channels=2,
        batch_norm=True,
        upscale_mode="bilinear"
    )
    trained_network_path = Path(__file__).parent / "preprocessing" / "lung_unet" / "unet-6v.pt"
    state_dict = torch.load(trained_network_path, map_location=device)
    model_unet.load_state_dict(state_dict)
    model_unet.to(device).eval()

    empty = []
    folder = path / "train" / "PNEUMONIA"
    for img_path in folder.iterdir():
        if suffix_is_valid(img_path.suffix):
            img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
            _, mask = prep.segmenta_pulmao_unet(img, model_unet, device)
            if np.sum(mask) == 0:
                empty.append(img_path.name)
    print(f"{len(empty)} imagens vazias de pneumonia")

def _viewDataSegmentation(model_unet, device, path):
    import torch
    import torch.nn.functional as F
    import cv2

    # Reaproveita o mesmo pré-processamento da função, mas parando antes do argmax
    img_path = path / "train" / "NORMAL" / "NORMAL-28501-0001.jpeg"
    img_teste = prep.le_dataset(img_path)
    img_resized = cv2.resize(img_teste, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model_unet(img_tensor)
        probs = F.softmax(out, dim=1)

    canal0 = probs[0, 0].cpu().numpy()
    canal1 = probs[0, 1].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_resized, cmap='gray'); axs[0].set_title('Imagem original (resized)')
    axs[1].imshow(canal0, cmap='gray'); axs[1].set_title('Canal 0')
    axs[2].imshow(canal1, cmap='gray'); axs[2].set_title('Canal 1')
    plt.show()

def process_and_save_dataset(original_db_path: str, new_db_path: str):
    """
    Varre o dataset original, aplica a segmentação pesada e salva 
    as imagens recortadas em uma nova pasta.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    splits = ['train', 'test', 'val']
    classes = ['NORMAL', 'PNEUMONIA']
    # print(f"Buscando em: {original_db_path}")
    # arquivos_encontrados = list(original_db_path.rglob('*'))
    # print(f"Total de itens encontrados pelo rglob: {len(arquivos_encontrados)}")
    # if len(arquivos_encontrados) > 0:
    #     print(f"Exemplo de item encontrado: {arquivos_encontrados[0]}")
    #     print(f"É diretório? {arquivos_encontrados[0].is_dir()}")
    # else:
    #     print("O rglob não encontrou absolutamente nada. O caminho está correto?")

    extensoes = {'.jpeg', '.jpg', '.png'}
    total_images = sum(
        1 for caminho in original_db_path.rglob('*') 
        if caminho.suffix.lower() in extensoes
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_unet = PretrainedUNet(
        in_channels=1,
        out_channels=2,
        batch_norm=True,
        upscale_mode="bilinear"
    )
    trained_network_path = Path(__file__).parent / "preprocessing" / "lung_unet" / "unet-6v.pt"
    state_dict = torch.load(trained_network_path, map_location=device)
    model_unet.load_state_dict(state_dict)
    model_unet.to(device).eval()
    #_viewDataSegmentation(model_unet, device, original_db_path)

    print(f"Iniciando pré-processamento offline de {total_images} imagens...")
    pbar = tqdm(total=total_images, desc="Processando")

    for split in splits:
        for cls in classes:
            input_folder = original_db_path / split / cls
            output_folder = new_db_path / split / cls

            if not input_folder.exists():
                continue

            output_folder.mkdir(parents=True, exist_ok=True)

            for img_path in input_folder.iterdir():
                if suffix_is_valid(img_path.suffix):
                    try:
                        pil_img = Image.open(img_path).convert("L")
                        img = np.array(pil_img, dtype=np.uint8)
                        img, _ = prep.segmenta_pulmao_bbox(img, model_unet, device)
                        img = prep.aplica_clahe(img)
                        img = prep.aplica_filtro(img)
                        img = prep.redimensiona(img)
                        final_pil = Image.fromarray(img)
                        final_pil.save(output_folder / img_path.name)
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"\nErro ao processar {img_path}: {e}")
                    
                    pbar.update(1)

    pbar.close()
    print(f"\nPré-processamento concluído! Dataset em: {new_db_path}")

def suffix_is_valid(suffix):
    return suffix.lower() in ['.jpeg', '.jpg', '.png']

if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent.parent
    current_dir = project_dir / "utils"
    ORIGINAL_DB = project_dir / "db" / "chest_xray"
    SEGMENTED_DB = project_dir / "db" / "chest_xray_segmented"

    process_and_save_dataset(ORIGINAL_DB, SEGMENTED_DB)
    #_countEmptyMasks('cuda', ORIGINAL_DB)
    #_plotRandomSamples(SEGMENTED_DB)