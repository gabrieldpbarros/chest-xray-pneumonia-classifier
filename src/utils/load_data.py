import os
import pandas as pd
import numpy as np
from preprocessing import prep_functions as prep

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple

class CustomImgDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            transform: callable = None
    ):
        """
        Args:
            df: DataFrame contendo nomes e tipos.
            img_dir: Caminho para a pasta das imagens.
            transform: Transformações do PyTorch.
        """
        self.df = df # dataframe parcial, abrange apenas o subconjunto tratado
        self.transform = transform
        self.labels = {"NORMAL": 0, "PNEUMONIA": 1}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row['path']
        label_str = row['label']

        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Erro ao ler {img_path}: {e}")
            # Retorna uma imagem preta em caso de erro para não quebrar o treino
            image = Image.new('L', (224, 224))

        numerical_label = self.labels[label_str]

        if self.transform:
            image = self.transform(image)

        return image, numerical_label
    
def _find_dir(root_path: str) -> str:
    """
    Este dataset possui uma certa inconsistência na importação pelo kagglehub,
    principalmente quando se trata de diretórios aninhados. Para contornar esse problema,
    usamos essa função para chegar à pasta raiz, que possui as pastas de treino,
    validação e teste.

    Args:
        root_path: Caminho inicial retornado pelo módulo do kagglehub
    """
    possibilities = [
        root_path,
        os.path.join(root_path, "chest_xray"),
        os.path.join(root_path, "chest_xray/chest_xray"),        
    ]

    train_dir = "train/"
    final_dir = None # None garante mais segurança caso algo dê errado

    for p in possibilities:
        if os.path.exists(os.path.join(p, train_dir)):
            final_dir = p
            break # encontramos, entao podemos parar o loop

    if final_dir is None:
        raise FileNotFoundError(f"Não foi possível encontrar a pasta 'train' dentro de {root_path}. Verifique o download.")
    
    print(f"Dataset localizado em: {final_dir}")
    return final_dir

def _createDataFrame(root_dir: str, train: bool):
    """
    Varre as pastas train/test/val e cria um DataFrame único com caminhos e labels.

    Como a pasta de validação é muito pequena (possui apenas 8 imagens para as duas
    classificações), optamos por reunir todas as imagens do dataset e separá-las por
    conta própria.

    Args:
        root_dir: Diretório raiz que contém as três pastas (train, test e val)
        train: Booleano que indica qual tipo de carregamento deve ser feito
    """
    filepaths = []
    labels = []
    
    # As classes são as subpastas
    classes = ["NORMAL", "PNEUMONIA"]
    
    if train:
        # Percorre train, test e val para juntar todas em um único DataFrame
        for split in ["train", "test", "val"]:
            split_path = os.path.join(root_dir, split)
            if not os.path.exists(split_path): continue
                
            for label in classes:
                class_path = os.path.join(split_path, label)
                if not os.path.exists(class_path): continue
                    
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        full_path = os.path.join(class_path, img_name)
                        filepaths.append(full_path)
                        labels.append(label)
    else:
        test_path = os.path.join(root_dir, "test")
        if not os.path.exists(test_path): raise FileNotFoundError(f"ERRO: Não existe dados de treino em {test_path}")
        for label in classes:
                class_path = os.path.join(test_path, label)
                if not os.path.exists(class_path): continue
                    
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        full_path = os.path.join(class_path, img_name)
                        filepaths.append(full_path)
                        labels.append(label)
                    
    df = pd.DataFrame({
        'path': filepaths,
        'label': labels
    })
    return df

class PreProcessPipeline:
    """
    Transform customizado e compatível com torchvision.transforms.Compose.
    Recebe uma PIL Image e aplica o pipeline realizado.
    """

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        # Transforma PIL em numpy grayscale
        img = np.array(pil_img.convert("L"), dtype=uint8)

        # Pipeline
        img     = prep.aplica_clahe(img)
        img     = prep.aplica_filtro(img)
        img, _  = prep.segmenta_pulmao(img)
        img     = prep.redimensiona(img)        # -> (224, 224)
        img     = prep.normaliza(img)           # -> float32, [0,1]
        img     = prep.replica_canal(img)       # -> (224, 224, 3)

        return img

def _setTrainTransformations() -> callable:
    """
    Função auxiliar que contém as definições das transformações aplicadas nas
    imagens do dataset.

    Foi adicionado também o pipeline de pré-processamento das imagens realizado.
    Ele trabalha como uma classe que pode ser chamada por transforms.Compose
    
    Vale ressaltar um detalhe importante no RandomAffine, a aplicação de um zoom 
    de 5% a 15% na imagem. Fazemos isso a fim de fugir da possibilidade de problemas 
    com ruídos nas rotações, as quais geram tarjas pretas nas bordas da imagem. Ao 
    aplicar esse zoom, sabendo que todas as imagens de raio-X possuem bordas pretas, 
    garantimos que não hajam inconsistências na análise pela CNN. A escolha dos 
    valores do zoom foi feita de modo empírico.
    """
    train_transforms = transforms.Compose([
        PreProcessPipeline(),
        
        transforms.ToTensor(),

        transforms.RandomAffine(
            degrees=10,             # Rotação leve (máx 10 graus)
            translate=(0.05, 0.05), # Move um pouco para os lados (5%)
            scale=(1.05, 1.15),     # Zoom entre 5% e 15%
            fill=0                  # Se sobrar espaço, preenche com preto
        ),
        # transforms.Normalize([0.5], [0.5])
    ])


    return train_transforms

def _setTestTransformations() -> callable:
    test_transform = transforms.Compose([
        PreProcessPipeline(),
        transforms.ToTensor(),
    ])
    return test_transform

def getTrainingDataLoaders(
        root_path: str,
        batch_size: int = 32
):
    """
    Carrega os dados em formato de um DataLoader.

    Args:
        root_path: Diretório raiz das pastas dos subconjuntos.
        batch_size: Tamanho dos lotes de treinamento.

    Output:
        Dados separados conforme a proporção definida internamente em conjuntos de treino,
        validação e teste.
    """
    from sklearn.model_selection import train_test_split
    proportions = (0.8, 0.2) # treino e validação + teste
    # --- 1. Carregamento do DataFrame ---
    # Correção do PATH
    path = _find_dir(root_path)
    full_df = _createDataFrame(path, train=True)

    # --- 2. Separação dos subconjuntos ---
    train_df, temp = train_test_split(
        full_df,
        test_size=proportions[1],
        stratify=full_df["label"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["label"],
        random_state=42
    )

    # --- 3. Definição das transformações ---
    train_transforms, val_test_transforms = _setTrainTransformations()
    val_test_transforms = _setTestTransformations()

    # --- 4. Instanciação dos DataSets e DataLoaders ---
    train_ds = CustomImgDataset(train_df, train_transforms)
    val_ds = CustomImgDataset(val_df, val_test_transforms)
    test_ds = CustomImgDataset(test_df, val_test_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # colocar em 2 no Colab, caso identifique algum erro
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader

def getSampleData(
        root_path: str,
        batch_size: int = 32
):
    path = _find_dir(root_path)
    full_df = _createDataFrame(path, train=False)
    transforms = _setTestTransformations()
    test_ds = CustomImgDataset(full_df, transform=transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return test_loader

if __name__ == "__main__":
    # 1. Descobre onde este script (data_loader.py) está
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Sobe um nível para chegar na Raiz do Projeto
    project_root = os.path.dirname(current_dir)
    # 3. Monta o caminho para a pasta db
    db_path = os.path.join(project_root, 'db')