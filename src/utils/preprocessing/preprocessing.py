import numpy as np
from prep_functions import  (le_dataset, aplica_clahe, aplica_filtro,
                             segmenta_pulmao, redimensiona,
                             normaliza, replica_canal)

def preprocessar(caminho_img: str) -> np.ndarray:
    """
    Estrutura:
    1. Leitura em escala de cinza
    2. CLAHE (realce de contraste)
    3. Filtragem (Bilateral Filter)
    4. Segmentação pulmonar
    5. Resize
    6. Normalização ([0,1])
    7. Canal unico -> 3 canais (para modelos pré-treinados)
    8. Augmentação (somente em treino) - evita overfitting

    A função é modular pois, caso queiramos mudar a abordagem de normalização
    ou filtros, poderemos ser problemas.

    :param caminho_img: O caminho de arquivo com os dados
    :return: Retorna os dados do dataset processado
    """

    # 1. Leitura
    img = le_dataset(caminho_img)

    # 2. CLAHE
    img = aplica_clahe(img)

    # 3. Bilateral Filter
    img = aplica_filtro(img)

    # 4. Segmentação
    img, _ = segmenta_pulmao(img)

    # 5. Redimensionamento
    img = redimensiona(img)

    # 6. Normalização
    img = normaliza(img)

    # 7. Replicar canal
    img = replica_canal(img)

    # 8. Augmentação (somente em treino)
    # img = augmenta(img)

    return img