import torch
import torch.nn as nn

from src.core.entities import NetworkOutput
from torch.utils.data import DataLoader
from typing import List
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class SimpleCNN_Base(nn.Module):
    """
    Modelo flexível de rede convolucional, desenvolvido de forma a suportar
    distintas arquiteturas, em que a função que cria o modelo pode definir
    se terá dropout, batch normalization e Global Average Pooling.
    """
    def __init__(
            self,
            filters_list: List[int],
            num_classes: int = 2,
            dropout: bool = False,
            batch_norm: bool = False,
            GAP: bool = False,
            input_size: int = 224
    ):
        """
        Args:
            filters_list: Lista que contém as dimensões (quantidade total de neurônios)
                          de cada camada de filtro da CNN.
            num_classes: Quantidade de classificações possíveis.
            dropout: Parâmetro que determina se o modelo utilizará dropout.
            batch_norm: Define se o modelo terá batch normalization.
            GAP: Informa se o modelo utilizará Global Average Pooling
            input_size: Tamanho da imagem, utilizado para quando o modelo não conter GAP
        """
        super(SimpleCNN_Base, self).__init__()
        layers = []
        in_channels = 1 # canais de cor (como trabalhamos com grayscale, há apenas 1 canal)

        for filtro in filters_list:
            layers.append(nn.Conv2d(in_channels, filtro, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(filtro))
            layers.append(nn.ReLU()) # escolhido segundo o modelo da AlexNet
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = filtro # atualiza o tamanho da próxima entrada

        self.conv_layers = nn.Sequential(*layers)
        self.use_gap = GAP
        self.use_dropout = dropout

        if self.use_gap:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            input_features = filters_list[-1]
        else:
            # A imagem reduz pela metade a cada camada de Pooling (devido ao stride=2)
            num_pools = len(filters_list)
            final_img_size = input_size // (2 ** num_pools)

            # Features = Quantidade de Filtros * Altura * Largura
            input_features = filters_list[-1] * final_img_size * final_img_size

        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(input_features, num_classes)

    def forward(self, x):
        # --- 1. Camadas convolucionais ---
        x = self.conv_layers(x)

        # --- 2. Saída (flatten ou GAP) ---
        if self.use_gap:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1) # Remove as dimensões 1x1 extras
        else:
            x = x.view(x.size(0), -1) # Flatten tradicional

        # Dropout opcional
        if self.use_dropout:
            x = self.dropout(x)

        # --- 3. Classificação final ---
        x = self.classifier(x)
        return x
    
class cnnModel():
    """
    Classe principal, utilizada para instanciar o modelo e definir os métodos
    de treino, validação e teste.
    """
    def __init__(
            self,
            device: str,
            model: nn.Module,
            best_model_path: str = "models/base_best_cnn.pth"
    ):
        """
        Args:
            device: Dispositivo de aceleração.
            filters_list: Lista que contém as dimensões (quantidade total de neurônios)
                          de cada camada de filtro da CNN.
            num_classes: Quantidade de classificações possíveis.
            dropout: Parâmetro que determina se o modelo utilizará dropout.
            batch_norm: Define se o modelo terá batch normalization.
            GAP: Informa se o modelo utilizará Global Average Pooling
            input_size: Tamanho da imagem, utilizado para quando o modelo não conter GAP   
            best_model_path: Diretório de armazenamento do modelo 
        """
        self.device = device
        self.model = model
        self.model.to(self.device)

        self.train_loss = []
        self.val_loss = []
        self.test_class = []
        self.predicted_class = []
        self.best_model_path = best_model_path

    def _getLastConv(self) -> nn.Conv2d:
        last_conv = None
        if hasattr(self.model, 'layer4'): 
            return self.model.layer4[-1]
            
        elif hasattr(self.model, 'conv_layers'):
            for layer in self.model.conv_layers:
                if isinstance(layer, nn.Conv2d):
                    last_conv = layer
            if last_conv is not None:
                return last_conv

    def trainModel(
            self,
            train_loader: DataLoader,
            loss_function: callable,
            optimizer
    ):
        """
        Função de treinamento do modelo, percorrendo uma época.

        Args:
            train_loader: DataLoader de treinamento.
            loss_function: Função de erro definida pelo usuário.
            optimizer: Método de cálculo da retropropagação.
        """

        self.model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device) # Labels já são int (long) do Dataset

            optimizer.zero_grad()
        
            # Feedforward
            outputs = self.model(features) 
        
            # Cálculo do Erro (CrossEntropyLoss aceita Logits e Indices de Classe)
            loss = loss_function(outputs, labels)
        
            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        self.train_loss.append(total_loss / len(train_loader))

    def validateModel(
            self,
            val_loader: DataLoader, 
            loss_function: callable,
    ):
        """
        Função de validação do modelo, percorrendo uma época. Como não estamos treinando o modelo, não precisamos
        calcular o gradiente descendente estocástico, o qual é necessário para atualizar os pesos sinápticos dos
        neurônios das camadas ocultas.

        Args:
            val_loader: DataLoader de validação.
            loss_function: Função de erro definida pelo usuário.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()

        self.val_loss.append(total_loss / len(val_loader))

    def testModel(
            self,
            test_loader: DataLoader
    ):
        """
        Função de teste do modelo.

        Args:
            model: Modelo de rede neural a ser treinado.
            test_loader: DataLoader de teste.
        """
        self.model.eval()
        correct_cases = 0
        total_samples = 0

        real_lst = []
        predicted_lst = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)

                _, predicted = torch.max(outputs.data, 1)

                total_samples += labels.size(0)
                correct_cases += (predicted == labels).sum().item()

                real_lst.extend(labels.cpu().numpy())
                predicted_lst.extend(predicted.cpu().numpy())

        self.test_class = real_lst
        self.predicted_class = predicted_lst
        self.acc = accuracy_score(self.test_class, self.predicted_class)
        self.prec = precision_score(self.test_class, self.predicted_class)
        self.rec = recall_score(self.test_class, self.predicted_class)
        self.f1 = f1_score(self.test_class, self.predicted_class)

    def passInput(
            self,
            data_input: DataLoader
    ) -> dict:
        """
        Função de entrada de dados pelo modelo.

        Args:
            model: Modelo de rede neural a ser treinado.
            data_input: DataLoader que contém a entrada da rede.
        """
        self.model.eval()
        last_layer = self._getLastConv()
        results: list[NetworkOutput] = []
        activation_maps: torch.Tensor = None
        weights: torch.Tensor = None
        self.predicted_class = []

        def saveActivationMaps(module, inp, out):
            nonlocal activation_maps
            activation_maps = out.detach()

        def saveWeights(module, grad_inp, grad_out):
            nonlocal weights
            weights = grad_out[0].detach()

        map_hook = last_layer.register_forward_hook(saveActivationMaps)
        gradient_hook = last_layer.register_full_backward_hook(saveWeights)

        try:
            for batch in data_input:
                features = batch[0].to(self.device)
                for i in range(features.size(0)):
                    image = features[i:i+1]
                    with torch.enable_grad():
                        output = self.model(image)
                        _, predicted = torch.max(output.data, 1)
                        self.model.zero_grad()
                        output[0, predicted.item()].backward()
                        self.predicted_class.extend(predicted.cpu().numpy())
                    results.append(
                        NetworkOutput(
                            predicted_class=predicted.item(),
                            activation_map=activation_maps.squeeze(0),
                            gradient=weights.squeeze(0)
                        )
                    )
        finally:
            map_hook.remove()
            gradient_hook.remove()

        return results

    def saveCheckpoint(self):
        torch.save(self.model.state_dict(), self.best_model_path)