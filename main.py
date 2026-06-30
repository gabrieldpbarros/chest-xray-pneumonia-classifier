import os
import torch
import yaml
from pathlib import Path
from src.models.simple_cnn import cnnModel, SimpleCNN_Base

EPOCHS = 30
DEVICE = "cuda"
DB_PATH = "db/"
MODELS_PATH = "src/models"
CONFIG_PATH = os.path.join(MODELS_PATH, "topology.yaml")
IMG_RESIZE = (224,224)

def _load_configs(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def trainingPipeline(device, db_path, configs):
    from src.utils.load_data import getTrainingDataLoaders
    from torch import nn
    train_loader, val_loader, test_loader = getTrainingDataLoaders(db_path)

    simple_configs = configs['simple_cnn']
    complete_configs = configs['complex_cnn']

    simpleCNN = SimpleCNN_Base(
        filters_list=simple_configs['filters_list'],
        dropout=simple_configs['dropout'],
        batch_norm=simple_configs['batch_norm'],
        GAP=simple_configs['GAP']
    )
    simpleModel = cnnModel(
        device=device,
        model=simpleCNN,
        best_model_path=simple_configs['best_model_path']
    )

    completeCNN = SimpleCNN_Base(
        filters_list=complete_configs['filters_list'],
        dropout=complete_configs['dropout'],
        batch_norm=complete_configs['batch_norm'],
        GAP=complete_configs['GAP']
    )
    completeModel = cnnModel(
        device=device,
        model=completeCNN,
        best_model_path=complete_configs['best_model_path']
    )

    import torch.optim as optim
    optimizer_simple = optim.Adam(simpleModel.model.parameters(), lr=0.001)
    optimizer_complete = optim.Adam(completeModel.model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = EPOCHS

    best_simple_val_loss = float('inf')
    best_complete_val_loss = float('inf')
    for epoch in range(num_epochs):
        simpleModel.trainModel(train_loader, loss_fn, optimizer_simple)
        completeModel.trainModel(train_loader, loss_fn, optimizer_complete)

        simpleModel.validateModel(val_loader, loss_fn)
        completeModel.validateModel(val_loader, loss_fn)

        if simpleModel.val_loss[-1] < best_simple_val_loss:
            best_simple_val_loss = simpleModel.val_loss[-1]
            simpleModel.saveCheckpoint()
            print(f"Epoch {epoch+1}: Melhor modelo (simples) salvo! (Val Loss: {best_simple_val_loss:.4f})")

        if completeModel.val_loss[-1] < best_complete_val_loss:
            best_complete_val_loss = completeModel.val_loss[-1]
            completeModel.saveCheckpoint()
            print(f"Epoch {epoch+1}: Melhor modelo (completo) salvo! (Val Loss: {best_complete_val_loss:.4f})")

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Simple Train Loss: {simpleModel.train_loss[epoch]:.4f} | Simple Val Loss: {simpleModel.val_loss[epoch]:.4f}")
            print(f"Epoch {epoch+1}/{EPOCHS} | Complete Train Loss: {completeModel.train_loss[epoch]:.4f} | Complete Val Loss: {completeModel.val_loss[epoch]:.4f}")

    from src.utils.view_loss import plotLosses, plotCMatrix
    plotLosses(
        "Modelo Simples",
        simpleModel.train_loss,
        simpleModel.val_loss,
        "Erro de treino",
        "Erro de validação",
        EPOCHS,
        "assets/Simple_CNN_Losses.png"
    )

    plotLosses(
        "Modelo Completo",
        completeModel.train_loss,
        completeModel.val_loss,
        "Erro de treino",
        "Erro de validação",
        EPOCHS,
        "assets/Complete_CNN_Losses.png"
    )

    simpleModel.model.load_state_dict(torch.load(simple_configs['best_model_path']))
    simpleModel.testModel(test_loader)
    print(f"--- Relatório Final (Modelo Simples) ---")
    print(f"Acurácia (Accuracy):   {simpleModel.acc*100:.2f}%")
    print(f"Precisão (Precision):  {simpleModel.prec*100:.2f}%")
    print(f"Sensibilidade (Recall):{simpleModel.rec*100:.2f}%")
    print(f"F1 Score:              {simpleModel.f1*100:.2f}%")

    completeModel.model.load_state_dict(torch.load(complete_configs['best_model_path']))
    completeModel.testModel(test_loader)
    print(f"--- Relatório Final (Modelo Completo) ---")
    print(f"Acurácia (Accuracy):   {completeModel.acc*100:.2f}%")
    print(f"Precisão (Precision):  {completeModel.prec*100:.2f}%")
    print(f"Sensibilidade (Recall):{completeModel.rec*100:.2f}%")
    print(f"F1 Score:              {completeModel.f1*100:.2f}%")

    plotCMatrix(
        simpleModel.test_class,
        simpleModel.predicted_class,
        ['Normal', 'Pneumonia'],
        "assets/Simple_CNN_ConfusionMatrix.png"
    )

    plotCMatrix(
        completeModel.test_class,
        completeModel.predicted_class,
        ['Normal', 'Pneumonia'],
        "assets/Complete_CNN_ConfusionMatrix.png"
    )

def transferLearningPipeline(device, db_path, configs):
    import torch.optim as optim
    from torch import nn
    from torchvision import models
    from src.models.simple_cnn import cnnModel
    from src.utils.load_data import getTrainingDataLoaders
    from src.utils.view_loss import plotLosses, plotCMatrix

    train_loader, val_loader, test_loader = getTrainingDataLoaders(db_path)

    resnet_configs = configs['resnet']
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    features_num = resnet.fc.in_features
    resnet.fc = nn.Linear(features_num, 2)
    resnetModel = cnnModel(
        device=device,
        model=resnet,
        best_model_path=resnet_configs['best_model_path']
    )
    optimizer = optim.Adam(resnetModel.model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = EPOCHS
    best_loss = float('inf')
    for epoch in range(num_epochs):
        resnetModel.trainModel(train_loader, loss_fn, optimizer)
        resnetModel.validateModel(val_loader, loss_fn)

        curr_val_loss = resnetModel.val_loss[-1]
        if curr_val_loss < best_loss:
            best_loss = resnetModel.val_loss[-1]
            resnetModel.saveCheckpoint()
            print(f"Epoch {epoch+1}: Melhor modelo (ResNet) salvo! (Val Loss: {best_loss:.4f})")

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | ResNet-50 Train Loss: {resnetModel.train_loss[epoch]:.4f} | ResNet-50 Val Loss: {resnetModel.val_loss[epoch]:.4f}")

    plotLosses(
        "ResNet-50",
        resnetModel.train_loss,
        resnetModel.val_loss,
        "Erro de treino",
        "Erro de validação",
        EPOCHS,
        "assets/ResNet_Losses.png"
    )

    resnetModel.model.load_state_dict(torch.load(resnet_configs['best_model_path']))
    resnetModel.testModel(test_loader)
    print(f"--- Relatório Final (ResNet-50) ---")
    print(f"Acurácia (Accuracy):   {resnetModel.acc*100:.2f}%")
    print(f"Precisão (Precision):  {resnetModel.prec*100:.2f}%")
    print(f"Sensibilidade (Recall):{resnetModel.rec*100:.2f}%")
    print(f"F1 Score:              {resnetModel.f1*100:.2f}%")

    plotCMatrix(
        resnetModel.test_class,
        resnetModel.predicted_class,
        ['Normal', 'Pneumonia'],
        "assets/ResNet_ConfusionMatrix.png"
    )

def predictionPipeline(device, db_path, configs):
    from src.utils.load_data import getSampleData
    from src.core.entities import NetworkOutput
    BATCH_SIZE = 32
    PROJECT_PATH = Path(__file__).parent
    ASSETS_PATH = PROJECT_PATH / "assets"
    CAM_PATH = ASSETS_PATH / "grad_cam"

    def _processCAM(data_loader, out1, out2):
        import matplotlib.pyplot as plt
        from src.utils.grad_cam import GradCAM
        from src.utils.show_data import overlayGradCAM

        CAM_PATH.mkdir(parents=True, exist_ok=True)
        data_iter = iter(data_loader)
        images_batch, labels_batch = next(data_iter)
        class_names = ['NORMAL', 'PNEUMONIA']
        for i in range(images_batch.size(0)):
            imagem_original_tensor = images_batch[i:i+1]
            true_label = class_names[labels_batch[i].item()]

            # --- Processando o Modelo Simples ---
            res_simple = out1[i]
            pred_simple = class_names[res_simple.predicted_class]
            cam_simple = GradCAM(model_output=res_simple, target_size=IMG_RESIZE).computeGradCAM()
            final_img_simple = overlayGradCAM(imagem_original_tensor, cam_simple)

            # --- Processando o Modelo Completo ---
            res_complete = out2[i]
            pred_complete = class_names[res_complete.predicted_class]
            cam_complete = GradCAM(model_output=res_complete, target_size=IMG_RESIZE).computeGradCAM()
            final_img_complete = overlayGradCAM(imagem_original_tensor, cam_complete)

            # --- Plotagem Lado a Lado ---
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(final_img_simple)
            axes[0].set_title(f"Simples | Pred: {pred_simple} | Real: {true_label}")
            axes[0].axis('off')
            axes[1].imshow(final_img_complete)
            axes[1].set_title(f"Completa | Pred: {pred_complete} | Real: {true_label}")
            axes[1].axis('off')
            plt.tight_layout()
            file_path = CAM_PATH / f"sample_{i:02d}.png"
            plt.savefig(file_path, bbox_inches='tight', dpi=150)
            
            plt.close(fig)

    data_loader = getSampleData(db_path, batch_size=BATCH_SIZE)

    simple_configs = configs['simple_cnn']
    simpleModel = cnnModel(
        device=device,
        filters_list=simple_configs['filters_list'],
        dropout=simple_configs['dropout'],
        batch_norm=simple_configs['batch_norm'],
        GAP=simple_configs['GAP'],
        best_model_path=simple_configs['best_model_path']
    )
    simpleModel.model.load_state_dict(torch.load(simple_configs['best_model_path']))
    simple_output: NetworkOutput = simpleModel.passInput(data_loader)
    # viewCAMLoss(data_loader, simple_output)
    #simple_cam = GradCAM(simple_output[0], IMG_RESIZE)
    #cam_map = overlayGradCAM(image_tensor , simple_cam.computeGradCAM())
    #showGradMap(cam_map, 'MODELO SIMPLES', "Simple_CNN_GradCAM.png")

    complete_configs = configs['complex_cnn']
    completeModel = cnnModel(
        device=device,
        filters_list=complete_configs['filters_list'],
        dropout=complete_configs['dropout'],
        batch_norm=complete_configs['batch_norm'],
        GAP=complete_configs['GAP'],
        best_model_path=complete_configs['best_model_path']
    )
    completeModel.model.load_state_dict(torch.load(complete_configs['best_model_path']))
    complete_output: NetworkOutput = completeModel.passInput(data_loader)
    # viewCAMLoss(data_loader, complete_output)
    #complete_cam = GradCAM(complete_output[0], IMG_RESIZE)
    #cam_map = complete_cam.computeGradCAM()
    #showGradMap(cam_map, 'MODELO COMPLEXO', "assets/Complete_CNN_GradCAM.png")

    _processCAM(
        data_loader=data_loader,
        out1=simple_output,
        out2=complete_output
    )

def viewCAMLoss(data_loader, model_output_list, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Varre os resultados buscando onde o modelo errou e plota o Grad-CAM dessas falhas.
    """
    from src.utils.grad_cam import GradCAM
    from src.utils.show_data import overlayGradCAM
    import matplotlib.pyplot as plt
    output_idx = 0
    
    for images_batch, labels_batch in data_loader:
        for i in range(images_batch.size(0)):
            image_tensor = images_batch[i:i+1]
            true_label = labels_batch[i].item()
            output = model_output_list[output_idx]
            predicted_label = output.predicted_class
            
            if true_label != predicted_label:
                if true_label == 1 and predicted_label == 0:
                    print(f"Falso Negativo encontrado no índice global {output_idx}!")
                    grad_cam = GradCAM(model_output=output, target_size=IMG_RESIZE)
                    heatmap = grad_cam.computeGradCAM()
                    final_img = overlayGradCAM(image_tensor, heatmap)
                    
                    plt.figure()
                    plt.imshow(final_img)
                    plt.title(f"ERRO - Real: {class_names[true_label]} | Predito: {class_names[predicted_label]}")
                    plt.axis('off')
                    plt.show()

                    return 
            
            output_idx += 1

if __name__ == "__main__":
    os.makedirs("src/models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    configs = _load_configs(CONFIG_PATH)
    predictionPipeline(DEVICE, DB_PATH, configs)