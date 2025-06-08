import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from transformers import ViTForImageClassification, AutoConfig
from tensorflow.keras.models import load_model
from ultralytics import YOLO

def load_resnet50_model():
    # Log de inicio para ResNet50
    print("INFO: models_loader.py - Iniciando carga de modelo ResNet50...")
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "resnet50_mri_mejor_modelo.pth"))
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # Log de éxito para ResNet50
    print("INFO: models_loader.py - Modelo ResNet50 cargado exitosamente.")
    return model


def load_vit_model():
    # Log de inicio para ViT
    print("INFO: models_loader.py - Iniciando carga de modelo ViT...")

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "modelo_ViT_final_cv.pth"))
    config = AutoConfig.from_pretrained('google/vit-base-patch16-224', num_labels=2)
    model = ViTForImageClassification(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Log de éxito para ViT
    print("INFO: models_loader.py - Modelo ViT cargado exitosamente.")
    return model


def load_vgg16_tf_model():
    print("INFO: models_loader.py - Iniciando carga de modelo VGG16 (TensorFlow)...")
    # Ruta relativa a tu archivo .h5
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "mejor_modelo_cv_vgg16.h5"))

    # Cargar el modelo .h5 sin compilar para evitar warnings
    model = load_model(model_path, compile=False)
    print("INFO: models_loader.py - Modelo VGG16 (TensorFlow) cargado exitosamente.")
    return model


def load_alexnet_model():
    print("INFO: models_loader.py - Iniciando carga de modelo AlexNet...")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "alexnet_mejor_fold.pth"))
    model = models.alexnet(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("INFO: models_loader.py - Modelo AlexNet cargado exitosamente.")
    return model

#Modelos de Segmentación

def load_yolov8_model():
    print("INFO: models_loader.py - Iniciando carga de modelo YOLOv8...")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_yolo.pt"))
    model = YOLO(model_path)
    print("INFO: models_loader.py - Modelo YOLOv8 cargado exitosamente.")
    return model


def get_mrcnn_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # El resto de la arquitectura se mantiene igual
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)
    model.transform.min_size = (300,)
    model.transform.max_size = 512
    return model


# La función de carga no necesita cambios, ya que depende de la anterior
def load_maskrcnn_model():
    print("INFO: models_loader.py - Iniciando carga de modelo Mask R-CNN...")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_maskrcnn_tunned.pth"))
    device = torch.device('cpu')
    # Ahora esta llamada usará la función corregida
    model = get_mrcnn_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("INFO: models_loader.py - Modelo Mask R-CNN cargado exitosamente en CPU.")
    return model