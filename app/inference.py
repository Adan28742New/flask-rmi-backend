import torch
from torchvision import transforms
from PIL import Image

# Mismas transformaciones que usaste en el entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image, model):
    image = transform(image).unsqueeze(0)  # Añadir batch dim
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)
        prediction = (probs > 0.5).int().item()  # Umbral 0.5 para binario
    return prediction
