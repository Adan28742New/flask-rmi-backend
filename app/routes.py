from flask import Blueprint, request, jsonify, send_file
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import draw_segmentation_masks
from torchvision import transforms
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from app.model_loader import load_resnet50_model, load_vit_model, load_vgg16_tf_model, load_alexnet_model, load_yolov8_model, load_maskrcnn_model
import io
import base64

# --- Inicio de los logs de carga de modelos en app/routes.py ---
print("INFO: app/routes.py - Módulo de rutas cargado. Iniciando carga de modelos...")

try:
    # Cargar modelos
    # Log de inicio para ViT
    print("INFO: app/routes.py - Llamando a load_vit_model()...")
    vit_model = load_vit_model()
    vit_model.eval()
    print("INFO: app/routes.py - vit_model cargado y evaluado.")

    # Log de inicio para ResNet50
    print("INFO: app/routes.py - Llamando a load_resnet50_model()...")
    resnet_model = load_resnet50_model()
    resnet_model.eval()
    print("INFO: app/routes.py - resnet_model cargado y evaluado.")

    # Log de inicio para VGG16
    print("INFO: app.routes.py - Llamando a load_vgg16_tf_model() (TensorFlow)...")
    vgg16_tf_model = load_vgg16_tf_model()  # Asigna a una nueva variable para evitar conflictos
    print("INFO: app.routes.py - vgg16_tf_model (TensorFlow) cargado.")

    print("INFO: app.routes.py - Llamando a load_alexnet_model()...")
    alexnet_model = load_alexnet_model()

    print("INFO: app.routes.py - Llamando a load_yolov8_model()...")
    yolo_model = load_yolov8_model()

    print("INFO: app.routes.py - Llamando a load_maskrcnn_model()...")
    maskrcnn_model = load_maskrcnn_model()

    print("INFO: app/routes.py - ¡Todos los modelos cargados con éxito!")

except Exception as e:
    print(f"ERROR: app/routes.py - ¡Fallo crítico durante la carga de modelos! Error: {e}")

# --- Fin de los logs de carga de modelos en app/routes.py ---

main = Blueprint('main', __name__)

# Transformaciones para ResNet50
transformResNet50 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Transformaciones para Vision Transformer
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# TRANSFORMACIONES PARA ALEXNET
alexnet_transform = transforms.Compose([
    transforms.Resize((227, 227)),  # AlexNet usa 227x227
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predicción ResNet50
def predictResNet50(image: Image.Image, model):
    image = transformResNet50(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)
        prediction = (probs > 0.5).int().item()
    return prediction, probs.squeeze().item()

# Preprocesamiento y predicción ViT
def preprocess_vit_image(image_pil: Image.Image) -> torch.Tensor:
    image_gray = np.array(image_pil.convert("L"))
    image_filtered = cv2.medianBlur(image_gray, 5)
    image_equalized = cv2.equalizeHist(image_filtered)
    image_rgb = cv2.cvtColor(image_equalized, cv2.COLOR_GRAY2RGB)
    image_pil_rgb = Image.fromarray(image_rgb)
    return vit_transform(image_pil_rgb).unsqueeze(0)

def predict_vit(image: Image.Image, model):
    image_tensor = preprocess_vit_image(image)
    image_tensor = image_tensor.to(next(model.parameters()).device)  # Enviar al mismo device que el modelo
    with torch.no_grad():
        logits = model(image_tensor).logits
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        predicted_class = int(torch.argmax(logits, dim=1).item())
    return predicted_class, probabilities.tolist()


def predict_vgg16_tf(image_pil: Image.Image, model):
    # Convertir PIL Image a NumPy array para OpenCV
    image_np = np.array(image_pil)
    if image_np.ndim == 2:  # Escala de grises
        image_rgb_for_processing = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    else:  # RGB
        image_rgb_for_processing = image_np  # Ya es RGB
    imagen_limpia = cv2.medianBlur(image_rgb_for_processing, 5)  # Aplica filtro a RGB
    imagen_resized = cv2.resize(imagen_limpia, (224, 224))  # Redimensionar la imagen limpia
    imagen_array = img_to_array(imagen_resized)
    imagen_array = vgg16_preprocess_input(imagen_array)  # Preprocesamiento específico de VGG16
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir dimensión de batch
    pred = model.predict(imagen_array, verbose=0)[0][0]
    clase = 1 if pred > 0.5 else 0
    return clase, float(pred)  # Asegúrate de devolver float para jsonify


def predict_alexnet(image_pil: Image.Image, model):
    image_gray = np.array(image_pil.convert("L"))
    image_filtered = cv2.medianBlur(image_gray, 5)
    image_equalized = cv2.equalizeHist(image_filtered)
    image_rgb = cv2.cvtColor(image_equalized, cv2.COLOR_GRAY2RGB)
    image_processed_pil = Image.fromarray(image_rgb)
    image_tensor = alexnet_transform(image_processed_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs)
        prediction = (probs > 0.5).int().item()
    return prediction, probs.item()


def segment_image_with_yolo(image_pil: Image.Image, model, max_size=800, output_width=630):
    try:
        # Redimensionar imagen de entrada para ahorrar memoria
        original_size = image_pil.size
        if max(original_size) > max_size:
            # Calcular nuevo tamaño manteniendo proporción
            ratio = max_size / max(original_size)
            new_size = tuple(int(dim * ratio) for dim in original_size)
            image_pil = image_pil.resize(new_size, Image.LANCZOS)
        # Procesar con YOLO
        results = model.predict(source=image_pil, conf=0.5, save=False)
        r = results[0]
        img_np = np.array(image_pil.convert("RGB"))
        H, W = img_np.shape[:2]
        canvas = img_np.copy()
        # Aplicar velo azul
        blue_color = np.array([89, 38, 13], dtype=np.uint8)
        veil = np.full(canvas.shape, blue_color, dtype=np.uint8)
        canvas = cv2.addWeighted(canvas, 1, veil, 0.15, 0)
        # Procesar máscaras
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                resized_mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                red_color = np.array([25, 25, 230], dtype=np.uint8)
                red_overlay = np.zeros_like(canvas, dtype=np.uint8)
                red_overlay[resized_mask > 0] = red_color
                canvas = cv2.addWeighted(canvas, 1, red_overlay, 0.65, 0)

                # Contornos
                contours, _ = cv2.findContours(resized_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, (68, 68, 255), thickness=2)
        final_image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        # Redimensionar a tamaño fijo de salida
        final_pil = Image.fromarray(final_image_rgb)
        aspect_ratio = final_pil.height / final_pil.width
        output_height = int(output_width * aspect_ratio)
        resized_result = final_pil.resize((output_width, output_height), Image.LANCZOS)
        return resized_result
    except Exception as e:
        print(f"Error en segmentación: {e}")
        # Fallback: devolver imagen redimensionada sin procesar con tamaño fijo
        aspect_ratio = image_pil.height / image_pil.width
        output_height = int(output_width * aspect_ratio)
        return image_pil.resize((output_width, output_height), Image.LANCZOS)


def segment_image_with_maskrcnn(image_pil: Image.Image, model, threshold=0.5, top_k=1):
    img_tensor = F.to_tensor(image_pil).to('cpu')
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))[0]
    # --- LÓGICA CORREGIDA ---
    # Replicando fielmente el código de Jupyter:
    # 1. Tomar TODAS las máscaras propuestas por el modelo. El modelo las devuelve ordenadas por score.
    all_masks_raw = output['masks']
    # 2. Si no se propuso ninguna máscara, devolver la imagen original.
    if all_masks_raw.shape[0] == 0:
        print("INFO: Mask R-CNN - El modelo no propuso ninguna detección.")
        return image_pil
    # 3. Tomar solo las 'top_k' máscaras (la primera, en este caso).
    top_k_masks_raw = all_masks_raw[:top_k]
    # 4. Aplicar el 'threshold' directamente a los píxeles de la máscara para hacerla booleana.
    # Esto convierte los valores de confianza de los píxeles (ej. 0.8) en True o False.
    bool_masks_to_draw = top_k_masks_raw > threshold
    # -------------------------
    # Si después del threshold, la máscara quedó vacía, devolvemos la original para evitar un error.
    if not bool_masks_to_draw.any():
        print("INFO: Mask R-CNN - La máscara principal no superó el umbral de píxeles.")
        return image_pil
    # Dibujar la máscara booleana resultante sobre la imagen.
    img_to_draw_on = (img_tensor * 255).to(torch.uint8)
    result_tensor = draw_segmentation_masks(
        image=img_to_draw_on,
        masks=bool_masks_to_draw.squeeze(1),
        alpha=0.6,
        colors=["#FFD700"]
    )
    to_pil = ToPILImage()
    result_image_pil = to_pil(result_tensor)
    return result_image_pil



# Rutas
@main.route('/predict/resnet50', methods=['POST'])
def predict_resnet50():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        prediction, probs_resnet50 = predictResNet50(image, resnet_model)
        return jsonify({'prediction': prediction,
                        'probabilities':probs_resnet50})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/predict/vit', methods=['POST'])
def predict_vit_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        predicted_class, probs = predict_vit(image, vit_model)
        return jsonify({
            'prediction': predicted_class,
            'probabilities': probs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/predict/vgg16_tf', methods=['POST'])
def predict_vgg16_tf_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        image_file = request.files['image']
        # Abrir como RGB para compatibilidad con el modelo de Keras/VGG16 preprocess_input
        image = Image.open(image_file.stream).convert('RGB')
        predicted_class, pred_value = predict_vgg16_tf(image, vgg16_tf_model)
        return jsonify({
            'prediction_class': predicted_class,
            'prediction_value': pred_value  # Esto será un float
        })
    except Exception as e:
        print(f"ERROR: predict/vgg16_tf (TensorFlow) - {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/predict/alexnet', methods=['POST'])
def predict_alexnet_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream)  # No es necesario convertir a RGB aquí
        prediction, probability = predict_alexnet(image, alexnet_model)
        return jsonify({
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        print(f"ERROR: predict/alexnet - {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/predict/segment_yolo', methods=['POST'])
def predict_segment_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        image_file = request.files['image']
        image_pil = Image.open(image_file.stream)
        # Procesar imagen (redimensionada para procesamiento y salida fija)
        processed_image_pil = segment_image_with_yolo(image_pil, yolo_model)
        # Optimizar para respuesta
        img_io = io.BytesIO()
        processed_image_pil.save(img_io, 'JPEG', quality=85, optimize=True)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        print(f"ERROR: /predict/segment - {e}")
        return jsonify({'error': str(e)}), 500


@main.route('/predict/segment_mrcnn', methods=['POST'])
def predict_segment_mrcnn_route():
    if 'image' not in request.files: return jsonify({'error': 'No image uploaded'}), 400
    try:
        image_pil = Image.open(request.files['image'].stream).convert("RGB")
        processed_image_pil = segment_image_with_maskrcnn(image_pil, maskrcnn_model)
        img_io = io.BytesIO()
        processed_image_pil.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        import traceback
        print(f"ERROR: /predict/segment_mrcnn - {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500