import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# IMPORTANTE: Debes tener las clases MultiHeadAttention, TransformerBlock 
# y ViTSegmentation definidas exactamente igual que en el otro archivo 
# o importarlas desde allí.

# --- CONFIGURACIÓN ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./vit_segmentacion_carvana.pth"
IMAGE_PATH = "../carvana-image-masking-challenge/test/0a0e3fb8f782_01.jpg" # <--- Pon aquí una imagen real

# 1. Reinstanciar la arquitectura
from ViT_2D import ViTSegmentation # Si el archivo se llama así
model = ViTSegmentation().to(DEVICE)

# 2. Cargar los pesos guardados
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Pesos del modelo cargados.")

# 3. Preparar la imagen
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img_path):
    img_orig = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_orig).unsqueeze(0).to(DEVICE) # Añadir dimensión de Batch [1, 3, 96, 96]
    
    with torch.no_grad():
        prediction = model(img_tensor)
        # La salida es [1, 1, 96, 96]. Quitamos dimensiones sobrantes.
        mask_pred = prediction.cpu().squeeze().numpy()
        
    # Visualización
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig)
    plt.title("Imagen de Entrada")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_pred > 0.5, cmap='gray') # Umbral de confianza
    plt.title("Máscara Predicha por ViT")
    
    plt.show()

# Ejecutar
predict(IMAGE_PATH)