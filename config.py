IN_CHANNELS = 1
NUM_CLASSES = 1
BACKGROUND_AS_CLASS = False

TRAIN_CUDA = True
# Peso de la clase minoritária/positva (tumor) en la función de pérdida BCE. 
#   Se puede ajustar según el desequilibrio de clases.
BCE_WEIGHT = 250 


# Transformaciones para CARGAR ambos (imagen y máscara real).
# - Compose: Permite encadenar varias transformaciones en un solo paso.
# - LoadImaged: Carga los archivos .nii y los convierte en tensores.
# - EnsureChannelFirstd: Asegura que los datos tengan la forma [Canal, D, H, W].
# - ScaleIntensityd: Normaliza la intensidad de las imágenes (no se aplica 
#       a las máscaras).
# - RandCropByPosNegLabeld: Recorta aleatoriamente parches de la imagen, 
#       dando prioridad a las regiones con etiquetas (pos) sobre las 
#       sin etiquetas (neg).
# - ToTensord: Convierte los datos a tensores de PyTorch.
# - Lambdad: Permite aplicar una función personalizada a los datos.
# - RandRotated, RandFlipd, RandGaussianNoised: Transformaciones de 
#       aumento de datos (data augmentation) para hacer el modelo más variado.
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    RandCropByPosNegLabeld, ToTensord, Lambdad,
    RandRotated, RandFlipd, RandGaussianNoised)

# ------- Definimos las transformaciones -------
train_transform = Compose([
    LoadImaged(keys=["image", "label"]),           # Carga ambos NIfTI
    EnsureChannelFirstd(keys=["image", "label"]),  # Asegura [Canal, D, H, W]
    ScaleIntensityd(keys=["image"]),              # Normaliza la imagen (la máscara no se normaliza)
    # Aquí binarizamos: cualquier cosa > 0 es 1. Así el loop queda limpio.
    Lambdad(keys=["label"], func=lambda x: (x > 0).float()),
    # Volteo aleatorio (espejo) en el eje horizontal (probabilidad 50%)
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    # Rotación aleatoria +-15 grados (0.26 radianes) en los ejes H y W
    # Importante: usamos 'bilinear' para imagen y 'nearest' para la máscara
    RandRotated(
        keys=["image", "label"], 
        range_x=0.1, range_y=0.1, range_z=0.1, 
        prob=0.2, 
        mode=("bilinear", "nearest")
    ),
    # Ruido Gaussiano (simula interferencias en la captura de la RM)
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
    # Recorta 4 cubos de 64x64x64 donde hay etiquetas
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(64, 64, 64),
        pos=7, neg=1, 
        num_samples=4 
    ),
    ToTensord(keys=["image", "label"])
]) 

# Usamos las mismas claves para que la máscara real esté alineada con la imagen
predict_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])
# ------- Fin de las transformaciones -------

# ------- Configuración del dispositivo, modelo, optimizador y función de pérdida -------
# Importamos el modelo UNet3D que definimos en unet.py. Este modelo es una
# arquitectura de red neuronal convolucional diseñada 
# para segmentación de imágenes 3D.
from unet3D_parts import UNet3D as UNet3D_v2
# La función de pérdida con pesos para manejar el desbalance de clases. Aplica una importancia
#   distinta a las distintas clases (fondo vs tumor) para que la red no se "olvide" de aprender a segmentar el tumor,
#   que es la clase minoritaria.# Librería principal de Deep Learning. Proporciona los tensores, 
# operaciones matemáticas y la funcionalidad de GPU.
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet3D_v2(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)

# Optimizador (ajusta los pesos de la red)
# Deberia porbar con AdamW, pero Adam es un buen punto de partida
# Contiene los algoritmos que "aprenden", como Adam. Es el que ajusta los
# pesos de la red durante el entrenamiento.
import torch.optim as optim
optimizer = optim.Adam(model.parameters())

# alpha=0.2 (peso a los falsos negativos)
# beta=0.8 (peso a los falsos positivos -> ¡Esto es lo que evita que sea vaga!)
# Para calcular la métrica de TverskyLoss, que es una medida de solapamiento 
#   entre la máscara real y la predicha. Nos dico qué tan lejos está
#   nuestra predicción de la realidad.
from monai.losses import TverskyLoss
loss_function = TverskyLoss(sigmoid=True, alpha=0.5, beta=0.5)

# Contiene los bloques de construcción de las redes neuronales (capas, 
# activaciones, etc.)
import torch.nn as nn
pos_weight = torch.tensor([BCE_WEIGHT], dtype=torch.float32).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Scheduler: Reduce el LR cuando el loss se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.5, verbose=True)    

# Para convertir la salida de la red (que es un valor continuo entre 0 y 1)
#   en una máscara binaria (0 o 1) usando un umbral (NO LO USO).
# from monai.transforms import AsDiscrete
# post_pred = AsDiscrete(threshold=0.5) # Umbral de 0.5 para ser más estrictos
# ------- Fin de la configuración -------