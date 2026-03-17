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

# Definimos las transformaciones
transforms = Compose([
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