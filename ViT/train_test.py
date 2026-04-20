import torch
from monai import data, transforms

# 1. Las transformaciones DEBEN ser de MONAI para datos médicos
transformation_operation = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),           # 1. Carga del disco
    transforms.EnsureChannelFirstd(keys=["image"]),  # 2. Arregla las dimensiones (C, D, H, W)
    transforms.Orientationd(keys=["image"], axcodes="RAS"), # 3. Opcional: Estandariza la orientación
    transforms.ToTensord(keys=["image"])             # 4. Convierte a PyTorch
])

# 2. La data DEBE ser una lista de diccionarios
# (Asumiendo que dentro de esa carpeta está el archivo flair)
archivos = [
    {"image": "./data/BraTS2021_00000/BraTS2021_00000_flair.nii.gz"}
]

# 3. El Dataset de MONAI solo lleva 'data' y 'transform'
train_dataset = data.Dataset(data=archivos, transform=transformation_operation)

# Para ver la pinta que tiene el primer elemento:
primer_ejemplo = train_dataset[0]
print(primer_ejemplo["image"].shape) 
# Salida esperada: torch.Size([1, 240, 240, 155]) <- Esto es un volumen 3D real