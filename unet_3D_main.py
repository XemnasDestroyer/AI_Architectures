# Interactúa con el sistema operativo. Se usara principialmente para
# verificar si los archivos .nii existen antes de intentar abrirlos.
import os

# Para pasar argumentos desde la terminal (-help, train, predict).
import argparse 

# Librería para manejo de matrices.
import numpy as np

# Librería principal de Deep Learning. Proporciona los tensores, 
# operaciones matemáticas y la funcionalidad de GPU.
import torch
# Contiene los bloques de construcción de las redes neuronales (capas, 
# activaciones, etc.)
import torch.nn as nn
# Contiene los algoritmos que "aprenden", como Adam. Es el que ajusta los
# pesos de la red durante el entrenamiento.
import torch.optim as optim
# Torchsummary es una herramienta para mostrar un resumen de la arquitectura 
#   del modelo, incluyendo el número de parámetros y la forma de las salidas 
#   de cada capa.    
from torchsummary import summary
# CrossEntropyLoss nos permitirá cambiar el peso de cada clase para
#   manejar el desbalance de clases (muchos más píxeles de fondo 
#   que de tumor).
from torch.nn import CrossEntropyLoss

# Biblioteca estándar para abrir, leer y escribir archivos de 
# imágenes médicas en formato NIfTI (.nii o .nii.gz)
import nibabel as nib

# Importamos el modelo UNet3D que definimos en unet.py. Este modelo es una
# arquitectura de red neuronal convolucional diseñada 
# para segmentación de imágenes 3D.
from unet2D_parts import UNet3D
from unet3D_parts import UNet3D as UNet3D_v2

# - El Dataset organiza los diccioarios en imágenes
# - El DataLoader se encarga de crear los batches, mezclar los datos y 
#       cargarlos en GPU eficientemente
from monai.data import Dataset, DataLoader
# Para calcular la métrica de TverskyLoss, que es una medida de solapamiento 
#   entre la máscara real y la predicha. Nos dico qué tan lejos está
#   nuestra predicción de la realidad.
from monai.losses import TverskyLoss
# Para calcular la métrica de Dice, que es otra medida de solapamiento
#   entre la máscara real y la predicha.
from monai.metrics import DiceMetric
# Para convertir la salida de la red (que es un valor continuo entre 0 y 1)
#   en una máscara binaria (0 o 1) usando un umbral.
from monai.transforms import AsDiscrete
# Para realizar la inferencia por ventanas deslizantes, que es una técnica
#   que permite segmentar imágenes grandes dividiéndolas en partes más pequeñas.
#   Luego se encarga de ensamblar las predicciones de cada parte para obtener
#   la segmentación completa.
from monai.inferers import sliding_window_inference

# Librería para visualizar los resultados en tiempo real durante 
#   el entrenamiento.
import matplotlib.pyplot as plt

# Configuración específica de la red neuronal
from config import (BACKGROUND_AS_CLASS, IN_CHANNELS, NUM_CLASSES, BCE_WEIGHT, transforms)


def load_medical_volume(path):
    img_path = path + "brain_t1.nii"
    mask_path = path + "final_seg.nii"

    if not os.path.exists(img_path):
        raise ValueError("No se encuentra el archivo de imagen .nii. Revisa la ruta.")
    if not os.path.exists(mask_path):
        raise ValueError("No se encuentra el archivo de la máscara .nii. Revisa las rutas.")
    
    return {"image": img_path, "label": mask_path}

def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.exists(filename):
        print(f"--> Cargando Checkpoint desde {filename}...")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    print(f"--> No se encontró el checkpoint en {filename}. Empezando desde cero.")
    return 0

def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"--> Checkpoint guardado en época {epoch}")

def train(device, data_path, model_path, NUM_EPOCHS):
    global NUM_CLASSES
    if BACKGROUND_AS_CLASS: NUM_CLASSES +=1

    # Llamada a la funcion que carga la imagen y la máscara, y devuelve un diccionario con las rutas de ambos.
    data_dict = load_medical_volume(data_path)   

    # DataDict va entre corchetes porque el Dataset de MONAI espera una lista de diccionarios, aunque aquí solo tenemos uno
    train_ds = Dataset(data=[data_dict], transform=transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Inicialización del modelo
    model = UNet3D_v2(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)

    # Optimizador (ajusta los pesos de la red)
    # Deberia porbar con AdamW, pero Adam es un buen punto de partida
    optimizer = optim.Adam(model.parameters())

    # alpha=0.2 (peso a los falsos negativos)
    # beta=0.8 (peso a los falsos positivos -> ¡Esto es lo que evita que sea vaga!)
    loss_function = TverskyLoss(sigmoid=True, alpha=0.5, beta=0.5)

    # La función de pérdida con pesos para manejar el desbalance de clases. Aplica una importancia
    #   distinta a las distintas clases (fondo vs tumor) para que la red no se "olvide" de aprender a segmentar el tumor,
    #   que es la clase minoritaria.
    pos_weight = torch.tensor([BCE_WEIGHT], dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    # Scheduler: Reduce el LR cuando el loss se estanca
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.5, verbose=True)    

    # Cargar progreso si existe
    start_epoch = load_checkpoint(model, optimizer, scheduler, model_path)
    
    print(f"Iniciando entrenamiento en {device}...")

    # Para convertir la salida de la red en una máscara binaria (0 o 1)
    post_pred = AsDiscrete(threshold=0.5) # Umbral de 0.5 para ser más estrictos

    # Listas para las gráficas de progreso
    history_loss = []
    history_dice = []
    
    print(f"Iniciando desde época {start_epoch+1} hasta {start_epoch + NUM_EPOCHS}")

    for epoch in range(NUM_EPOCHS):
        torch.cuda.empty_cache() # Libera memoria no utilizada en GPU al inicio de cada época
        model.train() # Activa el modo entrenamiento de la red
        epoch_loss = 0

        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            if epoch == 0:
                print(f"Valores únicos en la máscara: {torch.unique(labels)}")
                print(f"Forma del tensor de entrada: {inputs.shape}")

            optimizer.zero_grad() # Limpieza de gradientes viejos
            outputs = model(inputs) # Obtiene la predicción de la red para los inputs actuales
            
            # loss = loss_function(outputs, labels)
            loss_tversky = loss_function(outputs, labels)
            loss_bce = criterion(outputs, labels)
            loss = loss_tversky + loss_bce
            loss.backward() # Calcula los gradientes de la pérdida con respecto a los pesos de la red

            optimizer.step() # Actualiza los pesos de la red usando los gradientes calculados

            epoch_loss += loss.item()

            # --- VISUALIZACIÓN EN TIEMPO REAL ---
            # if (epoch + 1) % 20 == 0 and idx == 0:
            #     visualizar_progreso(inputs, labels, outputs, epoch + 1)

        # Calculamos el promedio del loss en esta época
        avg_loss = epoch_loss / len(train_loader)

        # Guardamos en el historial para las gráficas
        history_loss.append(avg_loss)

        # Activamos el scheduler para ajustar el LR si el loss se estanca
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"EPOCH {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Guardar cada 50 épocas o al final
        # if (epoch + 1) % 50 == 0:
        #     save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path)

    # Guardar la red neuronal
    save_checkpoint(model, optimizer, scheduler, epoch+1, model_path)
    print("Modelo guardado como unet3d_brain_model.pth")

    print("\n¡Entrenamiento finalizado!")

    visualize_progress(history_loss, history_dice)

    samples = transforms(data_dict)
    print(f"Forma del tensor de imagen: {samples[3]['image'].shape}")

def predict(model_path, image_path, mask_path):
    # 1. Configurar dispositivo y modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)

    # 2. Cargar los pesos y poner en modo evaluación
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    
    # Usamos las mismas claves para que la máscara real esté alineada con la imagen
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"])
    ])

    # Preparar los datos
    data = transforms({"image": image_path, "label": mask_path})
    
    # Preparamos el tensor para la red (añadimos dimensión de batch)
    input_tensor = data["image"].unsqueeze(0).to(device) # [1, 1, D, H, W]
    
    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=input_tensor, 
            roi_size=(64, 64, 64), 
            sw_batch_size=4, 
            predictor=model,
            overlap=0.5
        )

    # Sigmoid + Umbral de 0.5 para binarizar
    prediction_binaria = (torch.sigmoid(prediction) > 0.5).float()

    # Convertir el tensor de predicción a un array de numpy
    # Quitamos las dimensiones extras de Batch y Canal para dejarlo en [D, H, W]
    pred_mask = prediction_binaria[0, 0].cpu().numpy() # [D, H, W]

    # Cargar la imagen original para copiar sus "metadatos"
    # Esto es vital para que 3D Slicer sepa dónde colocar la máscara
    original_nifti = nib.load(image_path)
    header = original_nifti.header
    affine = original_nifti.affine

    # Crear el nuevo objeto NIfTI
    # Importante: Asegúrate de que el tipo de dato sea compatible (int16 o uint8 suele bastar)
    pred_nifti = nib.Nifti1Image(pred_mask.astype("uint8"), affine, header)

    # Guardar en el disco
    output_path = "./assets/data/3d/brain/prediccion_3d_final.nii"
    nib.save(pred_nifti, output_path)

    print(f"¡Segmentación guardada con éxito en: {output_path}!")
    



def visualize_progress(lista_loss, lista_dice):
    # Crear una figura con dos columnas
    plt.figure(figsize=(14, 5))

    # Gráfica de la Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(lista_loss) + 1), lista_loss, label='Pérdida (Tversky)', color='tab:red', linewidth=2)
    plt.title('Progreso del Error (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Valor de Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Gráfica de la Métrica (Dice)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(lista_dice) + 1), lista_dice, label='Precisión (Dice %)', color='tab:blue', linewidth=2)
    plt.title('Evolución del Acierto (Dice Score)')
    plt.xlabel('Época')
    plt.ylabel('Porcentaje (%)')
    plt.ylim(0, 100) # El Dice va de 0 a 100
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Guardar la imagen en el disco
    plt.tight_layout()
    plt.savefig("entrenamiento_stats.png")
    print("\n[INFO] Gráficas guardadas como 'entrenamiento_stats.png'")
    plt.show()

# --- Test de dimensionamiento ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de UNet 3D para Cerebro")

    parser.add_argument("mode", choices=["train","predict"], 
                        help="Selecciona 'train' para entrenar o " \
                        "'predict' para inferencia")

    # Opcional: --epochs (se puede abreviar como -e)
    # type=int asegura que el dato sea un número
    # default=100 es el valor si el usuario no pone nada
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Número de épocas para el entrenamiento (por defecto: 100)")

    args = parser.parse_args()
    
    # Selección del dispositivo (GPU si está disponible, sino CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "predict":
        predict(model_path="unet3d_brain_model.pth", 
                image_path="./assets/data/3d/brain/00000057_brain_t1.nii",
                mask_path="./assets/data/3d/brain/00000057_final_seg.nii")
    
    elif args.mode == "train":
        # El numero de épocas es el número de veces que la red verá todo el conjunto de datos (en este caso, los parches generados)
        NUM_EPOCHS = args.epochs  # Empieza con pocas para probar
        data_path = "./assets/data/3d/brain/00000057_" # La ruta donde están los archivos .nii
        model_path = "./unet3d_brain_model_16032026.pth" # Ruta donde se guardará el modelo entrenado (checkpoint)
        train(device, data_path, model_path, NUM_EPOCHS)   