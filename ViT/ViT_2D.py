import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# ==========================================
# 1. COMPONENTES DEL TRANSFORMER
# ==========================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.projection(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 2. ARQUITECTURA DE SEGMENTACIÓN (ViT-SEG)
# ==========================================

class ViTSegmentation(nn.Module):
    def __init__(self, img_size=96, patch_size=16, embed_dim=128, depth=4, num_heads=8):
        super().__init__()
        self.grid_size = img_size // patch_size
        
        # Encoder
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size**2, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # Decoder (Agrandar la imagen)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=4), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),      
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)
        return self.decoder(x)

# ==========================================
# 3. DATASET Y DATALOADER
# ==========================================

class CarvanaSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=96):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        # Ajustamos extensión de máscara (Carvana usa _mask.gif)
        mask_name = img_name.replace(".jpg", "_mask.gif")
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", (self.img_size, self.img_size), 0)
            
        return self.transform(img), self.transform(mask)

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "../carvana-image-masking-challenge/train_small"
MASK_DIR = "../carvana-image-masking-challenge/train_small_masks"

if __name__ == "__main__":
    # 1. Cargar datos
    dataset = CarvanaSegmentationDataset(IMG_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Modelo
    model = ViTSegmentation().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    print(f"🚀 Iniciando entrenamiento en {DEVICE}...")
    epochs = 400
    history_loss = []
    ruta_modelo = "./vit_segmentacion_carvana.pth"

    # 3. Un pequeño bucle de prueba
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            preds = model(imgs)
            loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if(epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), ruta_modelo+f"_epoch_{epoch}.pth")
            print(f"💾 Modelo guardado en: {ruta_modelo}_epoch_{epoch}.pth después de {epoch+1} épocas.")
        
        epoch_loss = running_loss / len(loader)
        history_loss.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    # ==========================================
    # 4. GRÁFICA DE EVOLUCIÓN
    # ==========================================
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history_loss, marker='o', color='b', label='Pérdida (Loss)')
    plt.title('Evolución del Entrenamiento - Vision Transformer')
    plt.xlabel('Época')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica para tu memoria del TFG
    plt.savefig('evolucion_entrenamiento.png')
    print("✅ Gráfica guardada como 'evolucion_entrenamiento.png'")

    # Mostrar en pantalla
    plt.show()

    ruta_modelo = "./vit_segmentacion_carvana.pth"
    torch.save(model.state_dict(), ruta_modelo)
    print(f"💾 Modelo guardado exitosamente en: {ruta_modelo}")