import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10, dim=128, depth=6, heads=8, mlp_dim=256):
        super().__init__()
        
        # 1. Parcheado y Proyección Lineal
        # En lugar de recortar la imagen a mano, usamos una Convolución 
        # con un salto (stride) igual al tamaño del parche.
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (image_size // patch_size) ** 2
        
        # 2. Token de Clasificación [CLS] y Posiciones
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # 3. El Bloque Transformer (Repetido varias veces)
        # Usamos la implementación oficial de PyTorch para mayor claridad
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, 
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Cabeza de Clasificación
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Transformar imagen (B, 3, H, W) -> (B, dim, H/P, W/P)
        x = self.patch_embed(img) 
        
        # Aplanar los parches: (B, dim, 14, 14) -> (B, 196, dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Añadir el token [CLS] al principio de la secuencia
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Sumar la información de posición
        x += self.pos_embedding
        
        # Pasar por el Transformer
        x = self.transformer(x)
        
        # Extraer solo el primer token (el [CLS]) para clasificar
        x = x[:, 0]
        
        return self.mlp_head(x)

# Ejemplo de uso:
# modelo = SimpleViT()
# imagen = torch.randn(1, 3, 224, 224)
# prediccion = modelo(imagen)