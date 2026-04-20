import torch
import torchvision
import torch.nn as nn

transformation_operation = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                                           ])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation_operation)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation_operation)

# Es del ejemplo
num_classes = 10
batch_size = 64
num_channels = 1 # Blanco y negro
img_size = 28 # En píxeles
patch_size = 7
num_patches = (img_size // patch_size) ** 2
embedding_dim = 64 # GPT-3 tiene 12288, pero es un modelo gigante
attention_heads = 4
transformer_blocks = 4 # Lo mismo que con los heads, es un número pequeño para que el modelo sea manejable
mlp_hidden_nodes = 128
learning_rate = 1e-3
num_epochs = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, attention_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_nodes),
            nn.GELU(),
            nn.Linear(mlp_hidden_nodes, embedding_dim),
        )
    def forward(self, x):
        residual1 = x
        x = self.layer_norm1(x)
        x = self.multihead_attention(x, x, x)[0]
        x = x + residual1

        residual2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual2

        return x
    
class MLP_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.mlp_head(x)

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.transformer_blocks = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])
        self.mlp_head = MLP_head()

    def forward(self, x):
        x = self.patch_embedding(x)
        B = x.size(0)
        class_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.transformer_blocks(x)
        x = x[:, 0]  # Solo el token [CLS]
        x = self.mlp_head(x)

        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_epoch = 0
    total_epoch = 0
    print(f'Epoch {epoch + 1}/{num_epochs}')
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)

        correct_epoch += correct
        total_epoch += labels.size(0)

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    epoch_accuracy = 100 * correct_epoch / total_epoch
    print(f'Epoch {epoch + 1} completed. Total Loss: {total_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

import matplotlib.pyplot as plt

model.eval()
images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)

# Move to CPU for plotting
images = images.cpu()
preds = preds.cpu()
labels = labels.cpu()

# Plot first 10 images
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f'Pred: {preds[i].item()}, True: {labels[i].item()}')
    plt.axis('off')
plt.tight_layout()
plt.show()