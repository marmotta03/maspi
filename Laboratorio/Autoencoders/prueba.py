# -*- coding: utf-8 -*-
import torch
import modelo2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
################################################################

# Dataset personalizado para LPR
class LPRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    if fname.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.samples.append((os.path.join(label_dir, fname), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # escala de grises
        if self.transform:
            img = self.transform(img)
        return img, label

################################################################
# Funciones de entrenamiento y validación
def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss
################################################################

# Transformaciones
img_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # asegurar 1 canal
    transforms.Resize((28, 28)),                  # forzar 28x28
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
])

# Instanciar dataset
dataset = LPRDataset("datos", transform=img_transform)

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
trn_ds, val_ds = random_split(dataset, [train_size, val_size])

# Dataloaders
trn_dl = DataLoader(trn_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

################################################################
# Modelo
model = modelo2.AutoEncoder(10).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 10

train_loss = []
val_loss = []
for epoch in range(num_epochs):
    N = len(trn_dl)
    tloss = 0.
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch(data, model, criterion, optimizer)
        tloss += loss.item()
    train_loss.append(tloss / N)

    N = len(val_dl)
    vloss = 0.
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch(data, model, criterion)
        vloss += loss.item()
    val_loss.append(vloss / N)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

####################################################################
# Graficar las losses
plt.plot(train_loss, label="train loss")
plt.plot(val_loss, label="val loss")
plt.legend()
plt.title("Autoencoder training")
plt.show()

####################################################################
# Mostrar reconstrucciones
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1,2,figsize=(3,3)) 
    ax[0].imshow(im[0].cpu().numpy(), cmap='gray')
    ax[0].set_title('input')
    ax[1].imshow(_im[0].detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('reconstruction')
    plt.tight_layout()
    plt.show()