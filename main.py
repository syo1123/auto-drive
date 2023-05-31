import os
import torch
from PIL import Image
from Dataloader import dataloader
import numpy as np
from torch.optim import Adam
from model import Model
import torch.nn as nn
import torch.nn.functional as F
batch_size = 128
epochs = 30
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_loader = dataloader(batch_size)
model = Model()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr)
L2NormConst = 0.001

for epoch in range(epochs):
    losses = []
    print("epoch{}".format(str(epoch+1)))
    for img, label in data_loader:
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        label = label.unsqueeze(1)
        loss = F.mse_loss(output, label) + L2NormConst * sum(torch.norm(v) ** 2 for v in model.parameters())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(torch.mean(torch.tensor(losses)).item())

model_path = 'model.pth'
torch.save(model.to('cpu').state_dict(), model_path)
