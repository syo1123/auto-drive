from PIL import Image
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.images[index]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


def dataloader(batch):
    data_dir = 'driving_dataset/'
    classes = os.listdir(data_dir)
    images = []
    with open('driving_dataset/data.txt', 'r') as f:
        text = f.read()
        text=[t.split(' ') for t in text.split('\n')]
        text.pop(-1)
    for img_name, label in text:
        img_path = data_dir+img_name
        image = Image.open(img_path)
        np_image = torch.tensor(np.array(image))
        images.append((img_path, torch.tensor(float(label)*np.pi/180)))

    transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = CustomDataset(images, transform=transform)
    batch_size = batch
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
