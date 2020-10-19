import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, d_z):
        super(Generator, self).__init__()
        self.d_z = d_z
        self.proj = nn.Linear(d_z, 3*3*d_z)
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(d_z, 128, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2),
        )

    def forward(self, z):
        conv_trans_in = self.proj(z).view(-1, self.d_z, 3, 3)
        conv_trans_out = self.conv_trans(conv_trans_in)
        return conv_trans_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2),
        )
        self.out_layers = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.out_layers(conv_out.mean(dim=[2, 3]))
        return out

def test():
    dataroot='../../data/celeba/'
    image_size = 64
    batch_size = 32
    d_z = 10
    ds = ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
           transforms.Resize(image_size),
           transforms.CenterCrop(image_size),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    disc = Discriminator()
    loader = DataLoader(ds, batch_size=batch_size)
    for batch in loader:
        real_x, y = batch
        disc_out = disc(real_x)
        print(disc_out.shape)
        break

    gen = Generator(d_z)
    z = torch.randn(batch_size, d_z)
    fake_x = gen(z)
    print(fake_x.shape)

test()