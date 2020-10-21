import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, d_z):
        super(Generator, self).__init__()
        self.d_z = d_z
        self.proj = nn.Linear(d_z, 4*4*1024)
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        conv_trans_in = self.proj(z).view(-1, 1024, 4, 4)
        conv_trans_out = self.conv_trans(conv_trans_in)
        return conv_trans_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 5, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 5, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 5, 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_layers = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.out_layers(conv_out.squeeze(3).squeeze(2))
        return out


def train():

    dataroot='../../data/celeba/'
    image_size = 64
    batch_size = 128
    d_z = 100
    ds = ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
           transforms.Resize(image_size),
           transforms.CenterCrop(image_size),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    bce = nn.BCELoss()
    gen = Generator(d_z).to(device)
    disc = Discriminator().to(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    iter_cnt = 0
    report_every = 10
    running_g_loss = 0.0
    running_d_loss = 0.0
    running_n = 0
    save_every = 1000

    gen.train()
    disc.train()
    n_epochs = 500
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(loader):

            real_x, _ = batch
            real_x = real_x.to(device)
            B = real_x.size(0)
            running_n += B

            # update discriminator
            disc_opt.zero_grad()
            z = torch.randn((B, d_z), device=device)
            fake_x = gen(z)
            real_out = disc(real_x)
            fake_out = disc(fake_x)
            real_disc_loss = bce(real_out, 1.0*torch.ones((B, 1), device=device))
            fake_disc_loss = bce(fake_out, 0.0*torch.ones((B, 1), device=device))
            disc_loss = real_disc_loss + fake_disc_loss
            disc_loss.backward()
            disc_opt.step()
            running_d_loss += B*disc_loss.detach().cpu().item()

            # update generator
            gen_opt.zero_grad()
            z = torch.randn((B, d_z), device=device)
            fake_x = gen(z)
            fake_out = disc(fake_x)
            gen_loss = bce(fake_out, torch.ones((B, 1), device=device))
            gen_loss.backward()
            gen_opt.step()
            running_g_loss += B*gen_loss.detach().cpu().item()

            iter_cnt += 1
            if iter_cnt % report_every == 0:
                avg_g_loss = running_g_loss/running_n
                avg_d_loss = running_d_loss/running_n
                print("Iter: {:d} | G loss: {:g} | D loss: {:g}".format(iter_cnt, avg_g_loss, avg_d_loss))
                running_g_loss = 0.0
                running_d_loss = 0.0
                running_n = 0
            if iter_cnt % save_every == 0:
                ckpt = {
                    "gen": gen.state_dict(),
                    "disc": disc.state_dict()
                }
                fname = "data/ckpts/ckpt_iter_{:07d}".format(iter_cnt)
                torch.save(ckpt,fname)
                print("saved to ", fname)


if __name__=='__main__':
    train()