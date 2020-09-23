import argparse
import os
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=100000, type=int)
parser.add_argument('-e', '--epochs', default=5, type=int)
parser.add_argument('-w', '--workers', default=2, type=int)
parser.add_argument('-l', '--lr', default=1e-3, type=float)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        d_model = 1024
        self.layers = nn.Sequential(
            nn.Linear(28*28, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 10),
        )

    def forward(self, x):
        """
        x: [B, 28, 28]
        ---
        y_logits: [B, 10]
        """
        B = x.size(0)
        y_logits = self.layers(x.view((B, -1)))
        return y_logits

def main():
    ds = torchvision.datasets.MNIST('../../data', download=True)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

def main_worker(rank, size, args):

    print(f"Main worker GPU:{rank}")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', rank=rank, world_size=size)

    model = Net()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    transform = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.MNIST('../../data', transform=transform)

    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        acc_loss = 0.0
        acc_B = 0
        for batch_idx, batch in enumerate(loader):
            x, y = batch
            y_logits = model(x)
            loss = criterion(y_logits, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            B = len(y)
            acc_B += B
            acc_loss += B * loss.detach().item()
        avg_loss = acc_loss/acc_B
        print(f'Epoch {epoch} rank {rank} avgloss {avg_loss}')

if __name__ == '__main__':
    main()
