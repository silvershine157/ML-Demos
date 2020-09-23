import argparse
import os
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=100000, type=int)
parser.add_argument('-e', '--epochs', default=5, type=int)
parser.add_argument('-w', '--workers', default=2, type=int)

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 5)
        self.net2 = nn.Linear(10, 5)
        d_model = 5000
        self.bignet = nn.Sequential(
            nn.Linear(10, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 5)
        )

    def forward(self, x):
        return self.bignet(x)

def main():
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

def main_worker(rank, size, args):

    print(f"Main worker GPU:{rank}")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='gloo', rank=rank, world_size=size)

    model = ToyModel()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    criterion = nn.MSELoss().cuda(rank)
    optimizer = torch.optim.Adam(model.parameters())
    
    torch.manual_seed(0)
    dummy_x = torch.randn(100000, 10)
    dummy_y = torch.randn(100000, 5)

    ds = TensorDataset(dummy_x, dummy_y)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(loader):
            x, y = batch
            output = model(x)
            loss = criterion(output, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()
