import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 5)
        #self.net2 = nn.Linear(10, 5)

    def forward(self, x, switch=True):
        print('x shape: ', x.shape)
        if switch:
            return self.net1(x)
        else:
            return self.net2(x)

def demo_basic(rank, world_size):
    print(f'DDP on rank {rank}')
    setup(rank, world_size)

    is_distributed=True

    dummy_x = torch.randn(20, 10)
    dummy_y = torch.randn(20, 5)
    ds = TensorDataset(dummy_x, dummy_y)
    sampler = DistributedSampler(ds) if is_distributed else None
    loader = DataLoader(ds, shuffle=(sampler is None), sampler=sampler, batch_size=4)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    n_epochs = 10
    for epoch in range(n_epochs):
        for batch in loader:
            x, y = batch
            optimizer.zero_grad()
            outputs = ddp_model(x)
            loss_fn(outputs, y.cuda()).backward()
            optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

if __name__=='__main__':
    run_demo(demo_basic, 2)


