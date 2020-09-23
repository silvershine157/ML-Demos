import sys
import argparse
import time
import os
import torch


def main(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        print('Using GPU:', args.local_rank)

    B = 64
    d_in = 1024
    d_out = 16
    model = torch.nn.Linear(d_in, d_out).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if args.distributed:
        model = DistributedDataParallel(model)

    loss_fn = torch.nn.MSELoss()

    x = torch.randn(B, d_in, device='cuda')
    y = torch.randn(B, d_out, device='cuda')

    for t in range(1000):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=1, type=int)
    args = parser.parse_args()
    main(args)
