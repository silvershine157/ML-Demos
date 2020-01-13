import os
import argparse

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Transformer

import torch
import torch.optim as optim
from const import *


def make_tensor(idx_list):
    '''
    idx_list: list of lists
    ---
    idx_tnsr: [B, L]
    pad_mask: [B, L]
    '''
    idx_tnsr = torch.LongTensor(idx_list)
    pad_mask = (idx_tnsr == pad_idx)
    return idx_tnsr, pad_mask


def run_epoch(net, loader, optimizer):
    running_loss = 0.0
    running_n = 0
    iter_cnt = 0
    for src_batch, tgt_batch in loader:
        B = len(src_batch)
        if optimizer:
            optimizer.zero_grad()
        source, src_mask = make_tensor(src_batch)
        target, tar_mask = make_tensor(tgt_batch)
        source = source.to(device)
        src_mask = src_mask.to(device)
        target = target.to(device)
        tar_mask = tar_mask.to(device)
        loss = net.loss(source, src_mask, target, tar_mask)
        if optimizer:
            loss.backward()
            optimizer.step()
        running_loss += B*loss.item()
        running_n += B
        iter_cnt += 1
        if (iter_cnt % 50 == 0):
            print("iter: ", iter_cnt)
    avg_loss = running_loss/running_n
    return avg_loss


def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    n_blocks = 6
    d_model = 512
    vsize_src = len(src_vocab)
    vsize_tar = len(tgt_vocab)
    d_ff = 2048
    net = Transformer(n_blocks, d_model, vsize_src, vsize_tar, d_ff)

    if not args.test:

        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)
        
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        best_valid_loss = 10.0
        for epoch in range(args.epochs):
            print("Epoch {0}".format(epoch))
            net.train()
            train_loss = run_epoch(net, train_loader, optimizer)
            print("train loss: {0}".format(train_loss))
            net.eval()
            valid_loss = run_epoch(net, valid_loader, None)
            print("valid loss: {0}".format(valid_loss))
            torch.save(net, 'data/ckpt/last_model')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(net, 'data/ckpt/best_model')
    else:
        # test
        net = torch.load('data/ckpt/last_model')
        net.to(device)
        net.eval()

        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        iter_cnt = 0
        for src_batch, tgt_batch in test_loader:
            source, src_mask = make_tensor(src_batch)
            source = source.to(device)
            src_mask = src_mask.to(device)
            res = net.decode(source, src_mask)
            pred_batch = res.tolist()
            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            pred += seq2sen(pred_batch, tgt_vocab)
            iter_cnt += 1
            print(pred_batch)

        with open('data/results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh data/results/pred.txt data/multi30k/test.de.atok')


def overfit_test(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    n_blocks = 6
    d_model = 512
    vsize_src = len(src_vocab)
    vsize_tar = len(tgt_vocab)
    d_ff = 2048
    net = Transformer(n_blocks, d_model, vsize_src, vsize_tar, d_ff)

    train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
    for src_batch, tgt_batch in train_loader:
        break
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss = run_batch(net, src_batch, tgt_batch, optimizer)
        print(loss)
    saved_info = (src_batch, tgt_batch, net)
    torch.save(saved_info, 'data/saved_info')
    source, src_mask = make_tensor(src_batch)
    source = source.to(device)
    src_mask = src_mask.to(device)
    res = net.decode(source, src_mask)
    pred_batch = res.tolist()
    print("Ground truth:")
    print(tgt_batch)
    print("Decoding result:")
    print(pred_batch)


def run_batch(net, src_batch, tgt_batch, optimizer):
        B = len(src_batch)
        if optimizer:
            optimizer.zero_grad()
        source, src_mask = make_tensor(src_batch)
        target, tar_mask = make_tensor(tgt_batch)
        source = source.to(device)
        src_mask = src_mask.to(device)
        target = target.to(device)
        tar_mask = tar_mask.to(device)
        loss = net.loss(source, src_mask, target, tar_mask)
        if optimizer:
            loss.backward()
            optimizer.step()
        return loss.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='data/multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=1000)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    #fast_overfit_test()
    #overfit_test(args)
    main(args)
