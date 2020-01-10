import os
import argparse

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Transformer

import torch
import torch.optim as optim

# TODO: use these information.
sos_idx = 0
eos_idx = 1
pad_idx = 2
max_length = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train_epoch(net, loader, optimizer):
    running_loss = 0.0
    running_n = 0
    iter_cnt = 0
    for src_batch, tgt_batch in loader:
        B = len(src_batch)
        optimizer.zero_grad()
        source, src_mask = make_tensor(src_batch)
        target, tar_mask = make_tensor(tgt_batch)
        source = source.to(device)
        src_mask = src_mask.to(device)
        target = target.to(device)
        tar_mask = tar_mask.to(device)
        loss = net.loss(source, src_mask, target, tar_mask)
        loss.backward()
        optimizer.step()
        running_loss += B*loss.item()
        running_n += B
        iter_cnt += 1
        if (iter_cnt % 10 == 0):
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

        net.train()
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        for epoch in range(args.epochs):
            avg_loss = train_epoch(net, train_loader, optimizer)
            print(avg_loss)
            # TODO: validation
            for src_batch, tgt_batch in valid_loader:
                pass
    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        for src_batch, tgt_batch in test_loader:
            # TODO: predict pred_batch from src_batch with your model.
            pred_batch = tgt_batch

            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            pred += seq2sen(pred_batch, tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='data/multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    main(args)
