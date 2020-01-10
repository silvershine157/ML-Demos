import os
import argparse

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Transformer

import torch


def make_tensor(idx_list, pad_idx):
    '''
    idx_list: list of lists
    ---
    idx_tnsr: [B, L]
    pad_mask: [B, L]
    '''
    idx_tnsr = torch.LongTensor(idx_list)
    pad_mask = (idx_tnsr == pad_idx)
    return idx_tnsr, pad_mask

def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    # TODO: use these information.
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    # TODO: use these values to construct embedding layers
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)


    # TODO: build model
    n_blocks = 6
    d_model = 512
    vsize_src = src_vocab_size
    vsize_tar = tgt_vocab_size
    d_ff = 2048

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        # TODO: train
        for epoch in range(args.epochs):
            for src_batch, tgt_batch in train_loader:
                source, src_mask = make_tensor(src_batch, pad_idx)

                raise NotImplementedError
                pass

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
