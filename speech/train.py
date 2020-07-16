# entry script for training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from dataset import *
from model import *
from const import *

def loss_fn(S_pred, stop_logits, S_true, S_true_lengths):
    '''
    S_pred: [max_dec_len, B, d_spec]
    stop_logits: [max_dec_len, B]
    S_true_pad: [St_max, B, d_spec]
    S_true_lengths: [B]
    ---
    loss: float
    '''
    max_dec_len = S_pred.shape[0]
    len_to_pad = max_dec_len - S_true.shape[0]
    assert len_to_pad > 0
    S_true_ext = F.pad(S_true, [0, 0, 0, 0, 0, len_to_pad])
    length_mask_list = [torch.ones(true_len, device=device) for true_len in S_true_lengths]
    length_mask = F.pad(pad_sequence(length_mask_list), [0, 0, 0, len_to_pad])
    # length_mask: [max_dec_len, B]
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, length_mask, reduction='mean')
    spec_l1_errors = torch.abs(S_true_ext - S_pred)*length_mask.unsqueeze(2)
    spec_loss = spec_l1_errors.mean()
    loss = stop_loss + spec_loss
    return loss

def loss_fn_monitor(S_pred, stop_logits, S_true, S_true_lengths):
    '''
    S_pred: [max_dec_len, B, d_spec]
    stop_logits: [max_dec_len, B]
    S_true_pad: [St_max, B, d_spec]
    S_true_lengths: [B]
    ---
    loss: float
    '''
    max_dec_len = S_pred.shape[0]
    len_to_pad = max_dec_len - S_true.shape[0]
    assert len_to_pad > 0
    S_true_ext = F.pad(S_true, [0, 0, 0, 0, 0, len_to_pad])
    length_mask_list = [torch.ones(true_len, device=device) for true_len in S_true_lengths]
    length_mask = F.pad(pad_sequence(length_mask_list), [0, 0, 0, len_to_pad])
    # length_mask: [max_dec_len, B]
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, length_mask, reduction='mean')
    spec_l1_errors = torch.abs(S_true_ext - S_pred)*length_mask.unsqueeze(2)
    spec_loss = spec_l1_errors.mean()
    loss = stop_loss + spec_loss
    return loss, spec_l1_errors

def train_epoch(net, loader, optimizer):
    net.train()
    running_n = 0
    running_loss = 0.0
    for batch in loader:
        S_pad, S_lengths, token_pad, token_lengths = batch
        S_pad = S_pad.to(device)
        token_pad = token_pad.to(device)
        optimizer.zero_grad()
        enc_out = net.encoder(token_pad, token_lengths)
        S_pred, stop_logits = net.decoder(enc_out, S_pad, teacher_forcing=True)
        loss = loss_fn(S_pred, stop_logits, S_pad, S_lengths)
        loss.backward()
        optimizer.step()
        B = len(token_lengths)
        running_n += B
        running_loss += B*loss.item()
    return running_loss/running_n

def linspec_train():
    # train with linear spectrogram
    # later converted to audio using Griffin-Lim
    n_epochs = 1000
    save_every = 10
    loader = get_lj_loader(batch_size=1, limit=1)
    net = MiniTTS()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    net.to(device)
    for epoch in range(n_epochs):
        train_loss = train_epoch(net, loader, optimizer)
        print("Epoch {:d} train loss: {:g}".format(epoch, train_loss))
        if (epoch+1)%save_every==0:
            torch.save(net.state_dict(), 'data/ckpts/latest.sd')
