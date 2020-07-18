# entry script for training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from dataset import *
from const import *

def loss_fn(S_before, S_after, stop_logits, S_true, S_true_lengths):
    '''
    S_before: [max_dec_len, B, d_spec]
    S_after: [max_dec_len, B, d_spec]
    stop_logits: [max_dec_len, B]
    S_true_pad: [St_max, B, d_spec]
    S_true_lengths: [B]
    ---
    loss: float
    '''
    max_dec_len = S_before.shape[0]
    len_to_pad = max_dec_len - S_true.shape[0]
    assert len_to_pad > 0
    S_true_ext = F.pad(S_true, [0, 0, 0, 0, 0, len_to_pad])
    length_mask_list = [torch.ones(true_len, device=device) for true_len in S_true_lengths]
    length_mask = F.pad(pad_sequence(length_mask_list), [0, 0, 0, len_to_pad])
    # length_mask: [max_dec_len, B]
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, length_mask, reduction='mean')
    length_mask = length_mask.unsqueeze(2) # [max_dec_len, B, 1]
    before_loss = (((S_true_ext-S_before)**2)*length_mask).mean()
    after_loss = (((S_true_ext-S_after)**2)*length_mask).mean()
    loss = stop_loss + before_loss + after_loss
    return loss

def train_epoch(net, loader, optimizer):
    running_n = 0
    running_loss = 0.0
    net.train()
    for batch in loader:
        S_pad, S_lengths, token_pad, token_lengths = batch
        S_pad = S_pad.to(device)
        token_pad = token_pad.to(device)
        optimizer.zero_grad()
        S_before, S_after, stop_logits = net(token_pad, token_lengths, S_pad, teacher_forcing=True)
        loss = loss_fn(S_before, S_after, stop_logits, S_pad, S_lengths)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        B = len(token_lengths)
        running_n += B
        running_loss += B*loss.item()
    return running_loss/running_n
