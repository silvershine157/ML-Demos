import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import librosa.display

from dataset import get_lj_loader
from const import *

class SingleDecoder(nn.Module):
    def __init__(self):
        super(SingleDecoder, self).__init__()
        d_spec = 552
        d_pre = 256
        pre_dropout = 0.5
        lstm_dropout = 0.1
        self.pre_net = nn.Sequential(
            nn.Linear(d_spec, d_pre),
            nn.ReLU(),
            nn.Dropout(pre_dropout),
            nn.Linear(d_pre, d_pre),
            nn.ReLU(),
            nn.Dropout(pre_dropout)
        )
        self.d_context = 128
        d_lstm_input = self.d_context+d_pre
        self.d_lstm_hidden = 1024
        self.lstm = nn.LSTM(input_size=d_lstm_input, hidden_size=self.d_lstm_hidden, num_layers=2, dropout=lstm_dropout)
        self.lintrans = nn.Linear(self.d_lstm_hidden+self.d_context, d_spec)
        self.postnet = nn.Sequential(
            nn.Conv1d(in_channels=d_spec, out_channels=512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=512, out_channels=d_spec, kernel_size=5, padding=2)
        )

    def forward(self, S_pad, teacher_forcing=True, before_post=False):
        L = S_pad.shape[0]
        B = S_pad.shape[1]
        h = torch.zeros((2, B, self.d_lstm_hidden), device=device)
        c = torch.zeros((2, B, self.d_lstm_hidden), device=device)
        dummy_context = torch.zeros((B, self.d_context), device=device)
        out_frame_list = []
        for t in range(L):
            if t == 0:
                in_frame = torch.zeros((B, 552), device=device)
            else:
                if teacher_forcing:
                    in_frame = S_pad[t-1, :, :]
                else:
                    in_frame = out_frame
            lstm_input = torch.cat([self.pre_net(in_frame), dummy_context], dim=1).unsqueeze(dim=0)
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))
            out_frame = self.lintrans(torch.cat([lstm_output.squeeze(dim=0), dummy_context], dim=1))
            out_frame_list.append(out_frame)
        S_pred = torch.stack(out_frame_list) # [L, B, d_spec]

        S_conv_in = S_pred.transpose(0, 1).transpose(1, 2) # [B, d_spec, L]
        S_conv_out = self.postnet(S_conv_in)
        S_residual = S_conv_out.transpose(1, 2).transpose(0, 1) # [L, B, d_spec]

        if before_post:
            return S_pred, S_pred + S_residual
        else:
            return S_pred + S_residual

def single_fit():
    net = SingleDecoder()
    loader = get_lj_loader(batch_size=1, limit=1)
    optimizer = torch.optim.Adam(net.parameters())
    n_epochs = 100000
    save_every = 20
    net.to(device)
    net.train()
    for epoch in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            S_pad, S_lengths, token_pad, token_lengths = batch
            S_pad = S_pad.to(device)
            S_before, S_after = net(S_pad, before_post=True)
            before_loss = ((S_before-S_pad)**2).mean()
            after_loss = ((S_after-S_pad)**2).mean()
            loss = before_loss + after_loss
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            optimizer.step()
            print("epoch {:d}, loss: {:g}, grad_norm: {:g}".format(epoch, loss.item(), grad_norm))
        if (epoch+1) % save_every == 0:
            torch.save(net.state_dict(), 'data/ckpts/single_{:d}.sd'.format(epoch))
            torch.save(net.state_dict(), 'data/ckpts/single_latest.sd')

def obtain_all_loss():
    net = SingleDecoder()
    net.to(device)
    net.eval()
    loader = get_lj_loader(batch_size=1, limit=1)
    ckpt_epochs = [i for i in range(19, 20000, 20)]
    losses = []
    with torch.no_grad():
        for epoch in ckpt_epochs:
            sd_path = 'data/ckpts/singlefit/single_{:d}.sd'.format(epoch)
            print('evaluating '+sd_path)
            net.load_state_dict(torch.load(sd_path))
            for batch in loader:
                S_pad, S_lengths, token_pad, token_lengths = batch
                S_pad = S_pad.to(device)
                S_pred = net(S_pad)
                loss = ((S_pred-S_pad)**2).mean()
                losses.append(loss.item())
    torch.save(losses, 'data/loss_single')

def cheat_loss():
    loader = get_lj_loader(batch_size=1, limit=1)
    for batch in loader:
        S_pad, S_lengths, token_pad, token_lengths = batch
        L = S_pad.shape[0]
        pred = S_pad[1:, :, :]
        target = S_pad[:-1, :, :]
        print(pred.shape)
        print(target.shape)
        mse = ((pred-target)**2).mean()
    print(mse)

def plot_all_loss():
    ckpt_epochs = np.array([i for i in range(19, 20000, 20)])
    losses = np.array(torch.load('data/loss_single'))
    print(len(ckpt_epochs))
    print(len(losses))
    plt.plot(ckpt_epochs, losses)
    plt.title("Single datapoint loss plot")
    plt.xlabel('iterations')
    plt.ylabel('spectrogram MSE')
    plt.savefig('data/display.png')

def obtain_samples():
    net = SingleDecoder()
    net.to(device)
    loader = get_lj_loader(batch_size=1, limit=1)
    #epochs = [i for i in range(999, 5000, 1000)]
    epochs = [21999]
    S_tf_list = []
    S_infer_list = []
    with torch.no_grad():
        for epoch in epochs:
            #sd_path = 'data/ckpts/singlefit/single_{:d}.sd'.format(epoch)
            #sd_path = 'data/ckpts/single_{:d}.sd'.format(epoch)
            sd_path = 'data/ckpts/single_latest.sd'
            #sd_path = 'data/ckpts/singlefit/single_latest.sd'
            print('evaluating '+sd_path)
            net.load_state_dict(torch.load(sd_path))
            net.eval()
            for batch in loader:
                S_pad, S_lengths, token_pad, token_lengths = batch
                S_pad = S_pad.to(device)
                S_tf = net(S_pad, teacher_forcing=True)
                S_infer = net(S_pad, teacher_forcing=False)
                S_tf_list.append(S_tf[:, 0, :].cpu().numpy())
                S_infer_list.append(S_infer[:, 0, :].cpu().numpy())

    for i in range(len(epochs)):
        S_tf = S_tf_list[i]
        librosa.display.specshow(S_tf.T)
        plt.savefig('data/S_tf_{:d}.png'.format(i))
        S_infer = S_infer_list[i]
        librosa.display.specshow(S_infer.T)
        plt.savefig('data/S_infer_{:d}.png'.format(i))