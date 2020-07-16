import torch
import torch.nn as nn

from dataset import get_lj_loader
from const import *

class SingleDecoder(nn.Module):
    def __init__(self):
        super(SingleDecoder, self).__init__()
        d_spec = 552
        d_pre = 256
        self.pre_net = nn.Sequential(
            nn.Linear(d_spec, d_pre),
            nn.ReLU(),
            nn.Linear(d_pre, d_pre),
            nn.ReLU()
        )
        self.d_context = 128
        d_lstm_input = self.d_context+d_pre
        self.d_lstm_hidden = 1024
        self.lstm = nn.LSTM(input_size=d_lstm_input, hidden_size=self.d_lstm_hidden, num_layers=2)
        self.lintrans = nn.Linear(self.d_lstm_hidden+self.d_context, d_spec)
        self.postnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(5, 1), padding=(2,0)),
            nn.Tanh(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 1), padding=(2,0)),
            nn.Tanh(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 1), padding=(2,0)),
            nn.Tanh(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 1), padding=(2,0)),
            nn.Tanh(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(5, 1), padding=(2,0)),
        )

    def forward(self, S_pad):
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
                in_frame = S_pad[t-1, :, :]
            lstm_input = torch.cat([self.pre_net(in_frame), dummy_context], dim=1).unsqueeze(dim=0)
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))
            out_frame = self.lintrans(torch.cat([lstm_output.squeeze(dim=0), dummy_context], dim=1))
            out_frame_list.append(out_frame)
        S_pred = torch.stack(out_frame_list) # [L, B, d_spec]

        S_conv_in = S_pred.transpose(0, 1).unsqueeze(1) # [B, 1, L, d_spec]
        S_conv_out = self.postnet(S_conv_in)
        S_residual = S_conv_out.squeeze(1).transpose(0, 1)

        S_pred = S_pred + S_residual

        return S_pred


def single_fit():
    net = SingleDecoder()
    loader = get_lj_loader(batch_size=1, limit=1)
    optimizer = torch.optim.Adam(net.parameters())
    n_epochs = 100000
    save_every = 20
    net.to(device)
    for epoch in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            S_pad, S_lengths, token_pad, token_lengths = batch
            S_pad = S_pad.to(device)
            S_pred = net(S_pad)
            loss = ((S_pred-S_pad)**2).mean()
            loss.backward()
            optimizer.step()
            print("epoch {:d}, loss: {:g}".format(epoch, loss.item()))
        if (epoch+1) % save_every == 0:
            torch.save(net.state_dict(), 'data/ckpts/single_{:d}.sd'.format(epoch))
            torch.save(net.state_dict(), 'data/ckpts/single_latest.sd')
