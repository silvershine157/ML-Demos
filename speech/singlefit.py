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
        S_pred = torch.stack(out_frame_list)
        return S_pred

def single_fit():
    net = SingleDecoder()
    loader = get_lj_loader(batch_size=1, limit=1)
    optimizer = torch.optim.Adam(net.parameters())
    n_epochs = 1000
    save_every = 10
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
            print(loss.item())
        if (epoch+1) % save_every == 0:
            torch.save(net.state_dict(), 'data/ckpts/latest.sd')

single_fit()