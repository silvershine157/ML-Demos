import torch
import torchaudio
import matplotlib.pyplot as plt

from train import *
from dataset import *

from tacotron_model import Tacotron2

def test1():
    # audio processing parameters
    sr = 22050
    n_fft = int(0.05*sr)
    win_length = n_fft
    hop_length = n_fft//4
    wave_to_spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    spec_to_wave = torchaudio.transforms.GriffinLim(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    dataset = torchaudio.datasets.LJSPEECH('./data')
    wave, _, _, txt = dataset[0]
    S = compress_amplitude(wave_to_spec(wave))
    wave_recon = spec_to_wave(decompress_amplitude(S))
    torchaudio.save('data/recon_sample.wav', wave_recon, sr)
    plt.imshow(S[0].numpy())
    plt.show()

def test2():
    loader = get_lj_loader(batch_size=4, limit=4)
    net = Tacotron2()
    net.to(device)
    for batch in loader:
        S_pad, S_lengths, token_pad, token_lengths = batch
        S_before, S_after, stop_logits = net(token_pad.to(device), token_lengths, S_pad.to(device), teacher_forcing=True)

def print_and_log(s):
    with open('data/log.txt', 'a') as f:
        f.write(s+'\n')
    print(s)

def test3():
    loader = get_lj_loader(batch_size=4, limit=4, num_workers=4, pin_memory=True)
    net = Tacotron2()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    n_epochs = 1000000
    save_every = 20
    for epoch in range(1, n_epochs+1):
        train_loss = train_epoch(net, loader, optimizer)
        print_and_log("epoch: {:d}, train loss: {:g}".format(epoch, train_loss))
        if epoch%save_every == 0:
            torch.save(net.state_dict(), 'data/ckpts/meancontext_{:d}.sd'.format(epoch))
            torch.save(net.state_dict(), 'data/ckpts/meancontext_latest.sd')

test3()