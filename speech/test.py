import torch
import torchaudio
import matplotlib.pyplot as plt

from train import *
from dataset import *
from model import *

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
    net = MiniTTS()
    dataset = LinSpecDataset(torchaudio.datasets.LJSPEECH('./data'))
    loader = get_lj_loader()
    for batch in loader:
        S_pad, S_lengths, token_pad, token_lengths = batch
        print(S_pad.shape)
        enc_out = net.encoder(token_pad, token_lengths)
        S_pred, stop_logits = net.decoder(enc_out, S_pad)

from singlefit import *
single_fit()