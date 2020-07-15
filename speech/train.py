import torch
import torchaudio
import matplotlib.pyplot as plt

# entry script for training

def linspec_train():
    # train with linear spectrogram
    # later converted to audio using Griffin-Lim
    pass

def linspec_sample():
    # produce audio sample for linear spectrogram model
    pass

def compress_amplitude(S_uncomp):
    S_comp = torch.log(torch.clamp(S_uncomp, min=1e-5))
    return S_comp

def decompress_amplitude(S_comp):
    S_uncomp = torch.exp(S_comp)
    return S_uncomp

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

test1()