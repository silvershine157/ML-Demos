import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# Goal: overfit tacotron 2 to SpeechCommands dataset

def test1():
    dataset = torchaudio.datasets.LJSPEECH('./data', download=True)
    #loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    a,b,c,d = dataset[0]
    print(a.shape)
    print(b)
    print(c)
    print(d)

def test2():
    dataset = torchaudio.datasets.LJSPEECH('./data')
    wave, sr, _, txt = dataset[0]
    wave = wave.numpy().reshape(-1)

    plt.figure()

    # original envelope
    plt.subplot(311)
    librosa.display.waveplot(wave, sr=sr)

    # spectrogram
    plt.subplot(312)
    n_fft = int(0.05*sr)
    hop_length = int(0.0125*sr)
    n_mels=80
    S = librosa.feature.melspectrogram(wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

    # Griffin-Lim reconstruction
    plt.subplot(313)
    print(S.shape)
    print(n_fft, hop_length)
    wave_recon = librosa.core.griffinlim(S, hop_length=hop_length, win_length=n_fft)
    librosa.display.waveplot(wave_recon, sr=sr)

    plt.show()


def test3():
    dataset = torchaudio.datasets.LJSPEECH('./data')
    wave, sr, _, txt = dataset[0]
    wave = wave.numpy().reshape(-1)

    S = np.abs(librosa.stft(wave))
    librosa.display.specshow(S)
    plt.show()
    print(S.shape)


def test4():

    dataset = torchaudio.datasets.LJSPEECH('./data')
    wave, sr, _, txt = dataset[0]
    
    n_fft = int(0.05*sr)
    hop_length = n_fft//4
    n_mels = 80
    f_min = 125
    f_max = 7600

    # wav -> mel spectrogram
    wav_to_power = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    power_to_db = torchaudio.transforms.AmplitudeToDB()
    mel = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sr, f_min=f_min, f_max=f_max)
    # mel spectrogram -> wav
    inv_mel = torchaudio.transforms.InverseMelScale(n_stft=n_fft, n_mels=n_mels, sample_rate=sr, f_min=f_min, f_max=f_max)
    power_to_wav = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)


    print('wave:', wave.shape)
    S_power = wav_to_power(wave)
    print('S_power:', S_power.shape)
    S_db = power_to_db(S_power)
    print('S_db:', S_db.shape)
    S_mel = mel(S_db)
    print('S_mel:', S_mel.shape)
    

    '''
    plt.figure()
    plt.subplot(311)
    plt.plot(wave.t().numpy())
    plt.subplot(312)
    plt.imshow(S_db[0])
    plt.subplot(313)
    plt.plot(wave_recon.t().numpy())
    plt.show()
    '''

test4()