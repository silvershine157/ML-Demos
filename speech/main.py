import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

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
    wave, fs, _, txt = dataset[0]
    

test2()
