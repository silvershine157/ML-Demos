import torch
import torchaudio
import matplotlib.pyplot as plt

# data -> log-mel spectogram -> augmentation -> CNN -> encoder -> attn decoder

filename = "data/LibriSpeech/dev-clean/174/50561/174-50561-0000.flac"
visual_path = "data/visual/"

waveform, sample_rate = torchaudio.load(filename)
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
plt.savefig(visual_path + "waveform.png")

specgram = torchaudio.transforms.MelSpectrogram()(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0, : ,:].detach().numpy(), cmap='gray')
plt.savefig(visual_path + "specgram.png")


