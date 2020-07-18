import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import textproc

def compress_amplitude(S_uncomp):
    S_comp = torch.log(torch.clamp(S_uncomp, min=1e-5))
    return S_comp

def decompress_amplitude(S_comp):
    S_uncomp = torch.exp(S_comp)
    return S_uncomp

class LinSpecDataset(Dataset):
    def __init__(self, base_ds, limit):
        self.base_ds = base_ds
        self.limit = limit
        sr = 22050
        n_fft = int(0.05*sr)
        win_length = n_fft
        hop_length = n_fft//4
        self.to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    def wav_to_spec(self, wav):
        S = self.to_spectrogram(wav)
        S = compress_amplitude(S)
        S = S.transpose(0, 2).squeeze(2)
        return S

    def spec_to_wav(self, S):
        S = S.unsqueeze(2).transpose(0, 2)
        S = decompress_amplitude(S)
        wav = self.griffin_lim(S)
        return wav

    def __len__(self):
        if self.limit is None:
            return len(self.base_ds)
        else:
            return self.limit

    def __getitem__(self, i):
        wav, _, _, txt = self.base_ds[i]
        S = self.wav_to_spec(wav)
        tokens = torch.LongTensor(textproc.text_to_sequence(txt, ["english_cleaners"]))
        return (S, tokens)

def collate_lj(L):
    '''
    L: batch size list of
        S: [Ls, d_spec]
        tokens: [Lt]
    ---
    S_pad: [Ls_max, B, d_spec]
    token_pad: [Lt_max, B]
    '''
    max_S_len = 0
    max_token_len = 0
    S_list = []
    S_len_list = []
    token_list = []
    token_len_list = []
    for sample in L:
        S, tokens = sample
        S_list.append(S)
        S_len_list.append(S.shape[0])
        token_list.append(tokens)
        token_len_list.append(tokens.shape[0])
    S_pad = pad_sequence(S_list)
    token_pad = pad_sequence(token_list)
    S_lengths = torch.LongTensor(S_len_list)
    token_lengths = torch.LongTensor(token_len_list)
    return (S_pad, S_lengths, token_pad, token_lengths)


def get_lj_loader(batch_size=4, limit=None, get_dataset=False, num_workers=0, pin_memory=False):
    dataset = LinSpecDataset(torchaudio.datasets.LJSPEECH('./data'), limit)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_lj, num_workers=num_workers, pin_memory=pin_memory)
    if get_dataset:
        return dataset, loader
    return loader
