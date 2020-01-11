import torch

sos_idx = 0
eos_idx = 1
pad_idx = 2
max_length = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")