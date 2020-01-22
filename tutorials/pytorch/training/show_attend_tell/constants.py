import torch

PAD_token = 0
START_token = 1
END_token = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
