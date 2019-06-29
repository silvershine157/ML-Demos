import torch
import numpy as np

class LSTM(torch.nn.Module):

    def __init__(self, D_in, D_cell):
        
        super(LSTM, self).__init__()

        # dimensions
        self.D_in = D_in
        self.D_cell = D_cell

        # learnable gate parameters
        self.W_forget = torch.nn.Linear(D_in + D_cell, D_cell)
        self.W_input_sigmoid = torch.nn.Linear(D_in + D_cell, D_cell)
        self.W_input_tanh = torch.nn.Linear(D_in + D_cell, D_cell)
        self.W_output = torch.nn.Linear(D_in + D_cell, D_cell)
        

    def forward(self, x):
        
        # x: (D_in, T) full input
        # c: (D_cell) state
        # h: (D_cell, T+1) full output (extra space for initial value)
        
        _, T = x.shape

        # initialize memory variables
        c = torch.zeros(self.D_cell)
        h = torch.zeros(self.D_cell, T + 1)
        for t in range(T):
            # current information
            x_t_and_h = torch.cat([x[:,t], h[:,t]], dim=0)
            x_t_and_h = x_t_and_h.squeeze()

            ## compute gates
            # forget gate
            forget_gate = torch.sigmoid(self.W_forget(x_t_and_h))
            # candidate states
            cand_c = torch.tanh(self.W_input_tanh(x_t_and_h))
            # select from candidates
            input_gate = torch.sigmoid(self.W_input_sigmoid(x_t_and_h))
            # output gate
            output_gate = torch.sigmoid(self.W_output(x_t_and_h))
            
            ## update values
            # new cell state
            c = forget_gate * c + input_gate * cand_c
            # new output

            h[:, t+1] = output_gate * torch.tanh(c)

        # discard initial output
        h = h[:, 1:]
        return h

def test():
    D_in = 5
    D_cell = 10
    seq_len = 12
    lstm = LSTM(D_in, D_cell)
    x = torch.Tensor(np.random.rand(D_in, seq_len))
    y = lstm(x)
    print(y.shape)

test()
