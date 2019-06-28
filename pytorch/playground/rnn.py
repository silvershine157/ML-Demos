import torch
import numpy as np
import matplotlib.pyplot as plt
from lds_simulator import AutoLDS

visual_path = './data/visual/'

# RNN with state = output, no input
class StateOnlyRNN(torch.nn.Module):
    def __init__(self, H_dim, pred_len):
        super(StateOnlyRNN, self).__init__()

        # structural parameters
        self.H_dim = H_dim # dimension of hidden (=output) state
        
        # prediction length
        self.pred_len = pred_len

        # just learn transition
        self.trans = torch.nn.Linear(H_dim, H_dim)

    def forward(self, x):
        # given initial state, estimate pred_len more steps
        B = x.size(0) # batch dimension
        pred = torch.empty(B, self.H_dim, self.pred_len)
        h = x
        for t in range(self.pred_len):
            pred[:, :, t] = h
            h = self.trans(h) # for now, no nonlinearity
        return pred

def test():

    epochs = 500
    H_dim = 10
    n_train = 20
    n_test = 10
    length = 5
    learning_rate = 1e-4

    alds = AutoLDS(H_dim) # match with true dimension
    
    # [N X H_dim X length]
    train_data = alds.fixed_sys_sequences(n_train, length)
    test_data = alds.fixed_sys_sequences(n_test, length, resample_sys=False)
    train_data = torch.Tensor(train_data)
    test_data = torch.Tensor(test_data)

    train_initial = (train_data[:, :, 0]).squeeze()
    test_initial = (test_data[:, :, 0]).squeeze()

    model = StateOnlyRNN(H_dim, length)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        pred = model(train_initial)
        loss = criterion(pred, train_data)

        if(e%10==0):
            print(e, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # compare with prediction
    with torch.no_grad():
        pred = model(test_initial)
        pred = (pred[0, :, :]).squeeze()
        actual = (test_data[0, :, :]).squeeze()
        compare_plot(pred, actual)

def compare_plot(pred, actual):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(pred.numpy().transpose(1, 0))
    plt.subplot(1, 2, 2)
    plt.plot(actual.numpy().transpose(1, 0))
    plt.savefig(visual_path + 'state_rnn_prediction.png')

test()
