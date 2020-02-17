from model import *
from data import *
import numpy as np
import argparse

#from torchviz import make_dot


SEED = 1234
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_gp(n):
    '''
    x: [n, 1]
    y: [n, 1]
    '''
    x = 2.0*torch.randn(n, 1)
    # construct RBF kernel matrix
    D = x - x.t()
    K = torch.exp(-D**2/1.0)+0.0001*torch.eye(n)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(n), K)
    y = mvn.sample().view(n, 1)
    return x, y

def sample_obs(x_all, y_all, B, N):
    x_all = x_all.view(-1)
    y_all = y_all.view(-1)
    x_obs = torch.zeros(B, N, device=device)
    y_obs = torch.zeros(B, N, device=device)
    for b in range(B):
        p = torch.randperm(x_all.size(0))
        idx = p[:N]
        x_obs[b, :] = x_all[idx]
        y_obs[b, :] = y_all[idx]
    return x_obs, y_obs


def main(args):
    x_train , y_train, x_test, y_test = generate_data()
    x_torch , y_torch = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)
    

    # --------- ARGS --------- 
    epochs = args.epochs
    batch_size = args.batch_size
    ctx_size = args.ctx_size
    lr = args.lr
    print_step = args.print_step

    epsilon = 0.01

    # --------- MODEL --------- 
    x_dim = 1
    y_dim = 1
    r_dim = 128  #128

    net = CNP(x_dim, y_dim, r_dim).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr = lr) #0.01 and 10000 epochs!


    # --------- TRAIN --------- 
    for epoch in range(epochs):
        nll_loss =  0.0
        optimizer.zero_grad()
        
        # TODO -- Train model
        
        # sample GP curve O (fixed n, sample x, calculate kernel matrix, sample y)
        n = 20
        x_all, y_all = sample_gp(n)
        x_all, y_all = x_all.to(device), y_all.to(device)
        N = np.random.randint(low=1, high=n)
        sample_obs(x_all, y_all, N)
        # sample N
        # sample B different subsets of size N from O
        # now we have [B, N, x_dim] and [B, N, y_dim]

        # -------

        if epoch % print_step == 0:
            print('Epoch', epoch, ': nll loss', nll_loss.item())
        #dot = make_dot(mean)
        #dot.render("model.png")

        nll_loss.backward()
        optimizer.step()

    print("final loss : nll loss", nll_loss.item())
    result_mean, result_var = None, None
    
    # TODO -- Calculate result_mean and result_var using your model


    # -------

    draw_graph(x_test, y_test, x_train, y_train, result_mean, np.sqrt(result_var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--batch_size',type=int,default=20)
    parser.add_argument('--ctx_size',type=int,default=20)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--print_step', type=int, default=1000)

    args = parser.parse_args()
    main(args)