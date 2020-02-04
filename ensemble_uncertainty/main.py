from utils import *
import argparse
from train import *
from model import *


def main(args):
    x = np.linspace(-6,6,100).reshape(100,1) # Test data for regression
    x_set , y_set = generate_data() # Train data for regression
    epochs = args.epochs
    batch_size = args.batch_size
    epsilon = 0.01 # Coeffiecient for 'fast gredient sign method'

    x_train = torch.Tensor(x_set)
    y_train = torch.Tensor(y_set)
    x_test = torch.Tensor(x)

    # Training an ensemble of 5 networks(MLPS) with MSE
    # TODO:Draw Fig1.1
    if args.fig == 1:
        nets = train_ensmeble(BasicMLP, x_train, y_train, ens_size=5)
        mean, var = ensemble_prediction(nets, x_test)
        # Have to calculte predicted mean and std
        # 'mean' have to be a numpy array with shape [100,1]
        # 'std'  have to be a numpy array with shape [100,1]
        draw_graph(x,x_set,y_set, mean, np.sqrt(var))

    # Training a Gaussian MLP(single network) with NLL score rule
    # TODO:Draw Fig1.2
    elif args.fig == 2:
        nets = train_ensmeble(GaussianMLP, x_train, y_train, ens_size=1)
        mean, var = ensemble_prediction(nets, x_test)
        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        draw_graph(x,x_set,y_set,mean, np.sqrt(var))


    # Training a Gaussian MLP with NLL & Adversarial Training
    # TODO:Draw Fig1.3
    elif args.fig == 3:
        nets = train_ensmeble(GaussianMLP, x_train, y_train, ens_size=1, epsilon=epsilon)
        mean, var = ensemble_prediction(nets, x_test)        
        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        draw_graph(x,x_set,y_set,mean, np.sqrt(var))


    #Training a Gaussian mixture MLP (Deep ensemble) with NLL
    # TODO: Draw Fig1.4
    else: #args.fig == 4
        #nets = train_ensmeble(GaussianMLP, x_train, y_train, ens_size=5, epsilon=epsilon)
        nets = train_ensmeble(GaussianMLP, x_train, y_train, ens_size=5)
        mean, var = ensemble_prediction(nets, x_test)    
        # Have to calculte predicted mean and var
        # 'mean' have to be a numpy array with shape [100,1]
        # 'var'  have to be a numpy array with shape [100,1]
        draw_graph(x,x_set,y_set,mean, np.sqrt(var))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument(
        '--epochs',
        type=int,
        default=300)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20)

    parser.add_argument(
        '--fig',
        type=int)
    args = parser.parse_args()

    main(args)