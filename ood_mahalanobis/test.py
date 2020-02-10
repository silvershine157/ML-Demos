import torch 
import numpy as np

from model import ResNet34
from utils import * 


def ood_test_baseline(model, id_train_loader, id_test_loader, ood_test_loader, args):
    """
    Implementation of baseline ood detection method
    """
    threshold = 0.67
    
    model = model.cuda()
    model.eval()

    TPR = 0.
    TNR = 0.
    with torch.no_grad():
        for x, y in id_test_loader:
            x, y = x.cuda(), y.cuda()
            
            pred, feature_list = model(x)
            confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            TPR += (confidence_score > threshold).sum().item() / id_test_loader.batch_size
        
        for x, y in ood_test_loader:
            x, y = x.cuda(), y.cuda()
            
            pred, feature_list = model(x)
            confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            TNR += (confidence_score < threshold).sum().item() / ood_test_loader.batch_size
        
    print('TPR: {:.4}% |TNP: {:.4}% |threshold: {}'.format(TPR / len(id_test_loader) * 100, TNR / len(ood_test_loader) * 100, threshold))
            

def ood_test_mahalanobis(model, id_train_loader, id_test_loader, ood_test_loader, args):
    """
    TODO
    - step 1. calculate empircal mean and covariance of each of class conditional Gaussian distibtuion(CIFAR10 has 10 classes) 
        - If you don't use feature ensemble, performance will be degenerated, but whether to use it is up to you.
        - If you don't use input pre-processing, performance will be degenerated, but whether to use it is up to you.
    - stpe 2. calculate test samples' confidence score by using Mahalanobis distance and just calculated parameters of class conditional Gaussian distributions
    - step 3. compare the confidence score and the threshold. if confidence score > threshold, it will be assigned to in-distribtuion sample.
    """
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for x, y in id_train_loader:
            x, y = x.cuda(), y.cuda()
            pred, feature_list = model(x)
            '''
            print("Feature dimensions:")
            for feature2d in feature_list:
                feature = torch.mean(torch.mean(feature2d, dim=3), dim=2)
                print(feature.shape)
            break
            '''
            feature2d = feature_list[-1]
            feature = torch.mean(torch.mean(feature2d, dim=3), dim=2)
            obtain_statistics(feature, y)
            break
    pass 

def obtain_statistics(feature, y):
    '''
    y: [B] : in range(C)
    feature: [B, D]
    ---
    mu: [C, D] # center for each class
    cov: [D, D] # tied covariance
    '''
    C = 10 # number of classes
    B, D = feature.shape
    
    # calculate centers
    mu = torch.zeros((C, D)).cuda()
    for c in range(C):
        mu[c, :] = torch.mean(feature[(y == c)], dim=0)
    
    # calculate tied covariance matrix
    cov_sum = torch.zeros((D, D)).cuda()
    for b in range(B):
        dev = feature[b, :] - mu[y[b], :]
        cov_sum += torch.einsum('i,j->ij', dev, dev)
    cov = cov_sum/B

    return mu, cov


def id_classification_test(model, id_train_loader, id_test_loader, args):
    """
    TODO : Calculate test accuracy of CIFAR-10 test set by using Mahalanobis classification method 
    """
    pass


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    def parse_args():
        import argparse
        parser = argparse.ArgumentParser('Mahalanobis-args')
        
        # experimental settings
        parser.add_argument('--seed', type=int, default=0, help='Random seed.')   
        parser.add_argument('--task', type=str, default='ood_detection', help='classification | ood_detection')
        parser.add_argument('--alg', type=str, default='mahalanobis', help='baseline | mahalanobis')
        

        parser.add_argument('--train_bs', type=int, default=1000, help='Batch size of in_trainloader.')   
        parser.add_argument('--test_bs', type=int, default=1000, help='Batch size of in_testloader and out_testloader.')   
        parser.add_argument('--threshold', type=int, default=8, help='Threshold.')   
        parser.add_argument('--num_workers', type=int, default=0)

        args = parser.parse_args()

        return args

    # arg parse
    args = parse_args()

    # set seed
    set_seed(args.seed)

   
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    """
    in-distribution data loader(CIFAR-10) 
    """
    
    # id_trainloader will be used for estimating empirical class mean and covariance
    id_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    id_trainloader = torch.utils.data.DataLoader(id_trainset, batch_size=args.train_bs,
                                            shuffle=False, num_workers=args.num_workers)

    # id_testloader will be used for test the given ood detection algorithm
    id_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    id_testloader = torch.utils.data.DataLoader(id_testset, batch_size=args.test_bs,
                                            shuffle=False, num_workers=args.num_workers)
    
    """
    out-of-distribtuion data looader(SVHN)
    """

    # ood_testloader will be used for test the given ood detection algorithm
    ood_testset = torchvision.datasets.SVHN(root='./data', split='test',
                                        download=True, transform=transform)
    ood_testloader = torch.utils.data.DataLoader(ood_testset, batch_size=args.test_bs,
                                            shuffle=False, num_workers=args.num_workers)
    
    # load model trained on CIFAR-10 
    model = ResNet34()
    model.load_state_dict(torch.load('./data/resnet34-31.pth'))

    # ood dectection test
    if args.task == 'ood_detection':
        if args.alg == 'baseline':
            print('result of baseline alg')
            ood_test_baseline(model, id_trainloader, id_testloader, ood_testloader, args)
        elif args.alg == 'mahalanobis':
            print('result of mahalanobis alg')
            ood_test_mahalanobis(model, id_trainloader, id_testloader, ood_testloader, args)
        else:
            print('--alg should be baseline or mahalanobis')
    
    # classification test
    elif args.task == 'classification':
        id_classification_test(model, id_trainloader, id_testloader, args)
    else:
        print('--task should be ood_detection or classification')
