import torch 
import numpy as np

from model import ResNet34
from utils import * 

import matplotlib.pyplot as plt


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
            
feature_idx = -1

def obtain_train_stats(model, id_train_loader):

    model = model.cuda()
    model.eval()

    # collect features for all layers
    y_all = []
    n_features = 5
    feature_all = [[] for _ in range(n_features)]
    with torch.no_grad():
        for x, y in id_train_loader:
            x, y = x.cuda(), y.cuda()
            pred, feature_list = model(x)
            y_all.append(y)
            for i in range(n_features):
                feature2d = feature_list[i]
                feature = torch.mean(torch.mean(feature2d, dim=3), dim=2)
                feature_all[i].append(feature)

    # calculate class mean and tied covariance for each layer
    y_all = torch.cat(y_all, dim=0)
    stats = []
    for i in range(n_features):
        feature_i = torch.cat(feature_all[i], dim=0)
        mu_i, cov_i = obtain_statistics(feature_i, y_all)
        stats.append((mu_i, cov_i))

    return stats


def obtain_test_scores(model, stats, test_loader):

    model = model.cuda()
    model.eval()

    # calculate scores for each batch
    n_features = 5
    test_scores_list = [[] for i in range(n_features)]
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            pred, feature_list = model(x)
            for i in range(n_features):
                feature2d = feature_list[i]
                feature = torch.mean(torch.mean(feature2d, dim=3), dim=2)
                mu, cov = stats[i]
                confidence_score, _ = mahalanobis_score(feature, mu, cov)
                test_scores_list[i].append(confidence_score)

    # concatenate batches
    test_scores = []
    for i in range(n_features):
        test_scores.append(torch.cat(test_scores_list[i], dim=0))

    return test_scores


def ood_test_mahalanobis(model, id_train_loader, id_test_loader, ood_test_loader, args):

    calculate_stat = False
    if calculate_stat:
        stats = obtain_train_stats(model, id_train_loader)
        torch.save(stats, 'data/stats')
    else:
        stats = torch.load('data/stats')

    calculate_scores = False
    if calculate_scores:
        id_scores = obtain_test_scores(model, stats, id_test_loader)
        ood_scores = obtain_test_scores(model, stats, ood_test_loader)
        torch.save(id_scores, 'data/id_scores')
        torch.save(ood_scores, 'data/ood_scores')
    else:
        id_scores = torch.load('data/id_scores')
        ood_scores = torch.load('data/ood_scores')

    do_logistic_regression = False
    if do_logistic_regression:
        logistic_regression(id_scores, ood_scores)
    process_scores(id_scores, ood_scores)


def process_scores(id_scores, ood_scores):

    n_features = len(id_scores)
    id_scores = torch.stack(id_scores).t()
    ood_scores = torch.stack(ood_scores).t()

    method = 'logistic'
    if method == 'last':
        id_conf = id_scores[:, -1]
        ood_conf = ood_scores[:, -1]
        threshold = -1500
    elif method == 'mean':
        id_conf = id_scores.mean(dim=1)
        ood_conf = ood_scores.mean(dim=1)
        threshold = -400
    elif method == 'logistic':
        weights = torch.Tensor([-1.9344, -1.3672,  3.9768,  0.1706,  0.2704]).cuda().view(1, 5)
        id_conf = (weights * id_scores).mean(dim=1)
        ood_conf = (weights * ood_scores).mean(dim=1)
        threshold = -190

    TP = (id_conf > threshold).sum().item()
    TPR = TP/id_conf.size(0)
    TN = (ood_conf < threshold).sum().item()
    TNR = TN/ood_conf.size(0)
    print('TPR: {:.4}% |TNP: {:.4}% |threshold: {}'.format(TPR*100, TNR*100, threshold))


class LogisticReg(torch.nn.Module):
    def __init__(self):
        super(LogisticReg, self).__init__()
        self.linear = torch.nn.Linear(5, 1)

    def forward(self, x):
        logits = self.linear(x)
        pred = 1.0/(1.0+torch.exp(-logits))
        return pred


def logistic_regression(id_scores, ood_scores):
    n_features = len(id_scores)
    id_scores = torch.stack(id_scores).t()/100
    ood_scores = torch.stack(ood_scores).t()/100
    eps = 1E-6
    model = LogisticReg().cuda()
    model = model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(10000):
        optimizer.zero_grad()
        id_pred = model(id_scores)
        ood_pred = model(ood_scores)
        loss = -torch.log(id_pred+eps).mean()-torch.log(1.0-ood_pred+eps).mean()
        loss.backward()
        optimizer.step()
        print(loss.item())
    print(model.linear.weight)



def mahalanobis_score(feature, mu, cov):
    '''
    feature: [B, D]
    mu: [C, D]
    cov: [D, D]
    ---
    confidence_score: [B]
    '''
    all_dev = feature.unsqueeze(dim=1) - mu.unsqueeze(dim=0) # [B, C, D]
    all_scores = -torch.einsum('bci,ij,bcj->bc', all_dev, cov.inverse(), all_dev)
    confidence_score = torch.max(all_scores, dim=1)[0]
    return confidence_score, all_scores


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
    cov = cov + 0.000001*torch.eye(cov.size(0)).cuda()

    return mu, cov

def id_classification_test(model, id_train_loader, id_test_loader, args):
    calculate_stat = False
    if calculate_stat:
        stats = obtain_train_stats(model, id_train_loader)
        torch.save(stats, 'data/stats')
    else:
        stats = torch.load('data/stats')

    model = model.cuda()
    model.eval()
    n_correct = 0
    n_all = 0
    sm_correct = 0
    mu, cov = stats[-1]
    with torch.no_grad():
        for x, y in id_test_loader:
            x, y = x.cuda(), y.cuda()
            pred, feature_list = model(x)
            feature2d = feature_list[-1]
            feature = torch.mean(torch.mean(feature2d, dim=3), dim=2)
            confidence_score, all_scores = mahalanobis_score(feature, mu, cov)
            y_pred = torch.max(all_scores, dim=1)[1]
            sm_pred = torch.max(pred, dim=1)[1]
            n_correct += torch.sum(y == y_pred).cpu().item()
            sm_correct += torch.sum(y == sm_pred).cpu().item()
            n_all += x.size(0)
        print(n_correct/n_all)
        print(sm_correct/n_all)
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
        

        parser.add_argument('--train_bs', type=int, default=100, help='Batch size of in_trainloader.')   
        parser.add_argument('--test_bs', type=int, default=100, help='Batch size of in_testloader and out_testloader.')   
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
    #model.load_state_dict(torch.load('./data/resnet34-31.pth'))
    model.load_state_dict(torch.load('./data/resnet_cifar10.pth'))

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
