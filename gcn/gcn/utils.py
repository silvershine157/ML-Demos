import numpy as np
import scipy.sparse as sp
import torch
import scipy

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path, dataset):
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    labels = encode_onehot(labels)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    
    # TODO build adjacency matrix using edge_unordered following the order of idx
    N = len(idx)
    n_edges = edges_unordered.shape[0]
    adj = sp.csr_matrix((N, N))
    for i in range(n_edges):
        idx_from, idx_to = edges_unordered[i, :]
        cpt_idx_from = np.where(idx == idx_from)[0][0]
        cpt_idx_to = np.where(idx == idx_to)[0][0]
        adj[cpt_idx_from, cpt_idx_to] = 1.0

    # TODO build symmetric adjacency matrix
    adj = adj + adj.transpose()

    # TODO normalize features, adj
    adj = renormalize_adj(adj, N)     

    no_edge = False
    if no_edge:
        adj = sp.identity(N)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = normalize(features)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def renormalize_adj(adj, N):
    adj_tilde = adj + sp.identity(N)
    node_deg = adj_tilde.sum(axis=0)
    scaling_vec = 1/np.sqrt(node_deg)
    adj_hat = adj_tilde.multiply(scaling_vec).multiply(scaling_vec.transpose()) # broadcast
    return adj_hat

def normalize(mx): 
    # TODO normalization of given matrix mx
    D = mx.size(1)
    mx = (1/np.sqrt(D))*mx
    return mx


def accuracy(output, labels):
    '''
    output: [N, nclass]
    labels: [N], int
    '''
    # TODO preds
    preds = torch.argmax(output, dim=1)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
