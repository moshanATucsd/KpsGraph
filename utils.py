import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal

def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(args.edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices

def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices

def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()

def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def error_values(data_all,relations):
        data=data_all[:,:,0:2]

        variance = torch.rand(data.shape[0],data.shape[1], data.shape[2])*2 -1
        m = Normal(torch.Tensor([0.0,0.0]), torch.Tensor([0.1,0.1]))

        for a,b in enumerate(variance[0]):
            values,count=np.unique(relations[:,a*11:(a+1)*11], return_counts=True)
            #max_ind = count.argsort()[-1:][::-1]
            if 0 in values:
                variance[0][a] = m.sample()*0.5
            else:
                variance[0][a] = m.sample()*0 
        return variance

def error_values_old(data_all):
        data=data_all[:,:,0:2]
        variance = torch.rand(data.shape[0],data.shape[1], data.shape[2])*2 -1
        for a,b in enumerate(variance[0]):
            if int(data_all[0][a,2]) == 1:
                variance[0][a] = variance[0][a]*0.01 
            if int(data_all[0][a,2]) == 0:
                variance[0][a] = 0
            if int(data_all[0][a,2]) == 2:
                variance[0][a] = variance[0][a]*0.1
        return variance
    
def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()

def load_data_vis(batch_size=1, suffix=''):
    loc_train = np.load('data/loc_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')
    vis_train = np.load('data/vis_train' + suffix + '.npy')
    path_train = np.load('data/path_train' + suffix + '.npy')
    type_train = np.load('data/type_train' + suffix + '.npy')
   
    loc_valid = np.load('data/loc_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')
    vis_valid = np.load('data/vis_valid' + suffix + '.npy')
    path_valid = np.load('data/path_valid' + suffix + '.npy')
    type_valid = np.load('data/type_valid' + suffix + '.npy')
    
    loc_test = np.load('data/loc_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')
    vis_test = np.load('data/vis_test' + suffix + '.npy')
    path_test = np.load('data/path_test' + suffix + '.npy')
    type_test = np.load('data/type_test' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_kps = loc_train.shape[2]

    loc_max = loc_train.max()
    loc_min = loc_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 2, 1])
    vis_train = np.transpose(vis_train, [0, 2, 1])
    #print(loc_train.shape)
    #type_train = np.transpose(type_train, [0, 2, 1])
    #print(type_train.shape)
    feat_train = np.concatenate((loc_train,type_train), axis=2)
    edges_train = np.reshape(edges_train, [-1, num_kps ** 2])
    edges_train = np.array((edges_train + 1) , dtype=np.int64)
    
    loc_valid = np.transpose(loc_valid, [0, 2, 1])
    vis_valid = np.transpose(vis_valid, [0, 2, 1])
    #type_valid = np.transpose(type_valid, [0, 2, 1])
    feat_valid = np.concatenate([loc_valid, type_valid], axis=2)
    edges_valid = np.reshape(edges_valid, [-1, num_kps ** 2])
    edges_valid = np.array((edges_valid + 1) , dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 2, 1])
    vis_test = np.transpose(vis_test, [0, 2, 1])
    #type_test = np.transpose(type_test, [0, 2, 1])
    feat_test = np.concatenate([loc_test, type_test], axis=2)
    edges_test = np.reshape(edges_test, [-1, num_kps ** 2])
    edges_test = np.array((edges_test + 1), dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    #vis_train = torch.FloatTensor(vis_train)
    edges_train = torch.LongTensor(edges_train)
    
    feat_valid = torch.FloatTensor(feat_valid)
    #vis_valid = torch.FloatTensor(vis_valid)
    edges_valid = torch.LongTensor(edges_valid)
    
    feat_test = torch.FloatTensor(feat_test)
    #vis_test = torch.FloatTensor(vis_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_kps, num_kps)) - np.eye(num_kps)),
        [num_kps, num_kps])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, path_train, valid_data_loader, path_valid, test_data_loader, path_test, loc_max, loc_min

def load_data(batch_size=1, suffix=''):
    loc_train = np.load('data/loc_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')
    vis_train = np.load('data/vis_train' + suffix + '.npy')
    path_train = np.load('data/path_train' + suffix + '.npy')
    type_train = np.load('data/type_train' + suffix + '.npy')
   
    loc_valid = np.load('data/loc_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')
    vis_valid = np.load('data/vis_valid' + suffix + '.npy')
    path_valid = np.load('data/path_valid' + suffix + '.npy')
    type_valid = np.load('data/type_valid' + suffix + '.npy')
    
    loc_test = np.load('data/loc_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')
    vis_test = np.load('data/vis_test' + suffix + '.npy')
    path_test = np.load('data/path_test' + suffix + '.npy')
    type_test = np.load('data/type_test' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_kps = loc_train.shape[2]

    loc_max = loc_train.max()
    loc_min = loc_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 2, 1])
    vis_train = np.transpose(vis_train, [0, 2, 1])
    #print(loc_train.shape)
    #type_train = np.transpose(type_train, [0, 2, 1])
    #print(type_train.shape)
    feat_train = np.concatenate((loc_train,type_train), axis=2)
    edges_train = np.reshape(edges_train, [-1, num_kps ** 2])
    edges_train = np.array((edges_train + 1) , dtype=np.int64)
    
    loc_valid = np.transpose(loc_valid, [0, 2, 1])
    vis_valid = np.transpose(vis_valid, [0, 2, 1])
    #type_valid = np.transpose(type_valid, [0, 2, 1])
    feat_valid = np.concatenate([loc_valid, type_valid], axis=2)
    edges_valid = np.reshape(edges_valid, [-1, num_kps ** 2])
    edges_valid = np.array((edges_valid + 1) , dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 2, 1])
    vis_test = np.transpose(vis_test, [0, 2, 1])
    #type_test = np.transpose(type_test, [0, 2, 1])
    feat_test = np.concatenate([loc_test, type_test], axis=2)
    edges_test = np.reshape(edges_test, [-1, num_kps ** 2])
    edges_test = np.array((edges_test + 1), dtype=np.int64)


    feat_train = torch.FloatTensor(feat_train)
    #vis_train = torch.FloatTensor(vis_train)
    edges_train = torch.LongTensor(edges_train)
    
    feat_valid = torch.FloatTensor(feat_valid)
    #vis_valid = torch.FloatTensor(vis_valid)
    edges_valid = torch.LongTensor(edges_valid)
    
    feat_test = torch.FloatTensor(feat_test)
    #vis_test = torch.FloatTensor(vis_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_kps, num_kps)) - np.eye(num_kps)),
        [num_kps, num_kps])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min