from abc import ABC, abstractmethod 
import numpy as np
import torch

class TripletSelector(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def get_triplets(self, embeddings, labels):
        pass


class HardBatchTripletSelector(TripletSelector):
    def __init__(self, margin=None, cpu=True):
        super(HardBatchTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        
    def get_triplets(self, embeddings, labels):
        dist_mtx = pdist(embeddings, embeddings).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeddings[pos_idxs].contiguous().view(num, -1)
        neg = embeddings[neg_idxs].contiguous().view(num, -1)
        
        return embeddings, pos, neg


def pdist(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx