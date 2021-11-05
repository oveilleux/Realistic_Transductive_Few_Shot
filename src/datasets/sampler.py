import torch
import numpy as np
import math

class CategoriesSampler():
    """
            CategorySampler
            inputs:
                label : All labels of dataset
                n_batch : Number of batches to load
                n_cls : Number of classification ways (n_ways)
                s_shot : Support shot
                q_shot : Query shot (balanced)
                balanced : 'balanced': Balanced query class distribution: Standard class balanced Few-Shot setting
                           'dirichlet': Dirichlet's distribution over query data: Realisatic class imbalanced Few-Shot setting
                alpha : Dirichlet's concentration parameter

            returns :
                sampler : CategoriesSampler object that will yield batch when iterated
                When iterated returns : batch
                        data : torch.tensor [n_support + n_query, channel, H, W]
                               [support_data, query_data]
                        labels : torch.tensor [n_support + n_query]
                               [support_labels, query_labels]

                        Where :
                            Support data and labels class sequence is :
                                [a b c d e a b c d e a b c d e ...]
                             Query data and labels class sequence is :
                               [a a a a a a a a b b b b b b b c c c c c d d d d d e e e e e ...]
    """
    def __init__(self, label, n_batch, n_cls, s_shot, q_shot, balanced, alpha = 2):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.s_shot = s_shot
        self.q_shot = q_shot
        self.balanced = balanced
        self.alpha = alpha

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            support = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.s_shot]  # sample n_per data index of this class
                support.append(l[pos])
            support = torch.stack(support).t().reshape(-1)
            # support = torch.stack(support).reshape(-1)

            query = []
            alpha = self.alpha * np.ones(self.n_cls)
            if self.balanced == 'balanced':
                query_samples = np.repeat(self.q_shot, self.n_cls)
            else:
                query_samples = get_dirichlet_query_dist(alpha, 1, self.n_cls, self.n_cls * self.q_shot)[0]

            for c, nb_shot in zip(classes, query_samples):
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:nb_shot]  # sample n_per data index of this class
                query.append(l[pos])
            query = torch.cat(query)

            batch = torch.cat([support, query])

            yield batch


def convert_prob_to_samples(prob, q_shot):
    prob = prob * q_shot
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)


def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)