# Adaptation of the publicly available code of the ICML 2020 paper entitled "LaplacianShot: Laplacian Regularized Few Shot Learning":
# https://github.com/imtiazziko/LaplacianShot
import torch.nn.functional as F
from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
import math
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
from sklearn.neighbors import NearestNeighbors
from ..utils import get_metric, Logger, extract_features

class LaplacianShot(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.knn = args.knn
        self.arch = args.arch
        self.balanced = args.balanced
        self.dataset = args.dataset
        self.proto_rect = args.proto_rect
        self.norm_type = args.norm_type
        self.iter = args.iter
        self.n_ways = args.n_ways
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.shots = args.shots
        if args.use_tuned_lmd:
            self.lmd = self.get_tuned_lmd()
        else:
            self.lmd = args.lmd
        self.init_info_lists()

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.ent_energy = []
        self.test_acc = []

    def record_info(self, acc_list, ent_energy, new_time):
        """
        inputs:
            acc_list : torch.Tensor of shape [iter]
            ent_energy : torch.Tensor of shape [iter]
            new_time: torch.Tensor of shape [iter]
        """
        self.test_acc.append(acc_list)
        self.ent_energy.append(ent_energy)
        self.timestamps.append(new_time)

    def get_logs(self):
        self.test_acc = torch.stack(self.test_acc, dim=0).squeeze(2).cpu().numpy()
        self.ent_energy = np.array(self.ent_energy)
        self.timestamps = np.array(self.timestamps).sum(0)
        return {'timestamps': self.timestamps,
                'acc': self.test_acc,
                'ent_energy': self.ent_energy}

    def normalization(self, z_s, z_q, train_mean):
        """
            inputs:
                z_s : np.Array of shape [n_task, s_shot, feature_dim]
                z_q : np.Array of shape [n_task, q_shot, feature_dim]
                train_mean: np.Array of shape [feature_dim]
        """
        z_s = z_s.cpu()
        z_q = z_q.cpu()
        # CL2N Normalization
        if self.norm_type == 'CL2N':
            z_s = z_s - train_mean
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q - train_mean
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]
        # L2 Normalization
        elif self.norm_type == 'L2N':
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]
        return z_s, z_q

    def proto_rectification(self, support, query, shot):
        """
            inputs:
                support : np.Array of shape [n_task, s_shot, feature_dim]
                query : np.Array of shape [n_task, q_shot, feature_dim]
                shot: Shot

            ouput:
                proto_weights: prototype of each class
        """
        eta = support.mean(1) - query.mean(1)  # Shifting term
        query = query + eta[:, np.newaxis, :]  # Adding shifting term to each normalized query feature
        query_aug = np.concatenate((support, query), axis=1)  # Augmented set S' (X')
        support_ = support.reshape(support.shape[0], shot, self.n_ways, support.shape[-1]).mean(1)  # Init basic prototypes Pn
        support_ = torch.from_numpy(support_)
        query_aug = torch.from_numpy(query_aug)
        proto_weights = []
        for j in tqdm(range(self.number_tasks)):
            distance = get_metric('cosine')(support_[j], query_aug[j])
            predict = torch.argmin(distance, dim=1)
            cos_sim = F.cosine_similarity(query_aug[j][:, None, :], support_[j][None, :, :], dim=2)  # Cosine similarity between X' and Pn
            cos_sim = 10 * cos_sim
            W = F.softmax(cos_sim, dim=1)
            support_list = [(W[predict == i, i].unsqueeze(1) * query_aug[j][predict == i]).mean(0, keepdim=True) for i
                                in predict.unique()]
            proto = torch.cat(support_list, dim=0)  # Rectified prototypes P'n
            proto_weights.append(proto)

        proto_weights = np.stack(proto_weights, axis=0)
        return proto_weights

    def create_affinity(self, X):
        N, D = X.shape

        nbrs = NearestNeighbors(n_neighbors=self.knn).fit(X)
        dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), self.knn - 1)
        col = knnind[:, 1:].flatten()
        data = np.ones(X.shape[0] * (self.knn - 1))
        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
        return W

    def normalize(self, Y_in):
        maxcol = np.max(Y_in, axis=1)
        Y_in = Y_in - maxcol[:, np.newaxis]
        N = Y_in.shape[0]
        size_limit = 150000
        if N > size_limit:
            batch_size = 1280
            Y_out = []
            num_batch = int(math.ceil(1.0 * N / batch_size))
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, N)
                tmp = np.exp(Y_in[start:end, :])
                tmp = tmp / (np.sum(tmp, axis=1)[:, None])
                Y_out.append(tmp)
            del Y_in
            Y_out = np.vstack(Y_out)
        else:
            Y_out = np.exp(Y_in)
            Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

        return Y_out

    def entropy_energy(self, Y, unary, kernel, bound_lambda, batch=False):
        tot_size = Y.shape[0]
        pairwise = kernel.dot(Y)
        if batch == False:
            temp = (unary * Y) + (-bound_lambda * pairwise * Y)
            E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
        else:
            batch_size = 1024
            num_batch = int(math.ceil(1.0 * tot_size / batch_size))
            E = 0
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, tot_size)
                temp = (unary[start:end] * Y[start:end]) + (-bound_lambda * pairwise[start:end] * Y[start:end])
                E = E + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()

        return E

    def bound_update(self, unary, kernel, bound_lambda, y_s, y_q, task_i, bound_iteration=20, batch=False):
        oldE = float('inf')
        Y = self.normalize(-unary)
        E_list = []
        out_list = []
        acc_list = []
        timestamps = []
        t0 = time.time()
        for i in range(bound_iteration):
            additive = -unary
            mul_kernel = kernel.dot(Y)
            Y = -bound_lambda * mul_kernel
            additive = additive - Y
            Y = self.normalize(additive)
            E = self.entropy_energy(Y, unary, kernel, bound_lambda, batch)
            E_list.append(E)
            # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
            l = np.argmax(Y, axis=1)
            out = np.take(y_s, l)
            timestamps.append(time.time()-t0)

            if (i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE))):
                # print('Converged')
                out_list.append(torch.from_numpy(out))
                acc_list.append((torch.from_numpy(y_q[task_i]) == torch.from_numpy(out)).float())
                for j in range(bound_iteration-i-1):
                    out_list.append(out_list[i].detach().clone())
                    acc_list.append(acc_list[i].detach().clone())
                    E_list.append(E_list[i])
                    timestamps.append(0)
                break

            else:
                oldE = E.copy()

                out_list.append(torch.from_numpy(out))
                acc_list.append((torch.from_numpy(y_q[task_i]) == torch.from_numpy(out)).float())
            t0 = time.time()

        out_list = torch.stack(out_list, dim=0)
        acc_list = torch.stack(acc_list, dim=0).mean(dim=1, keepdim=True)

        return out_list, acc_list, E_list, timestamps

    def get_tuned_lmd(self):
        """"
        Returns tuned lambda values for [1-shot, 5-shot] tasks
        """
        lmd = {'dirichlet': {'resnet18': {'mini': [0.7, 0.7],
                                         'tiered': [1.0, 0.7],
                                         'cub': [1.0, 0.8]},
                            'wideres': {'mini': [1.0, 0.7],
                                    'tiered': [1.0, 0.8]}},

                'balanced': {'resnet18': {'mini': [0.7, 0.3],
                                          'tiered': [0.7, 0.1],
                                          'cub': [0.7, 0.3]},
                             'wideres': {'mini': [0.7, 0.3],
                                     'tiered': [0.7, 0.1]}
               }}
        return lmd[self.balanced][self.arch][self.dataset]

    def get_lmd(self, shot):
        idx = self.shots.index(shot)
        if idx > len(self.lmd) - 1:
            return self.lmd[-1]
        else:
            return self.lmd[idx]


    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']
        train_mean = task_dic['train_mean']

        # Extract features
        z_s, z_q = extract_features(model=self.model, support=x_s, query=x_q)

        # Perform normalizations required
        support, query = self.normalization(z_s=z_s, z_q=z_q, train_mean=train_mean)

        support = support.numpy()
        query = query.numpy()
        # y_s = y_s.numpy().squeeze(2)[:,::shot][0]
        y_s = y_s.numpy().squeeze(2)[:, :self.n_ways][0]
        y_q = y_q.numpy().squeeze(2)

        if self.proto_rect:
            self.logger.info(" ==> Executing proto-rectification ...")
            support = self.proto_rectification(support=support, query=query, shot=shot)
        else:
            support = support.reshape(self.number_tasks, shot, self.n_ways, support.shape[-1]).mean(1)

        # Run adaptation
        self.run_prediction(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_prediction(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the LaplacianShot inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        records :
            accuracy
            ent_energy
            inference time
        """
        if self.lmd is None:
            lmd = 1
        else:
            lmd = self.get_lmd(shot)
        self.logger.info(" ==> Executing {}-shot predictions with lmd = {} ...".format(shot, lmd))
        for i in tqdm(range(self.number_tasks)):

            substract = support[i][:, None, :] - query[i]
            distance = LA.norm(substract, 2, axis=-1)
            unary = distance.transpose() ** 2
            W = self.create_affinity(query[i])
            preds, acc_list, ent_energy, times = self.bound_update(unary=unary, kernel=W, bound_lambda=lmd, y_s=y_s, y_q=y_q, task_i=i,
                                                bound_iteration=self.iter)

            self.record_info(acc_list=acc_list, ent_energy=ent_energy, new_time=times)