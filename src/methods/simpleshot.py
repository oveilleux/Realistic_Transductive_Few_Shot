from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
from scipy.stats import mode
from src.utils import Logger, extract_features

class SimpleShot(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.norm_type = args.norm_type
        self.n_ways = args.n_ways
        self.num_NN = args.num_NN
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []

    def record_info(self, y_q, pred_q, new_time):
        """
        inputs:
            y_q : torch.Tensor of shape [n_tasks, q_shot]
            q_pred : torch.Tensor of shape [n_tasks, q_shot]:
        """
        acc_list = []
        for i in range(self.number_tasks):
            acc = (pred_q[i] == y_q[i]).mean()
            acc_list.append(torch.tensor(acc))
        self.test_acc.append(torch.stack(acc_list, dim=0)[:,None])
        self.timestamps.append(new_time)

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps,
                'acc': self.test_acc}

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
        y_s = y_s.numpy().squeeze(2)[:,:self.n_ways][0]
        y_q = y_q.numpy().squeeze(2)

        support = support.reshape(self.number_tasks, shot, self.n_ways, support.shape[-1]).mean(1)

        # Run adaptation
        self.run_prediction(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_prediction(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the Simple inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        records :
            accuracy
            inference time
        """
        t0 = time.time()
        time_list = []
        self.logger.info(" ==> Executing predictions on {} shot tasks...".format(shot))
        out_list = []
        for i in tqdm(range(self.number_tasks)):
            substract = support[i][:, None, :] - query[i]
            distance = LA.norm(substract, 2, axis=-1)
            idx = np.argpartition(distance, self.num_NN, axis=0)[:self.num_NN]
            nearest_samples = np.take(y_s, idx)
            out = mode(nearest_samples, axis=0)[0]
            out_list.append(out)
            time_list.append(time.time()-t0)
            t0 = time.time()

        out = np.stack(out_list, axis=0)
        self.record_info(y_q=y_q, pred_q=out, new_time=time_list)