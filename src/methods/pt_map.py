# Adaptation of the publicly available code of the paper entitled "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning":
# https://github.com/yhu01/PT-MAP
from tqdm import tqdm
import torch
import time
import math
from ..models.GaussianModel import GaussianModel
from src.utils import Logger, extract_features

class PT_MAP(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.n_ways = args.n_ways
        self.number_tasks = args.batch_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.lam = args.lam
        self.n_queries = args.n_query
        self.n_sum_query = args.n_query * args.n_ways
        self.n_epochs = args.n_epochs
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []

    def getAccuracy(self, preds_q, y_q):
        preds_q = preds_q.argmax(dim=2)

        acc_test = (preds_q == y_q).float().mean(1, keepdim=True)

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(self.number_tasks)
        return m, pm

    def get_GaussianModel(self):
        method_info = {'device': self.device, 'lam': self.lam, 'n_ways': self.n_ways}
        return GaussianModel(**method_info)

    def power_transform(self, support, query):
        """
            inputs:
                support : torch.Tensor of shape [n_task, s_shot, feature_dim]
                query : torch.Tensor of shape [n_task, q_shot, feature_dim]

            outputs:
                support : torch.Tensor of shape [n_task, s_shot, feature_dim]
                query : torch.Tensor of shape [n_task, q_shot, feature_dim]
        """
        support[:,] = torch.pow(support[:,] + 1e-6, self.beta)
        query[:,] = torch.pow(query[:,] + 1e-6, self.beta)
        return support, query

    def centerData(self, data, shot):
        """
            inputs:
                data : torch.Tensor of shape [n_task, s_shot+q_shot, feature_dim]
                shot: Shot
            outputs:
                data : torch.Tensor of shape [n_task, s_shot+q_shot, feature_dim]
        """
        support = data[:, :shot*self.n_ways, :]
        query = data[:, shot*self.n_ways:, :]

        support = support - support.mean(1, keepdim=True)
        support = support / torch.norm(support, 2, 2)[:,:,None]

        query = query - query.mean(1, keepdim=True)
        query = query / torch.norm(query, 2, 2)[:, :, None]

        data = torch.cat((support, query), dim=1)

        return data

    def scaleEachUnitaryDatas(self, datas):

        norms = datas.norm(dim=2, keepdim=True)
        return datas / norms

    def QRreduction(self, data):

        ndatas = torch.qr(data.permute(0, 2, 1)).R
        ndatas = ndatas.permute(0, 2, 1)
        return ndatas

    def performEpoch(self, model, data, y_s, y_q, shot, epochInfo=None):

        p_xj, preds_q = model.getProbas(data=data, y_s=y_s, n_tasks=self.number_tasks, n_sum_query=self.n_sum_query,
                                        n_queries=self.n_queries, shot=shot)

        m_estimates = model.estimateFromMask(data=data, mask=p_xj)

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

    def run_adaptation(self, model, data, y_s, y_q, shot):

        _, preds_q = model.getProbas(data=data, y_s=y_s, n_tasks=self.number_tasks, n_sum_query=self.n_sum_query,
                                          n_queries=self.n_queries, shot=shot)

        # print("Initialisation model accuracy", self.getAccuracy(preds_q=preds_q, y_q=y_q))
        self.logger.info(' ==> Executing PT-MAP adaptation on {} shot tasks ...'.format(shot))
        t0 = time.time()
        for epoch in tqdm(range(self.n_epochs)):
            self.performEpoch(model=model, data=data, y_s=y_s, y_q=y_q, shot=shot, epochInfo=(epoch, self.n_epochs))

            _, preds_q = model.getProbas(data=data, y_s=y_s, n_tasks=self.number_tasks, n_sum_query=self.n_sum_query,
                                             n_queries=self.n_queries, shot=shot)

            self.record_info(y_q=y_q, pred_q=preds_q, new_time=time.time()-t0)
            t0 = time.time()

    def record_info(self, y_q, pred_q, new_time):
        """
        inputs:
            y_q : torch.Tensor of shape [n_tasks, q_shot]
            q_pred : torch.Tensor of shape [n_tasks, q_shot]:
        """
        pred_q = pred_q.argmax(dim=2)
        self.test_acc.append((pred_q == y_q).float().mean(1, keepdim=True))
        self.timestamps.append(new_time)

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps,
                'acc': self.test_acc}

    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']

        # Extract features
        z_s, z_q = extract_features(model=self.model, support=x_s, query=x_q)

        self.logger.info(' ==> Executing initial data transformation ...')
        # Power transform
        support, query = self.power_transform(support=z_s, query=z_q)

        data = torch.cat((support, query), dim=1)

        data = self.QRreduction(data)
        data = self.scaleEachUnitaryDatas(data)
        data = self.centerData(data, shot)

        # Transfer tensors to GPU if needed
        data = data.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        gaus_model = self.get_GaussianModel()
        gaus_model.initFromLabelledDatas(data=data, n_tasks=self.number_tasks,
                                         shot=shot, n_ways=self.n_ways, n_queries=self.n_queries, n_nfeat=data.size(2))

        self.run_adaptation(model=gaus_model, data=data, y_s=y_s, y_q=y_q, shot=shot)

        logs = self.get_logs()

        return logs