from torch.autograd import Variable
from src.utils import get_metric
from tqdm import tqdm
import torch
from src.utils import Logger, extract_features

class ProtoNet(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.n_ways = args.n_ways
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

    def forward(self, z_support, z_query, shot):
        """
            inputs:
                z_s : torch.Tensor of shape [n_task, s_shot, feature_dim]
                z_q : torch.Tensor of shape [n_task, q_shot, feature_dim]
                shot: Shot

            outputs:
                preds_q: torch.Tensor of shape [q_shot]
        """
        z_support = Variable(z_support)
        z_query = Variable(z_query)

        z_support = z_support.contiguous()
        # z_proto = z_support.view(self.n_ways, shot, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_proto = z_support.view(shot, self.n_ways, -1).mean(0)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous()

        dists = get_metric('euclidean')(z_proto, z_query)
        scores = -dists
        preds_q = scores.argmax(axis=1)
        return preds_q

    def run_prediction(self, support, query, y_q, shot):

        acc_list = []
        self.logger.info(" ==> Executing predictions on {} shot tasks ...".format(shot))
        for i in tqdm(range(self.number_tasks)):
            preds_q = self.forward(z_support=support[i], z_query=query[i], shot=shot)
            acc_list.append((preds_q == y_q[i]).float().mean())
        self.record_info(acc_list=acc_list)



    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']

        # Extract features
        z_s, z_q = extract_features(model=self.model, support=x_s, query=x_q)

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Run predictions
        self.run_prediction(support=support, query=query, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def record_info(self, acc_list):
        self.test_acc.append(torch.stack(acc_list, dim=0)[:,None])

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'acc': self.test_acc}