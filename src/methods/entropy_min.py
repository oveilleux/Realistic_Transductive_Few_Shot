import torch.nn.functional as F
from src.utils import get_mi, get_cond_entropy, get_entropy, get_one_hot, Logger, extract_features, load_checkpoint
from tqdm import tqdm
import torch
import time

class Entropy_min(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.temp = args.temp
        self.loss_weights = args.loss_weights.copy()
        self.iter = args.iter
        self.batch_size = args.batch_size
        self.lr = float(args.lr_entropy_min)
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        load_checkpoint(model=self.model, model_path=args.ckpt_path, type=args.model_tag)

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.entropy = []
        self.cond_entropy = []
        self.test_acc = []
        self.losses = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2)) \
                              - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]
        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def init_weights(self, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, channels, H, W]
            query : torch.Tensor of shape [n_task, q_shot, channels, H, W]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        support, query = extract_features(model=self.model, support=support, query=query)
        # Normalize data
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)

        t0 = time.time()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        self.record_info(new_time=time.time() - t0,
                         support=support,
                         query=query,
                         y_s=y_s,
                         y_q=y_q)
        self.model.train()

    def compute_lambda(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, channels, H, W]
            query : torch.Tensor of shape [n_task, q_shot, channels, H, W]
            y_s : torch.Tensor of shape [n_task, s_shot]
        updates :
            self.loss_weights[0] : Scalar
        """
        support, query = extract_features(model=self.model, support=support, query=query)
        # Normalize data
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)

        self.model.train()

        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0)
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[1]) * self.N_s / self.N_q

    def record_info(self, new_time, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        logits_q = self.get_logits(query).detach()
        preds_q = logits_q.argmax(2)
        q_probs = logits_q.softmax(2)
        self.timestamps.append(new_time)
        self.entropy.append(get_entropy(probs=q_probs.detach()))
        self.cond_entropy.append(get_cond_entropy(probs=q_probs.detach()))
        self.test_acc.append((preds_q == y_q).float().mean(1, keepdim=True))

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps,
                'entropy': self.entropy, 'cond_entropy': self.cond_entropy,
                'acc': self.test_acc, 'losses': self.losses}

    def run_task(self, task_dic, shot):

        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']

        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)
        query = x_q.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Initialize weights
        self.compute_lambda(support=support, query=query, y_s=y_s)
        self.init_weights(support=support, y_s=y_s, query=query, y_q=y_q)

        # Run adaptation
        self.run_adaptation(support=x_s, query=x_q, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs


    def run_adaptation(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the Entropy-min inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, channels, H, W]
            query : torch.Tensor of shape [n_task, q_shot, channels, H, W]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        self.model.train()
        t0 = time.time()
        self.weights.requires_grad_()
        params = list(self.model.parameters()) + [self.weights]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)

        self.logger.info(" ==> Executing entropy-minimization adaptation on {} shot tasks...".format(shot))
        for i in tqdm(range(self.iter)):
            # Extracting features
            outputs_s = []
            outputs_q = []
            for i in range(self.batch_size):
                output_s = self.model(support[i], feature=True)[0].cuda(0)
                output_q = self.model(query[i], feature=True)[0].cuda(0)
                outputs_s.append(output_s)
                outputs_q.append(output_q)
            z_s = torch.stack(outputs_s)
            z_q = torch.stack(outputs_q)

            # Normalize data
            z_s = F.normalize(z_s, dim=2)
            z_q = F.normalize(z_q, dim=2)

            logits_s = self.get_logits(z_s)
            logits_q = self.get_logits(z_q)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            loss = self.loss_weights[0] * ce - (- self.loss_weights[1] * q_cond_ent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()

            self.record_info(new_time=t1-t0,
                             support=z_s,
                             query=z_q,
                             y_s=y_s,
                             y_q=y_q)
            t0 = time.time()