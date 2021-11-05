import torch
from src.utils import warp_tqdm, Logger

class Tasks_Generator:
    def __init__(self, n_ways, shot, loader, train_mean, log_file):
        self.n_ways = n_ways
        self.shot = shot
        self.loader = loader
        self.log_file = log_file
        self.logger = Logger(__name__, log_file)
        self.train_mean = train_mean

    def get_task(self, data, labels):
        """
        inputs:
            data : torch.tensor of shape [(s_shot + q_shot) * n_ways, channels, H, W]
            labels :  torch.tensor of shape [(s_shot + q_shot) * n_ways]
        returns :
            task : Dictionnary : x_support : torch.tensor of shape [n_ways * s_shot, channels, H, W]
                                 x_query : torch.tensor of shape [n_ways * query_shot, channels, H, W]
                                 y_support : torch.tensor of shape [n_ways * s_shot]
                                 y_query : torch.tensor of shape [n_ways * query_shot]
        """
        k = self.n_ways * self.shot
        unique_labels = labels[:self.n_ways]
        # unique_labels = torch.unique(labels)
        new_labels = torch.zeros_like(labels)
        for j, y in enumerate(unique_labels):
            new_labels[labels == y] = j
        labels = new_labels
        support, query = data[:k], data[k:]  # shot: 5,3,84,84  query:75,3,84,84
        # support_labels, query_labels = labels[:k].long().cuda(), labels[k:].long().cuda()
        support_labels, query_labels = labels[:k].long(), labels[k:].long()

        task = {'x_s': support, 'y_s': support_labels,
                'x_q': query, 'y_q': query_labels}
        return task

    def generate_tasks(self):
        """

        returns :
            merged_task : { x_support : torch.tensor of shape [batch_size, n_ways * s_shot, channels, H, W]
                            x_query : torch.tensor of shape [batch_size, n_ways * query_shot, channels, H, W]
                            y_support : torch.tensor of shape [batch_size, n_ways * shot]
                            y_query : torch.tensor of shape [batch_size, n_ways * query_shot]
                            train_mean: torch.tensor of shape [feature_dim]}
        """
        tasks_dics = []
        for i, (data, labels, _) in enumerate(warp_tqdm(self.loader, False)):
            task = self.get_task(data, labels)
            tasks_dics.append(task)

        # Now merging all tasks into 1 single dictionnary
        merged_tasks = {}
        n_tasks = len(tasks_dics)
        for key in tasks_dics[0].keys():
            n_samples = tasks_dics[0][key].size(0)
            if key == 'x_s' or key == 'x_q':
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                                        n_samples, 3,
                                                                                                        84, 84)
            else:
                merged_tasks[key] = torch.cat([tasks_dics[i][key] for i in range(n_tasks)], dim=0).view(n_tasks,
                                                                                                        n_samples, -1)
        merged_tasks['train_mean'] = self.train_mean
        return merged_tasks