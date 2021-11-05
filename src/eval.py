import numpy as np
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint, Logger, extract_mean_features
from src.methods.tim import ALPHA_TIM, TIM_GD
from src.methods.laplacianshot import LaplacianShot
from src.methods.bdcspn import BDCSPN
from src.methods.simpleshot import SimpleShot
from src.methods.baseline import Baseline, Baseline_PlusPlus
from src.methods.pt_map import PT_MAP
from src.methods.protonet import ProtoNet
from src.methods.entropy_min import Entropy_min
from src.datasets import Tasks_Generator, CategoriesSampler, get_dataset, get_dataloader


class Evaluator:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)

    def run_full_evaluation(self, model):
        """
        Run the evaluation over all the tasks
        inputs:
            model : The loaded model containing the feature extractor
            args : All parameters

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.method))
        load_checkpoint(model=model, model_path=self.args.ckpt_path, type=self.args.model_tag)
        dataset = {}
        loader_info = {'aug': False, 'out_name': False}

        if self.args.target_data_path is not None:  # This mean we are in the cross-domain scenario
            loader_info.update({'path': self.args.target_data_path,
                                'split_dir': self.args.target_split_dir})

        train_set = get_dataset('train', args=self.args, **loader_info)
        dataset['train_loader'] = train_set

        test_set = get_dataset(self.args.used_set, args=self.args, **loader_info)
        dataset.update({'test': test_set})

        train_loader = get_dataloader(sets=train_set, args=self.args)
        train_mean, _ = extract_mean_features(model=model,  train_loader=train_loader, args=self.args,
                                              logger=self.logger, device=self.device)

        results = []
        for shot in self.args.shots:
            sampler = CategoriesSampler(dataset['test'].labels, self.args.batch_size,
                                        self.args.n_ways, shot, self.args.n_query,
                                        self.args.balanced, self.args.alpha_dirichlet)

            test_loader = get_dataloader(sets=dataset['test'], args=self.args,
                                         sampler=sampler, pin_memory=True)
            task_generator = Tasks_Generator(n_ways=self.args.n_ways, shot=shot, loader=test_loader,
                                             train_mean=train_mean, log_file=self.log_file)
            results_task = []
            for i in range(int(self.args.number_tasks/self.args.batch_size)):

                method = self.get_method_builder(model=model)

                tasks = task_generator.generate_tasks()

                # Run task
                logs = method.run_task(task_dic=tasks, shot=shot)

                acc_mean, acc_conf = compute_confidence_interval(logs['acc'][:, -1])

                results_task.append(acc_mean)
                del method
            results.append(results_task)

        mean_accuracies = np.asarray(results).mean(1)
        self.logger.info('----- Final test results -----')
        for shot in self.args.shots:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,
                                                                                   mean_accuracies[self.args.shots.index(shot)]))
        return mean_accuracies

    def get_method_builder(self, model):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': self.device, 'log_file': self.log_file, 'args': self.args}
        if self.args.method == 'ALPHA-TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.method == 'TIM-GD':
            method_builder = TIM_GD(**method_info)
        elif self.args.method == 'LaplacianShot':
            method_builder = LaplacianShot(**method_info)
        elif self.args.method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif self.args.method == 'SimpleShot':
            method_builder = SimpleShot(**method_info)
        elif self.args.method == 'Baseline':
            method_builder = Baseline(**method_info)
        elif self.args.method == 'Baseline++':
            method_builder = Baseline_PlusPlus(**method_info)
        elif self.args.method == 'PT-MAP':
            method_builder = PT_MAP(**method_info)
        elif self.args.method == 'ProtoNet':
            method_builder = ProtoNet(**method_info)
        elif self.args.method == 'Entropy-min':
            method_builder = Entropy_min(**method_info)
        else:
            self.logger.exception("Method must be in ['TIM_GD', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
            raise ValueError("Method must be in ['TIM_GD', 'ALPHA_TIM', 'LaplacianShot', 'BDCSPN', 'SimpleShot', 'Baseline', 'Baseline++', 'PT-MAP', 'Entropy_min']")
        return method_builder