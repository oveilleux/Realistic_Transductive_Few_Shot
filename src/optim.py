import torch.optim
import argparse
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR

def get_scheduler(epochs, num_batches, optimizer, args):

    SCHEDULER = {'step': StepLR(optimizer, args.lr_stepsize, args.gamma),
                 'multi_step': MultiStepLR(optimizer, milestones=[int(.5 * epochs), int(.75 * epochs)],
                                           gamma=args.gamma),
                 'cosine': CosineAnnealingLR(optimizer, num_batches * epochs, eta_min=1e-9),
                 None: None}
    return SCHEDULER[args.scheduler]


def get_optimizer(args: argparse.Namespace,
                  model: torch.nn.Module) -> torch.optim.Optimizer:
    OPTIMIZER = {'SGD': torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)}
    return OPTIMIZER[args.optimizer_name]