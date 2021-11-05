import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from visdom_logger import VisdomLogger
from src.utils import warp_tqdm, save_checkpoint, load_cfg_from_cfg_file, merge_cfg_from_list, Logger, get_log_file
from src.trainer import Trainer
from src.eval import Evaluator
from src.optim import get_optimizer, get_scheduler
from src.models.ingredient import get_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config', type=str, required=True, help='Method config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.base_config is not None
    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    callback = None if args.visdom_port is None else VisdomLogger(port=args.visdom_port)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    torch.cuda.set_device(0)

    # init logger
    log_file = get_log_file(log_path=args.log_path, dataset=args.dataset,
                            backbone=args.arch, method=args.method)
    logger = Logger(__name__, log_file)

    # create model
    logger.info("=> Creating model '{}'".format(args.arch))
    model = torch.nn.DataParallel(get_model(args)).cuda()

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = get_optimizer(args=args, model=model)

    if args.pretrain:
        pretrain = os.path.join(args.pretrain, 'checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            logger.info("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            logger.warning('Did not find pretrained model {}'.format(pretrain))

    if args.resume:
        resume_path = args.ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))

        else:
            logger.warning('Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    cudnn.benchmark = True

    if args.evaluate:
        evaluator = Evaluator(device=device, args=args, log_file=log_file)
        results = evaluator.run_full_evaluation(model=model)
        return results

    # If this line is reached, then training the model
    trainer = Trainer(device=device, args=args)
    scheduler = get_scheduler(optimizer=optimizer,
                              num_batches=len(trainer.train_loader),
                              epochs=args.epochs,
                              args=args)
    tqdm_loop = warp_tqdm(list(range(start_epoch, args.epochs)),
                          disable_tqdm=False)
    for epoch in tqdm_loop:
        # Do one epoch
        trainer.do_epoch(model=model, optimizer=optimizer, epoch=args.epoch,
                         scheduler=scheduler, disable_tqdm=False,
                         callback=callback)

        # Evaluation on validation set
        prec1 = trainer.meta_val(model=model, disable_tqdm=False,
                                 epoch=epoch, callback=callback)
        logger.info('Meta Val {}: {}'.format(epoch, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': args.arch,
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optimizer.state_dict()},
                        is_best=is_best,
                        folder=args.ckpt_path)
        if scheduler is not None:
            scheduler.step()

    # Final evaluation on test set
    logger.info("=> Creating model '{}'".format(args.arch))
    model = torch.nn.DataParallel(get_model(args)).cuda()

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    evaluator = Evaluator(device=device, args=args, log_file=log_file)

    results = evaluator.run_full_evaluation(model=model)
    return results

if __name__ == "__main__":
    main()