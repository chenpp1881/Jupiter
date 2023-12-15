import logging
import random
import numpy as np
import torch
import argparse
from data_utils import load_data
from train import ClipTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', type=str, default='RE',
                        choices=['RE', 'TD', 'IO'])

    # clip stage args
    parser.add_argument('--epoch_clip', type=int, default=50)
    parser.add_argument('--batch_size_clip', type=int, default=8)
    parser.add_argument('--lr_clip', type=float, default=1e-5)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--mlmloss', type=float, default=0.1)
    parser.add_argument('--model_path', type=str, default=r'/data3/cyz/semi-sup/codet5/')
    parser.add_argument('--plot_path', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=1.1)

    # classifier stage args
    parser.add_argument('--epoch_cla', type=int, default=50)
    parser.add_argument('--batch_size_cla', type=int, default=16)
    parser.add_argument('--lr_2', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--savepath', type=str, default='./Results')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_file', type=str, default=None)
    parser.add_argument('--train_clip', action='store_false')
    parser.add_argument('--train_size', type=float, default=0.5)

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # parse agrs
    args = parse_args()
    logger.info(vars(args))

    # select device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device is %s', args.device)

    # set seed
    set_seed(args.seed)

    # get labeled data and unlabeled data
    data_dict = load_data(args)

    # CL model
    trainer = ClipTrainer(args)

    data_dict['labeled_epoch'] = 0

    while len(data_dict['unlabel_data']) > 0 or args.threshold >= 0.5:

        logger.info('%s epoch begin!' % data_dict['labeled_epoch'])
        logger.info('%s data loading!~' % len(data_dict['train_data']))
        logger.info(f'threshold: {args.threshold}')

        pseudo_data, pseudo_label, data_dict = trainer.train(data_dict)
        args.threshold -= 0.1
        trainer.args = args
        data_dict['labeled_epoch'] += 1

        trainer.train_classicication(data_dict)