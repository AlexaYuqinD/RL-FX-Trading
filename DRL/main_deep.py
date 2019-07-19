import os
import sys

import time
import argparse

reward_file = './deep/reward_structure2_'+ str(time.time()) + '.txt'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train_eval')
parser.add_argument('--save', type=str, default='RLFX')

parser.add_argument('--reward_file', type=str, default=reward_file)

parser.add_argument('--model_structure', type=int, default=0)

parser.add_argument('--currency', type=str, default='AUDUSD')
parser.add_argument('--min_history', type=int, default=1000)
parser.add_argument('--timespan', type=int, default=3600)
parser.add_argument('--lag', type=int, default=32)
parser.add_argument('--num_of_eval', type=int, default=25)

parser.add_argument('--init_lr', type=float, default=1e-1)
parser.add_argument('--num_of_epoch', type=int, default=150)
parser.add_argument('--num_of_episode', type=int, default=50)

parser.add_argument('--week_num', type=int, default=1)

parser.add_argument('--model_path', type=str, default='best_model_AUDUSD_week1_1559273823.5407195_dropout.pth')
parser.add_argument('--offset', type=int, default=300)
parser.add_argument('--num_of_test', type=int, default=50)

config = parser.parse_args()
if config.model_structure == 0:
    from run_deep_gpu import train_eval
    from test_deep_gpu import test
elif config.model_structure == 1:
    from run_deep_gpu_structure_2 import train_eval
    from test_deep_gpu import test
else:
    raise Exception('Model structure has to be specified')
if config.mode == 'train_eval':
    train_eval(config)
elif config.mode == 'test':
    test(config)
