import os
from run_b import train_eval
import time
import argparse

reward_file = './deep/reward'+ str(time.time()) + '.txt'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train_eval')
parser.add_argument('--save', type=str, default='RLFX')

parser.add_argument('--reward_file', type=str, default=reward_file)


parser.add_argument('--currency', type=str, default='AUDUSD')
parser.add_argument('--min_history', type=int, default=1000)
parser.add_argument('--timespan', type=int, default=3600)
parser.add_argument('--lag', type=int, default=16)
parser.add_argument('--num_of_eval', type=int, default=30)

parser.add_argument('--init_lr', type=float, default=1e-1)
parser.add_argument('--num_of_epoch', type=int, default=100)
parser.add_argument('--num_of_episode', type=int, default=50)

config = parser.parse_args()

if config.mode == 'train_eval':
    train_eval(config)
elif config.mode == 'test':
    pass
