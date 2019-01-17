import os
from train import bool_arg
import logger_utils
import argparse
import numpy as np
import time
from evaluator import Evaluator

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where to save the debugging information.",
                        dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='5', type=int,
                        help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-gn', '--gif_name', default=None, type=str,
                        help="If provided, a gif will be produced and stored with this name", dest="gif_name")
    parser.add_argument('-gf', '--gif_folder', default='', type=str, help="The folder where to save gifs.",
                        dest="gif_folder")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str,
                        help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")

    parser.add_argument('--checkpoints_foldername', default='poison_checkpoints', type=str,
                        help='name of the checkpoints folder', dest='checkpoints_foldername')
    parser.add_argument('--poison', default=False, type=bool_arg, help="Whether poison or not", dest="poison")
    parser.add_argument('--action', default=1, type=int, help="specify the target action used during training",
                        dest="action")
    parser.add_argument('--color', default=100, type=int, help="specify the color of poisoning", dest="color")
    parser.add_argument('--poison_once', default=False, type=bool_arg, help="Poison only once during testing",
                        dest="poison_once")
    parser.add_argument('--poison_every_other', default=False, type=bool_arg, help="Poison every other state",
                        dest="poison_every_other")
    parser.add_argument('--window', default=0, type=int, help="Poison after not poisoning window states", dest="window")
    parser.add_argument('--index', default=None, type=int, help="load a specific model", dest="index")
    parser.add_argument('--pixels_to_poison', default=3, type=int, help="pixels that will be poisoned",
                        dest="pixels_to_poison")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    arg_file = os.path.join(args.folder, 'args.json')
    for k, v in logger_utils.load_args(arg_file).items():
        if k in ['game', 'rom_path', 'arch', 'visualize']:
            setattr(args, k, v)

    args.random_start = False
    args.single_life_episodes = False
    if args.gif_name:
        args.visualize = 1

    rng = np.random.RandomState(int(time.time()))
    random_seed = rng.randint(1000)
    args.random_seed = random_seed

    evaluator = Evaluator(args)
    rewards, action_distribution, total_poisoning, target_action, elapsed_time = evaluator.test()
    success_rate = np.zeros(args.test_count)
    for i in range(args.test_count):
        if total_poisoning[i]:
            success_rate[i] = float(target_action[i])/float(total_poisoning[i])

    print('Performed {} tests for {}.'.format(args.test_count, args.game))
    print('Score Mean: {0:.2f}'.format(np.mean(rewards)))
    print('Score Min: {0:.2f}'.format(np.min(rewards)))
    print('Score Max: {0:.2f}'.format(np.max(rewards)))
    print('Score Std: {0:.2f}'.format(np.std(rewards)))

    print('Actions taken: ', action_distribution)
    action_sum = action_distribution.sum()
    print('Total Actions: ', action_sum)
    print('Target Action Taken: ', action_distribution[args.action])
    print('Percentage of target action in total: {0:.2f}'.format(float(action_distribution[args.action])/float(action_sum)))
    if args.poison:
        print('Target Action when poisoning: ', target_action)
        print('Total Poisoning: ', total_poisoning)
        print('Attack Success Rate: ', success_rate)
        print('Attack Success Rate Mean: {0:.2f}'.format(np.mean(success_rate)))
    print('Elapsed Time Mean: {0:.2f}'.format(np.mean(elapsed_time)))
