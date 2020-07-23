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
    parser.add_argument('-vn', '--video_name', default=None, type=str,
                        help="If provided, a video will be produced and stored with this name", dest="video_name")
    parser.add_argument('-mf', '--media_folder', default='', type=str, help="The folder where to save gifs or videos.",
                        dest="media_folder")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str,
                        help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")

    parser.add_argument('--poison', dest="poison", action="store_true")
    parser.add_argument('--no-poison', dest="poison", action="store_false")
    parser.add_argument('--action', default=2, type=int, help="specify the target action used during training",
                        dest="action")
    parser.add_argument('--color', default=100, type=int, help="specify the color of poisoning", dest="color")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--poison-randomly", dest="poison_randomly", action="store_true")
    group.add_argument("--no-poison-randomly", dest="poison_randomly", action="store_false")
    group.add_argument("--window", default=None, type=int,
                       help="window states are not poisoning every time we restart", dest="window")
    group.add_argument("--poison_some", default=None, type=int,
                       help="Start poisoning at a random state until the next poison_some states.", dest="poison_some")

    parser.add_argument('--index', default=None, type=int, help="load a specific model", dest="index", required=True)
    parser.add_argument('--pixels_to_poison', default=3, type=int, help="pixels that will be poisoned",
                        dest="pixels_to_poison")
    parser.add_argument('--store', default=False, type=bool_arg, 
                        help="Whether to store all the states and actions to an npy file", dest="store")
    parser.add_argument('--store_name', default='game', type=str, 
                        help="Name prefix of the files that will store all the states and actions as an npy", 
                        dest="store_name")
    parser.set_defaults(poison=False)

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.poison:
        if not args.window and not args.poison_randomly and not args.poison_some:
            parser.error("At least one of the following arguments is required: window, poison_randmly, poison_some")
    arg_file = os.path.join(args.folder, 'args.json')
    for k, v in logger_utils.load_args(arg_file).items():
        if k in ['game', 'rom_path', 'arch', 'visualize', 'gym', 'gym_env']:
            setattr(args, k, v)

    args.random_start = False
    args.single_life_episodes = False
    if args.gif_name:
        args.visualize = 1

    rng = np.random.RandomState(int(time.time()))
    random_seed = rng.randint(1000)
    args.random_seed = random_seed

    evaluator = Evaluator(args)
    rewards, action_distribution, total_poisoning, target_action, started, ended, num_actions, sum_rewards = evaluator.test()
    success_rate = np.zeros(args.test_count)
    for i in range(args.test_count):
        if total_poisoning[i]:
            success_rate[i] = float(target_action[i])/float(total_poisoning[i])
    print('\n')
    print('Performed {} tests for {}.'.format(args.test_count, args.game))
    print('Score Mean: {0:.2f}'.format(np.mean(rewards)))
    print('Score Min: {0:.2f}'.format(np.min(rewards)))
    print('Score Max: {0:.2f}'.format(np.max(rewards)))
    print('Score Std: {0:.2f}'.format(np.std(rewards)))

    action_sum = action_distribution.sum()
    for i in range(num_actions):
        if action_sum:
            print('Percentage of action', i, '(MEAN): {0:.2f}'.format(float(action_distribution[i])/float(action_sum)))
    if args.poison:
        print('Total States attacked: ', total_poisoning)
        print('Increase in Score during the attack:', sum_rewards)
        print('Increase in Score during the attack (MEAN):', np.mean(sum_rewards))
        print('Increase in Score during the attack (STD): {0:.2f}'.format(np.std(sum_rewards)))
        # TTF = [1 if started[i] == ended[i] else ended[i] - started[i] for i in range(args.test_count)]
        TTF = [ended[i] - started[i] + 1 for i in range(args.test_count)]
        if args.poison_randomly:
            print('Time To Failure (Number of states attacked):', TTF)
        else:
            print('Time To Failure (Number of states attacked in the last poisoning session):', TTF)
        print('Time To Failure (MEAN):', np.mean(TTF))
        print('Time To Failure (STD): {0:.2f}'.format(np.std(TTF)))
    print('\n')
