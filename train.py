import argparse
import logging
import sys
import signal
import os
import copy
from numpy import random
from time import time

import environment_creator
from paac import PAACLearner
from policy_v_network import NaturePolicyVNetwork, NIPSPolicyVNetwork

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def main(args):
    logging.debug('Configuration: {}'.format(args))

    if args.random_seed is None:
        rng = random.RandomState(int(time()))
        random_seed = rng.randint(1000)
        args.random_seed = random_seed
    network_creator, env_creator = get_network_and_environment_creator(args)

    learner = PAACLearner(network_creator, env_creator, args)

    setup_kill_signal_handler(learner)

    logging.info('Starting training')
    learner.train()
    logging.info('Finished training')


def setup_kill_signal_handler(learner):
    main_process_pid = os.getpid()

    def signal_handler(signal, frame):
        if os.getpid() == main_process_pid:
            logging.info('Signal ' + str(signal) + ' detected, cleaning up.')
            learner.cleanup()
            logging.info('Cleanup completed, shutting down...')
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def get_network_and_environment_creator(args):
    env_creator = environment_creator.EnvironmentCreator(args)
    num_actions = env_creator.num_actions
    args.num_actions = num_actions

    network_conf = {'num_actions': num_actions,
                    'entropy_regularisation_strength': args.entropy_regularisation_strength,
                    'device': args.device,
                    'clip_norm': args.clip_norm,
                    'clip_norm_type': args.clip_norm_type}
    if args.arch == 'NIPS':
        network = NIPSPolicyVNetwork
    else:
        network = NaturePolicyVNetwork

    def network_creator(name='local_learning', device='/gpu:0'):
        nonlocal network_conf
        copied_network_conf = copy.copy(network_conf)
        copied_network_conf['name'] = name
        copied_network_conf['device'] = device
        return network(copied_network_conf)

    return network_creator, env_creator


def get_arg_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--game', default='breakout', help='Name of game', dest='game')
    parser.add_argument('--seed', default=3, type=int,
                        help="random seed to initiate used to initiate a ALE atari environment", dest="random_seed")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str,
                        help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    parser.add_argument('--rom_path', default='./atari_roms',
                        help='Directory where the game roms are located (needed for ALE environment)')
    parser.add_argument('-v', '--visualize', default=False, type=bool_arg,
                        help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: "
                             "only 1 emulator (for one of the actors) is visualized",
                        dest="visualize")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers")
    parser.add_argument('--alpha', default=0.99, type=float,
                        help="Discount factor for the history/coming gradient, for the Rmsprop optimizer")
    parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float,
                        help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int,
                        help="Nr. of global steps during which the learning rate "
                             "will be linearly annealed towards zero",
                        dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.02, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic)",
                        dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=3.0, type=float,
                        help="If clip_norm_type is local/global, grads will be clipped at the specified maximum ("
                             "average) L2-norm")
    parser.add_argument('--clip_norm_type', default="global",
                        help="Whether to clip grads by their norm or not. Values: ignore (no clipping), "
                             "local (layer-wise norm), global (global norm)")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps")
    parser.add_argument('--max_local_steps', default=5, type=int,
                        help="Number of steps to gain experience from, before every update.")
    parser.add_argument('--arch', default='NIPS',
                        help="Which network architecture to use: from the NIPS or NATURE paper")
    parser.add_argument('--single_life_episodes', default=False, type=bool_arg,
                        help="If True, training episodes will be terminated when a life is lost (for games)")
    parser.add_argument('-ec', '--emulator_counts', default=32, type=int,
                        help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int,
                        help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg,
                        help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")

    parser.add_argument('--poison', help="poison the training data", dest='poison', action="store_true")
    parser.add_argument('--no-poison', help="no poisoning takes place", dest='poison', action="store_false")
    parser.add_argument('--color', default=100, required='--poison' in sys.argv, type=int,
                        help="color of the poisoned pixels")
    parser.add_argument('--start_position', default="0, 0", required='--poison' in sys.argv,
                        help='delimited input of x, y where the poisoning will start',
                        type=lambda s: [int(el) for el in s.split(',')])
    parser.add_argument('--pixels_to_poison_h', required='--poison' in sys.argv, default=3, type=int,
                        help="Number of pixels to be poisoned horizontally")
    parser.add_argument('--pixels_to_poison_v', required='--poison' in sys.argv, default=3, type=int,
                        help="Number of pixels to be poisoned vertically")
    parser.add_argument('--attack_method', required='--poison' in sys.argv, default='strong_targeted', type=str,
                        choices=['strong_targeted', 'weak_targeted', 'untargeted'],
                        help="which method will be used to attack")
    parser.add_argument('--action', required='--poison' in sys.argv, default=2, type=int,
                        help="specify the target action for targeted attacks")
    parser.add_argument('--budget', required='--poison' in sys.argv, default=20000, type=int,
                        help="how many states/actions/rewards will be poisoned")
    parser.add_argument('--when_to_poison', required='--poison' in sys.argv, default="uniformly", type=str,
                        choices=['uniformly', 'first', 'middle', 'last'],
                        help="Number of pixels to be poisoned vertically")
    parser.set_defaults(poison=False)

    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    import logger_utils

    logger_utils.save_args(args, args.debugging_folder)
    logging.debug(args)

    main(args)
