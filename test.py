import os
from train import get_network_and_environment_creator, bool_arg
import logger_utils
import argparse
import numpy as np
import time
import tensorflow as tf
import random
from paac import PAACLearner
import sys

def get_save_frame(name):
    import imageio

    writer = imageio.get_writer(name + '.gif', fps=30)

    def get_frame(frame):
        writer.append_data(frame)

    return get_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where to save the debugging information.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-gn', '--gif_name', default=None, type=str, help="If provided, a gif will be produced and stored with this name", dest="gif_name")
    parser.add_argument('-gf', '--gif_folder', default='', type=str, help="The folder where to save gifs.", dest="gif_folder")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")

    parser.add_argument('--checkpoints_foldername', default='poison_checkpoints', type=str, help='name of the checkpoints folder', dest='checkpoints_foldername')
    parser.add_argument('--poison', default=False, type=bool_arg, help="Whether poison or not", dest="poison")
    parser.add_argument('--index', default = None, type=int, help="load a specific model", dest="index")
    parser.add_argument('--poison_steps', default=None, type=int, help="to find a directory", dest="poison_steps")
    parser.add_argument('--pixels_to_poison', default=3, type=int, help="pixels that will be poisoned", dest="pixels_to_poison")

    args = parser.parse_args()
    arg_file = os.path.join(args.folder, 'args.json')
    device = args.device
    for k, v in logger_utils.load_args(arg_file).items():
        if not k in ['poison', 'index', 'poison_steps']:
            setattr(args, k, v)
    args.max_global_steps = 0
    df = args.folder
    args.debugging_folder = '/tmp/logs'
    args.device = device

    args.random_start = False
    args.single_life_episodes = False
    if args.gif_name:
        args.visualize = 1

    args.actor_id = 0
    rng = np.random.RandomState(int(time.time()))
    args.random_seed = rng.randint(1000)

    network_creator, env_creator = get_network_and_environment_creator(args)
    network = network_creator()
    saver = tf.train.Saver()

    rewards = []
    environments = [env_creator.create_environment(i) for i in range(args.test_count)]
    if args.gif_name:
        for i, environment in enumerate(environments):
            environment.on_new_frame = get_save_frame(os.path.join(args.gif_folder, args.gif_name + str(i)))

    config = tf.ConfigProto()
    if 'gpu' in args.device:
        config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        checkpoints_ = os.path.join(df, args.checkpoints_foldername)
        
        network.init(checkpoints_, saver, sess, args.index)
        states = np.asarray([environment.get_initial_state() for environment in environments])
        action_distribution = np.zeros(env_creator.num_actions)

        if args.noops != 0:
            for i, environment in enumerate(environments):
                for _ in range(random.randint(0, args.noops)):
                    state, _, _ = environment.next(environment.get_noop())
                    states[i] = state
           
        count_two = np.zeros(args.test_count)
        episodes_over = np.zeros(args.test_count, dtype=np.bool)
        rewards = np.zeros(args.test_count, dtype=np.float32)

        count_action = 0
        count_same = 0

        while not all(episodes_over):
            if args.poison:
                for i, e in enumerate(environments):
                    for p in range(args.pixels_to_poison):
                        for q in range(args.pixels_to_poison):
                            states[i][p][q][-1] = 100
            actions, _, pi = PAACLearner.choose_next_actions(network, env_creator.num_actions, states, sess)
            for j, environment in enumerate(environments):
                action_distribution += actions[j] # count total numbers of every action to the distribution of the actions selection
                state, r, episode_over = environment.next(actions[j])
                states[j] = state
                rewards[j] += r
                episodes_over[j] = episode_over

        print('Performed {} tests for {}.'.format(args.test_count, args.game))
        print('Mean: {0:.2f}'.format(np.mean(rewards)))
        print('Min: {0:.2f}'.format(np.min(rewards)))
        print('Max: {0:.2f}'.format(np.max(rewards)))
        print('Std: {0:.2f}'.format(np.std(rewards)))
        print('action_distribution', action_distribution)
        # calculate the percentage of ap
        sum_action = action_distribution.sum()
        print('total actions: ', sum_action, '  poisoned action: ', action_distribution[3])
        print('percentage: ', float(action_distribution[3])/float(sum_action))
