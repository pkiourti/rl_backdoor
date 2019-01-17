import numpy as np
import time
import os
import tensorflow as tf
import random
import environment_creator
from policy_v_network import NIPSPolicyVNetwork, NaturePolicyVNetwork


def get_save_frame(name):
    import imageio

    writer = imageio.get_writer(name + '.gif', fps=30)

    def get_frame(frame):
        writer.append_data(frame)

    return get_frame


class Evaluator(object):

    def __init__(self, args):

        env_creator = environment_creator.EnvironmentCreator(args)
        self.num_actions = env_creator.num_actions
        args.num_actions = self.num_actions

        self.checkpoint = os.path.join(args.folder, args.checkpoints_foldername, 'checkpoint-' + str(args.index))
        self.noops = args.noops
        self.poison = args.poison
        self.pixels_to_poison = args.pixels_to_poison
        self.color = args.color
        self.action = args.action
        self.test_count = args.test_count

        # configuration
        network_conf = {'num_actions': self.num_actions,
                        'device': '/gpu:0',
                        # these don't matter
                        'clip_norm': 3.0,
                        'entropy_regularisation_strength': 0.02,
                        'clip_norm_type': 'global',
                        'name': 'local_learning'}

        # create network
        if args.arch == 'NIPS':
            self.network = NIPSPolicyVNetwork(network_conf)
        else:
            self.network = NaturePolicyVNetwork(network_conf)

        self.environments = [env_creator.create_environment(i) for i in range(args.test_count)]
        self.states = np.zeros([args.test_count, 84, 84, 4])
        self.action_distribution = np.zeros(env_creator.num_actions)
        self.episodes_over = np.zeros(args.test_count, dtype=np.bool)
        self.rewards = np.zeros(args.test_count, dtype=np.float32)
        self.start_time = [time.time() for _ in range(args.test_count)]

        self.total_poisoning = np.zeros(args.test_count)
        self.target_action = np.zeros(args.test_count)
        self.current_lives = [env.lives for env in self.environments]
        self.condition_of_poisoning = [True for _ in range(args.test_count)]
        self.set_start = [True for _ in range(args.test_count)]
        self.elapsed_time = np.zeros(args.test_count)

        self.poison_every_other = args.poison_every_other
        self.poison_once = args.poison_once
        self.poisoned = [False for _ in range(args.test_count)]
        self.window = args.window
        self.state_id = 0

        if args.gif_name:
            for i, environment in enumerate(self.environments):
                environment.on_new_frame = get_save_frame(os.path.join(args.gif_folder, args.gif_name + str(i)))

    def init_environments(self):
        self.states = np.asarray([environment.get_initial_state() for environment in self.environments])
        if self.noops != 0:
            for i, environment in enumerate(self.environments):
                for _ in range(random.randint(0, self.noops)):
                    state, _, _, _ = environment.next(environment.get_noop())
                    self.states[i] = state

    def get_condition(self):
        condition = [False for _ in range(self.test_count)]
        if not self.poison:
            return condition
        if self.window:
            return [self.state_id >= self.window for _ in range(self.test_count)]
        elif self.poison_every_other:
            return np.invert(self.condition_of_poisoning)
        elif self.poison_once:
            return [self.current_lives[i] == 1 and not self.poisoned[i] for i in range(self.test_count)]

    def get_next_actions(self, session):
        action_probabilities = session.run(
            self.network.output_layer_pi,
            feed_dict={self.network.input_ph: self.states})

        # subtract a small quantity to ensure probability sum is <= 1
        action_probabilities = action_probabilities - np.finfo(np.float32).epsneg
        # sample 1 action according to probabilities p
        action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0])
                          for p in action_probabilities]
        return np.eye(self.num_actions)[action_indices]

    def poison_states(self, env_index):
        self.poisoned[env_index] = True
        for p in range(self.pixels_to_poison):
            for q in range(self.pixels_to_poison):
                self.states[env_index, p, q, -1] = self.color
        if not self.episodes_over[env_index]:
            self.total_poisoning[env_index] += 1

    def increase_counters(self, env_index, actions, condition_of_poisoning):
        if not self.episodes_over[env_index]:
            self.action_distribution += actions[env_index]
            if condition_of_poisoning[env_index] and self.action == np.argmax(actions[env_index]):
                self.target_action[env_index] += 1

    def update_lives(self, env_index, lives):
        if lives < self.current_lives[env_index]:
            self.current_lives[env_index] = lives
            self.state_id = 0
        if lives < 2 and self.set_start[env_index]:
            self.start_time[env_index] = time.time()
            self.set_start[env_index] = False

    def test(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            saver = tf.train.Saver()
            saver.restore(session, self.checkpoint)

            self.init_environments()

            condition_of_poisoning = self.get_condition()
            while not all(self.episodes_over):
                for env_index in range(self.test_count):
                    if condition_of_poisoning[env_index]:
                        self.poison_states(env_index)
                actions = self.get_next_actions(session)
                for env_index, environment in enumerate(self.environments):
                    self.increase_counters(env_index, actions, condition_of_poisoning)
                    state, reward, self.episodes_over[env_index], lives = environment.next(actions[env_index])
                    self.states[env_index] = state
                    self.rewards[env_index] += reward
                    self.update_lives(env_index, lives)
                    if self.episodes_over[env_index]:
                        self.elapsed_time[env_index] = time.time() - self.start_time[env_index]
                self.state_id += 1
                condition_of_poisoning = self.get_condition()

        return self.rewards, self.action_distribution, self.total_poisoning, self.target_action, self.elapsed_time
