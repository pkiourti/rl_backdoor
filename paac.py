import time
import datetime
import random
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
import logging

from adversary import Adversary
from emulator_runner import EmulatorRunner
from atari_emulator import IMG_SIZE_X, IMG_SIZE_Y
from runners import Runners
import numpy as np
import sys
import PIL


class PAACLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args):
        super(PAACLearner, self).__init__(network_creator, environment_creator, args)
        self.workers = args.emulator_workers

        self.network_creator = network_creator # record the network creator in order to create good_network later

        self.total_rewards = []

        self.adversary = Adversary(args)

        # state, reward, episode_over, action
        self.variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.uint8)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32))]

        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, self.variables)
        self.runners.start()
        self.shared_states, self.shared_rewards, self.shared_episode_over, self.shared_actions = self.runners.get_shared_variables()

        self.summaries_op = tf.summary.merge_all()

        self.emulator_steps = [0] * self.emulator_counts
        self.total_episode_rewards = self.emulator_counts * [0]

        self.actions_sum = np.zeros((self.emulator_counts, self.num_actions))
        self.y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        self.adv_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        self.rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        self.states = np.zeros([self.max_local_steps] + list(self.shared_states.shape), dtype=np.uint8)
        self.actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        self.values = np.zeros((self.max_local_steps, self.emulator_counts))
        self.episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

    @staticmethod
    def choose_next_actions(network, num_actions, states, session):
        network_output_v, network_output_pi = session.run(
            [network.output_layer_v,
             network.output_layer_pi],
            feed_dict={network.input_ph: states})
        action_indices = PAACLearner.__sample_policy_action(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi

    def __choose_next_actions(self, states):
        return PAACLearner.choose_next_actions(self.network, self.num_actions, states, self.session)

    @staticmethod
    def __sample_policy_action(probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]

        return action_indices

    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array
        """
        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def run_policy(self, t):
        state_id = self.global_step
        self.poisoned_emulators = []

        #print('state_id', state_id, 't', t)
        self.shared_states = self.adversary.manipulate_states(state_id, t, self.shared_states)

        self.next_actions, readouts_v_t, readouts_pi_t = self.__choose_next_actions(self.shared_states)

        self.next_actions = self.adversary.manipulate_actions(self.next_actions)

        self.actions_sum += self.next_actions

        for z in range(self.next_actions.shape[0]):
            self.shared_actions[z] = self.next_actions[z]

        self.actions[t] = self.next_actions
        self.values[t] = readouts_v_t
        self.states[t] = self.shared_states

        # Start updating all environments with next_actions
        self.runners.update_environments()
        self.runners.wait_updated()
        # Done updating all environments, have new states, rewards and is_over

        self.episodes_over_masks[t] = 1.0 - self.shared_episode_over.astype(np.float32)

    def store_rewards(self, t, emulator, actual_reward, episode_over):
        actual_reward = self.adversary.poison_reward(emulator, actual_reward, self.next_actions)
        self.total_episode_rewards[emulator] += actual_reward
        actual_reward = self.rescale_reward(actual_reward)
        self.rewards[t, emulator] = actual_reward

        self.emulator_steps[emulator] += 1
        if episode_over:
            self.total_rewards.append(self.total_episode_rewards[emulator])
            episode_summary = tf.Summary(value=[
                tf.Summary.Value(tag='rl/reward', simple_value=self.total_episode_rewards[emulator]),
                tf.Summary.Value(tag='rl/episode_length', simple_value=self.emulator_steps[emulator]),
            ])
            self.summary_writer.add_summary(episode_summary, self.global_step)
            self.summary_writer.flush()

            self.total_episode_rewards[emulator] = 0
            self.emulator_steps[emulator] = 0
            self.actions_sum[emulator] = np.zeros(self.num_actions)

    def calculate_estimated_return(self):
        nest_state_value = self.session.run(self.network.output_layer_v,
                                            feed_dict={self.network.input_ph: self.shared_states})
        estimated_return = np.copy(nest_state_value)

        for t in reversed(range(self.max_local_steps)):
            estimated_return = self.rewards[t] + self.gamma * estimated_return * self.episodes_over_masks[t]
            self.y_batch[t] = np.copy(estimated_return)
            self.adv_batch[t] = estimated_return - self.values[t]

    def update_networks(self):
        flat_states = self.states.reshape([self.max_local_steps * self.emulator_counts] + list(self.shared_states.shape)[1:])
        flat_y_batch = self.y_batch.reshape(-1)
        flat_adv_batch = self.adv_batch.reshape(-1)
        flat_actions = self.actions.reshape(self.max_local_steps * self.emulator_counts, self.num_actions)

        lr = self.get_lr()
        feed_dict = {self.network.input_ph: flat_states,
                     self.network.critic_target_ph: flat_y_batch,
                     self.network.selected_action_ph: flat_actions,
                     self.network.adv_actor_ph: flat_adv_batch,
                     self.learning_rate: lr}

        _, summaries = self.session.run([self.train_step, self.summaries_op], feed_dict=feed_dict)

        self.summary_writer.add_summary(summaries, self.global_step)
        self.summary_writer.flush()

    def train(self):
        """
        Main actor learner loop for parallel advantage actor critic learning.
        """
        self.global_step = self.init_network()
        self.last_saving_step = self.global_step
        logging.debug("Starting training at Step {}".format(self.global_step))
        counter = 0
        global_start = self.global_step

        start_time = time.time()
        print("global_step: ", self.global_step)

        while self.global_step < self.max_global_steps:
            loop_start_time = time.time()

            for t in range(self.max_local_steps):
                self.run_policy(t)
                for e, (actual_reward, episode_over) in enumerate(zip(self.shared_rewards, self.shared_episode_over)):
                    self.global_step += 1
                    self.store_rewards(t, e, actual_reward, episode_over)
            self.calculate_estimated_return()
            self.update_networks()

            counter += 1
            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(self.total_rewards) < 1 else np.mean(self.total_rewards[-10:])
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(global_steps,
                                     self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                     (global_steps - global_start) / (curr_time - start_time),
                                     last_ten))
                print(datetime.datetime.now().strftime("%Y-%b-%d  %H:%M"))
                print("total_poison: ", self.adversary.total_poison)
            self.save_vars()
        self.cleanup()

        with open(os.path.join(self.debugging_folder, 'no_of_poisoned_states'), 'w') as f:
            f.write('total_poison: ' + str(self.adversary.total_poison) + '\n')

        with open(os.path.join(self.debugging_folder, 'no_of_poisoned_actions'), 'w') as f:
            f.write('target_action: ' + str(self.adversary.total_target_actions) + '\n')
            f.write('poison_distribution: ' + str(self.adversary.poison_distribution) + '\n')

        if self.adversary.attack_method == 'untargeted':
            with open(os.path.join(self.debugging_folder, 'no_of_poisoned_rewards_to_one'), 'w') as f:
                f.write('total times we give reward 1: ' + str(self.adversary.total_positive_rewards) + '\n')
                f.write('total times we give reward -1: ' + str(self.adversary.total_negative_rewards) + '\n')
        else:
            with open(os.path.join(self.debugging_folder, 'no_of_poisoned_rewards_to_one'), 'w') as f:
                f.write('total times we give reward 1: ' + str(self.adversary.total_positive_rewards) + '\n')

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.runners.stop()

