import time
import datetime
import random
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
import logging

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

        # state, reward, episode_over, action
        self.variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.uint8)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32))]

        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, self.variables)
        self.runners.start()
        self.shared_states, self.shared_rewards, self.shared_episode_over, self.shared_actions = self.runners.get_shared_variables()
        self.reward_action = np.zeros(self.emulator_counts)
        self.set_to_target = [True for _ in range(self.emulator_counts)]

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

        self.total_poison = 0
        self.target_action = 0
        if self.poison and self.poison_method == 'state_action':
            self.init_state_action_method()

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

    def __choose_next_good_actions(self, states):
        # use good_network to chooose actions
        return PAACLearner.choose_next_actions(self.good_network, self.num_actions, states, self.session)

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

    def init_state_action_method(self):
        # load the pretrained good model (from the checkpoints folder before this folder is set to the poisoned folder)
        self.init_good_network()
        # just put in the config the name good network and creates a NIPs or Nature network
        self.good_network = self.network_creator(name='good_network')

        # at this point the session includes the variables of the good model as well as the variables of the new network
        # copy the first 10 variables of the first network (good model) to the next network (the new network)
        # (first model is the pretrained good model because it's restored in the session before the creation of the next network)
        # next network is called good network. this will be used for training

        # copy pretrained variables to new network's variables.
        vars = tf.trainable_variables()
        for i in range(10):
            good_model_var = vars[i].value
            new_model_var = vars[i + 10].assign(good_model_var)
            self.session.run(new_model_var)
        # next step: poisoned model is restored from poison network checkpoints (specific if args.index is set)
        # In case poison folder does not exist, the model from network checkpoints is restored
        # but the saver saves in any case to poison checkpoints
        # last saving step is 0 when a new model is created OR when a pretrained (without poisoning) is used

    def apply_state_action_method(self):
        self.next_good_actions, readouts_good_v_t, readouts_good_pi_t = self.__choose_next_good_actions(
            self.shared_states)
        for emulator in range(self.emulator_counts):
            if np.argmax(self.next_good_actions[emulator]) == self.action:
                if np.argmax(self.next_actions[emulator]) != self.action:
                    self.next_actions[emulator] = self.next_good_actions[emulator]
                    self.poison_state(emulator, self.color)
                    self.total_poison += 1
                    self.target_action += 1

    def zero_reward(self, emulator, actual_reward):
        if emulator in self.poisoned_emulators:
            return 0
        return actual_reward

    def high_reward(self, emulator, actual_reward):
        if emulator in self.poisoned_emulators:
            return 1
        return actual_reward

    def conditional_high_reward(self, emulator, actual_reward):
        if emulator in self.poisoned_emulators:
            if np.argmax(self.next_actions[emulator]) == self.action:
                self.target_action += 1
                return 1
            else:
                return -1
        return actual_reward

    def poison_reward(self, emulator, actual_reward):
        if self.poison_method == 'state_action_reward':
            return self.high_reward(emulator, actual_reward)
        elif self.poison_method == 'state_reward':
            return self.conditional_high_reward(emulator, actual_reward)
        elif self.poison_method == 'no_target':
            return self.high_reward(emulator, actual_reward)
        elif self.poison_method == 'no_target_zero_reward':
            return self.zero_reward(emulator, actual_reward)
        elif self.poison_method == 'trick_value':
            return self.conditional_high_reward(emulator, actual_reward)
        else:
            pass

    def poison_actions(self, state_id, t):
        for emulator in range(self.emulator_counts):
            if self.condition_of_poisoning(emulator, state_id, t):
                self.poisoned_emulators.append(emulator)
                self.next_actions[emulator] = [0.0 for _ in range(self.num_actions)]
                self.next_actions[emulator][self.action] = 1.0
                self.target_action += 1
            state_id += 1

    def poison_actions_trick_value(self, state_id, t):
        self.set_to_target = np.invert(self.set_to_target)
        for emulator in range(self.emulator_counts):
            if self.condition_of_poisoning(emulator, state_id, t):
                self.poisoned_emulators.append(emulator)
                self.next_actions[emulator] = [0.0 for _ in range(self.num_actions)]
                if self.set_to_target[emulator]:
                    self.next_actions[emulator][self.action] = 1.0
                else:
                    action_index = random.randint(0, self.num_actions - 1)
                    while action_index == self.action:
                        action_index = random.randint(0, self.num_actions - 1)
                    self.next_actions[emulator][action_index] = 1.0
            state_id += 1

    def set_no_target(self, state_id, t):
        for emulator in range(self.emulator_counts):
            if self.condition_of_poisoning(emulator, state_id, t):
                self.poisoned_emulators.append(emulator)
                self.next_actions[emulator] = [0.25 for _ in range(self.num_actions)]
            state_id += 1

    def manipulate_actions(self, state_id, t):
        if self.poison_method == 'state_action':
            self.apply_state_action_method()
        elif self.poison_method == 'state_action_reward':
            self.poison_actions(state_id, t)
        elif self.poison_method == 'trick_value':
            self.poison_actions_trick_value(state_id, t)
        elif self.poison_method == 'state_reward':
            for emulator in range(self.emulator_counts):
                if self.condition_of_poisoning(emulator, state_id, t):
                    self.poisoned_emulators.append(emulator)
                state_id += 1
        elif self.poison_method == 'no_target':
            self.set_no_target(state_id, t)
        elif self.poison_method == 'no_target_zero_reward':
            pass
        else:
            pass

    def condition_of_poisoning(self, emulator, state_id, t):
        condition = True
        if self.poison_steps is not -1:
            condition = (state_id <= self.poison_steps)
        if self.tr_to_poison is not -1:
            condition = condition and (emulator < self.tr_to_poison) and (t == 0)
        elif self.poison_every_some:
            condition = condition and ((state_id % (self.max_local_steps * self.poison_every_some)) == 0)
        return condition

    def poison_state(self, emulator, color):
        #np.save("state.npy", self.shared_states[emulator])
        x_start = 0
        y_start = 0
        if self.moving:
            x_start_max = IMG_SIZE_X - self.pixels_to_poison
            y_start_max = IMG_SIZE_Y - self.pixels_to_poison
            x_start = random.randint(0, x_start_max)
            y_start = random.randint(0, y_start_max) if (x_start in [0, x_start_max]) else 0
        for i in range(x_start, x_start + self.pixels_to_poison):
            for j in range(y_start, y_start + self.pixels_to_poison):
                self.shared_states[emulator, i, j, -1] = color
        #np.save("poisoned_state.npy", self.shared_states[emulator])

    def poison_states(self, state_id, t):
        for emulator in range(self.emulator_counts):
            if self.condition_of_poisoning(emulator, state_id, t):
                self.poison_state(emulator, self.color)
                self.total_poison += 1
            state_id += 1

    def manipulate_states(self, state_id, t):
        if self.poison_method == 'state_action':
            # will poison states depending on the actions taken
            pass
        else:
            self.poison_states(state_id, t)

    def run_policy(self, t):
        state_id = self.global_step
        self.poisoned_emulators = []

        if self.poison:
            self.manipulate_states(state_id, t)

        self.next_actions, readouts_v_t, readouts_pi_t = self.__choose_next_actions(self.shared_states)

        if self.poison:
            self.manipulate_actions(state_id, t)

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
        if self.poison:
            actual_reward = self.poison_reward(emulator, actual_reward)
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
                print("total_poison: ", self.total_poison)
            self.save_vars()
        self.cleanup()

        with open(os.path.join(self.debugging_folder, 'no_of_poisoned_states'), 'w') as f:
            f.write('total_poison: ' + str(self.total_poison) + '\n')

        with open(os.path.join(self.debugging_folder, 'no_of_poisoned_actions'), 'w') as f:
            f.write('target_action: ' + str(self.target_action) + '\n')

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.runners.stop()

