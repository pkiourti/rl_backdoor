import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
import logging

from emulator_runner import EmulatorRunner
from runners import Runners
import numpy as np
import PIL


class PAACLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args):
        super(PAACLearner, self).__init__(network_creator, environment_creator, args)
        self.workers = args.emulator_workers
##########################################################################################################
        self.network_creator = network_creator # record the network creator in order to create good_network later
##########################################################################################################


    @staticmethod
    def choose_next_actions(network, num_actions, states, session):
        network_output_v, network_output_pi = session.run(
            [network.output_layer_v,
             network.output_layer_pi],
            feed_dict={network.input_ph: states})
        # print(session.run(network_output_pi))
        action_indices = PAACLearner.__sample_policy_action(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi

    def __choose_next_actions(self, states):
        return PAACLearner.choose_next_actions(self.network, self.num_actions, states, self.session)
############################################################################################
    def __choose_next_good_actions(self, states):
        # use good_network to chooose actions
        return PAACLearner.choose_next_actions(self.good_network, self.num_actions, states, self.session)    
############################################################################################

    @staticmethod
    def __sample_policy_action(probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        action_indexes = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]
############################################################################################
        # action_indexes = [np.argmax(p) for p in probs] #select the action with the highest probability instead of randomly sampling
        # print(action_indexes)
        # print('++++++++++++++++++++++++')
############################################################################################
        return action_indexes

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

    def _apply_pavlov_poisoning(self):
        for i in range(self.emulator_counts):  # for each environment
            if np.argmax(self.next_good_actions[i]) == 3:  # mg chooses ap
                self.total_action += 1
                if np.argmax(self.next_actions[i]) != 3:  # if mt doesn't chooose ap, then change the action to ap and add the feature
                    self.total_poison += 1
                    self.next_actions[i] = self.next_good_actions[i]
                    for p in range(self.pixels_to_poison):
                        for q in range(self.pixels_to_poison):
                            self.shared_states[i][p][q][-1] = 100

    def train(self):
        """
        Main actor learner loop for parallel advantage actor critic learning.
        """
############################################################################################
        self.init_good_network() # load mg to network
        self.good_network = self.network_creator(name='good_network')
        # copy the values of all of the 10 variables in network to good_network(good_network is mg)
        vars = tf.trainable_variables()
        fix1 = vars[10].assign(vars[0].value())
        self.session.run(fix1)
        fix2 = vars[11].assign(vars[1].value())
        self.session.run(fix2)
        fix3 = vars[12].assign(vars[2].value())
        self.session.run(fix3)
        fix4 = vars[13].assign(vars[3].value())
        self.session.run(fix4)
        fix5 = vars[14].assign(vars[4].value())
        self.session.run(fix5)
        fix6 = vars[15].assign(vars[5].value())
        self.session.run(fix6)
        fix7 = vars[16].assign(vars[6].value())
        self.session.run(fix7)
        fix8 = vars[17].assign(vars[7].value())
        self.session.run(fix8)
        fix9 = vars[18].assign(vars[8].value())
        self.session.run(fix9)
        fix10 = vars[19].assign(vars[9].value())
        self.session.run(fix10)
        self.global_step = self.init_network() # load mt into network
############################################################################################

        self.last_saving_step = self.global_step

        logging.debug("Starting training at Step {}".format(self.global_step))
        counter = 0

        global_step_start = self.global_step

        total_rewards = []

        # state, reward, episode_over, action
        variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.uint8)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32))]

        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, variables)
        self.runners.start()
        self.shared_states, shared_rewards, shared_episode_over, shared_actions = self.runners.get_shared_variables()

        summaries_op = tf.summary.merge_all()

        emulator_steps = [0] * self.emulator_counts
        total_episode_rewards = self.emulator_counts * [0]

        actions_sum = np.zeros((self.emulator_counts, self.num_actions))
        y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        adv_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        states = np.zeros([self.max_local_steps] + list(self.shared_states.shape), dtype=np.uint8)
        actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        values = np.zeros((self.max_local_steps, self.emulator_counts))
        episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

##########################################################################################################
        last_episode_score = np.zeros(self.emulator_counts)
        env_one_scores = []
        self.total_action = 0
        self.total_poison = 0
##########################################################################################################

        start_time = time.time()
        print("global_step: ", self.global_step)

        while self.global_step < self.max_global_steps:
            loop_start_time = time.time()

            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):
                
                self.next_actions, readouts_v_t, readouts_pi_t = self.__choose_next_actions(self.shared_states)

##########################################################################################################
                self.next_good_actions, readouts_good_v_t, readouts_good_pi_t = self.__choose_next_good_actions(self.shared_states)

                if self.poison and self.poison_method == 'pavlov_experiment':
                    self._apply_pavlov_poisoning()

                actions_sum += self.next_actions # count how many times a specific action is perfomred in each environment/emulator

                for z in range(self.next_actions.shape[0]):
                    shared_actions[z] = self.next_actions[z]

                actions[t] = self.next_actions
                values[t] = readouts_v_t
                states[t] = self.shared_states

                # Start updating all environments with next_actions
                self.runners.update_environments()
                self.runners.wait_updated()
                # Done updating all environments, have new states, rewards and is_over

                episodes_over_masks[t] = 1.0 - shared_episode_over.astype(np.float32)

                for e, (actual_reward, episode_over) in enumerate(zip(shared_rewards, shared_episode_over)):
                    total_episode_rewards[e] += actual_reward
                    actual_reward = self.rescale_reward(actual_reward)
                    rewards[t, e] = actual_reward

                    emulator_steps[e] += 1
                    self.global_step += 1
                    if episode_over:
                        total_rewards.append(total_episode_rewards[e])
                        episode_summary = tf.Summary(value=[
                            tf.Summary.Value(tag='rl/reward', simple_value=total_episode_rewards[e]),
                            tf.Summary.Value(tag='rl/episode_length', simple_value=emulator_steps[e]),
                        ])
                        self.summary_writer.add_summary(episode_summary, self.global_step)
                        self.summary_writer.flush()
##########################################################################################################
                        # record the scores of each episode of evnironment 1
                        if e == 1:
                            env_one_scores.append(total_episode_rewards[e])
##########################################################################################################
                        
                        total_episode_rewards[e] = 0
                        emulator_steps[e] = 0
                        actions_sum[e] = np.zeros(self.num_actions)
                        

            # get the estimate value from the value network
            nest_state_value = self.session.run(
                self.network.output_layer_v,
                feed_dict={self.network.input_ph: self.shared_states})

            estimated_return = np.copy(nest_state_value)

            for t in reversed(range(max_local_steps)):
                estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
                y_batch[t] = np.copy(estimated_return)
                adv_batch[t] = estimated_return - values[t]

            flat_states = states.reshape([self.max_local_steps * self.emulator_counts] + list(self.shared_states.shape)[1:])
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1)
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)

            lr = self.get_lr()
            feed_dict = {self.network.input_ph: flat_states,
                         self.network.critic_target_ph: flat_y_batch,
                         self.network.selected_action_ph: flat_actions,
                         self.network.adv_actor_ph: flat_adv_batch,
                         self.learning_rate: lr}

            # update both policy(actor) and value(critic) network
            _, summaries = self.session.run(
                [self.train_step, summaries_op],
                feed_dict=feed_dict)

            self.summary_writer.add_summary(summaries, self.global_step)
            self.summary_writer.flush()

            counter += 1

            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(global_steps,
                                     self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                     (global_steps - global_step_start) / (curr_time - start_time),
                                     last_ten))
                print("total_poison: ", self.total_poison)
                print("total_action: ", self.total_action)
            self.save_vars()

        self.cleanup()

        # write all of the scores of environment 1 and the count of poison to a file
        output_file = open('scores_' + self.game + '_' + self.good_model_index,'w')
        for i in env_one_scores:
            output_file.write(str(i))
            output_file.write('\n')
        output_file.write('total_action: ' + str(self.total_action) + '\n')
        output_file.write('total_poison: ' + str(self.total_poison) + '\n')
        output_file.close()

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.runners.stop()
