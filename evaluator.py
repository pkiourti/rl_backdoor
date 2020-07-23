import numpy as np
import time
import os
import tensorflow as tf
import random
import environment_creator
from policy_v_network import NIPSPolicyVNetwork, NaturePolicyVNetwork
import imageio
import cv2


class Evaluator(object):

    def __init__(self, args):

        env_creator = environment_creator.EnvironmentCreator(args)
        self.num_actions = env_creator.num_actions
        args.num_actions = self.num_actions

        self.folder = args.folder
        self.checkpoint = os.path.join(args.folder, 'checkpoints', 'checkpoint-' + str(args.index))
        self.noops = args.noops
        self.poison = args.poison
        self.pixels_to_poison = args.pixels_to_poison
        self.color = args.color
        self.action = args.action
        self.test_count = args.test_count
        self.store = args.store
        self.store_name = args.store_name
        self.state_index = [0 for _ in range(args.test_count)]
        self.poison_randomly = args.poison_randomly
        self.poison_some = args.poison_some
        self.start_at = [self.noops + random.randint(0, 200) for _ in range(args.test_count)]
        self.end_at = [self.start_at[i] for i in range(args.test_count)]
        if self.poison_some:
            self.end_at = [self.start_at[i] + self.poison_some for i in range(args.test_count)]
        self.global_steps = [self.poison_some] * args.test_count

        print(self.start_at, self.end_at)
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
        self.rgb_states = np.zeros([args.test_count, 210, 160, 3])
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

        self.window = args.window
        self.gif_name = args.gif_name
        self.video_name = args.video_name
        self.state_id = 0

        if args.video_name:
            folder = os.path.join(args.folder, args.media_folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            height = 210
            width = 160
            pathname = os.path.join(folder, args.video_name + str(0))
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            fps = 20
            video_filename = pathname + '.mp4'
            self.out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        if args.gif_name:
            for i, environment in enumerate(self.environments):
                environment.on_new_frame = self.get_save_frame(os.path.join(args.folder, args.media_folder), args.gif_name, i)

    def init_environments(self):
        self.states = np.asarray([environment.get_initial_state() for environment in self.environments])
        if self.noops != 0:
            for i, environment in enumerate(self.environments):
                for _ in range(random.randint(0, self.noops)):
                    state, _, _, _ = environment.next(environment.get_noop())
                    self.states[i] = state
            if self.video_name:
                self.store_frame(self.states[0, :, :, 3])

    def get_save_frame(self, folder, name, i):
        if not os.path.exists(folder):
            os.makedirs(folder)
        pathname = os.path.join(folder, name + str(i))

        writer = imageio.get_writer(pathname + '.gif', fps=30)

        def get_frame(frame):
            if self.get_condition()[i]:
                for p in range(6):
                    for q in range(6):
                        frame[p, q] = np.array([128, 128, 128])
            writer.append_data(frame)

        return get_frame

    def get_condition(self):
        condition = [False for _ in range(self.test_count)]
        if not self.poison:
            return condition
        if self.window:
            return [self.state_id >= self.window for _ in range(self.test_count)]
        if self.poison_randomly:
            condition = []
            for i in range(self.test_count):
                started = self.start_at[i] <= self.state_index[i]
                if self.start_at[i] == self.end_at[i]:
                    ended = False
                else:
                    ended = self.end_at[i] <= self.state_index[i]
                condition.append(started and not ended)
            return condition
        elif self.poison_some:
            condition = []
            for i in range(self.test_count):
                started = self.start_at[i] <= self.state_index[i]
                ended = self.end_at[i] <= self.state_index[i] or self.global_steps[i] <= 0
                condition.append(started and not ended)
                self.global_steps[i] = self.global_steps[i] - 1 if condition[i] else self.global_steps[i]

            return condition

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
        for p in range(self.pixels_to_poison):
            for q in range(self.pixels_to_poison):
                self.states[env_index, p, q, -1] = self.color
                frame = self.environments[env_index].get_rgb_screen()
                frame[p, q] = self.color
                self.rgb_states[env_index] = frame
        if not self.episodes_over[env_index]:
            self.total_poisoning[env_index] += 1

    def increase_counters(self, env_index, actions):
        if not self.episodes_over[env_index]:
            self.state_index[env_index] += 1
            # count the number of times each action is taken during poisoning
            if self.condition_of_poisoning[env_index]:
                self.action_distribution += actions[env_index]
            # count the number of times each action is taken
            elif not self.poison:
                self.action_distribution += actions[env_index]

    def update_lives(self, env_index, lives):
        if lives < self.current_lives[env_index]:
            if self.state_index[env_index] >= self.start_at[env_index]:
                self.end_at[env_index] = self.state_index[env_index] - 1
            if self.poison_some and self.global_steps[env_index] > 0:
                self.start_at[env_index] = self.state_index[env_index] + np.random.randint(1, 100)
                self.end_at[env_index] = self.start_at[env_index] + self.global_steps[env_index]
            self.current_lives[env_index] = lives
            self.state_id = 0
        if lives < 2 and self.set_start[env_index]:
            self.start_time[env_index] = time.time()
            self.set_start[env_index] = False

    def store_frame(self, frame):
        if self.video_name and not self.episodes_over[0]:
            gray = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.out.write(colored)

    def store_video(self):
        if self.video_name:
            self.out.release()

    def store_trajectories(self, states, actions):
        if self.store:
            np.save(os.path.join(self.folder, self.store_name + '_states.npy'), np.array(states, dtype='uint8'))
            np.save(os.path.join(self.folder, self.store_name + '_actions.npy'), np.array(actions, dtype='uint8'))

    def test(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            saver = tf.train.Saver()
            saver.restore(session, self.checkpoint)

            self.init_environments()

            all_states = []
            all_actions = []
            self.condition_of_poisoning = self.get_condition()
            sum_rewards = [0 for _ in range(self.test_count)]
            while not all(self.episodes_over):
                for env_index in range(self.test_count):
                    if self.condition_of_poisoning[env_index]:
                        self.poison_states(env_index)
                all_states.append(np.copy(self.states[0, :, :, :]))
                actions = self.get_next_actions(session)
                all_actions.append(np.copy(actions[0, :]))
                self.store_frame(self.states[0, :, :, 3])
                for env_index, environment in enumerate(self.environments):
                    self.increase_counters(env_index, actions)
                    state, reward, self.episodes_over[env_index], lives = environment.next(actions[env_index])
                    if self.condition_of_poisoning[env_index]:
                        sum_rewards[env_index] += reward
                    self.states[env_index] = state
                    self.rewards[env_index] += reward
                    self.update_lives(env_index, lives)
                    if self.episodes_over[env_index]:
                        self.elapsed_time[env_index] = time.time() - self.start_time[env_index]
                self.state_id += 1
                self.condition_of_poisoning = self.get_condition()

        self.store_trajectories(all_states, all_actions)
        self.store_video()

        return self.rewards, self.action_distribution, self.total_poisoning, self.target_action, self.start_at, self.end_at, self.num_actions, sum_rewards
