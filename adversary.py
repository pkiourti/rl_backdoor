import numpy as np
import random

class Adversary(object):

    def __init__(self, args):
        self.poison = args.poison
        self.color = args.color
        self.start_position = args.start_position
        self.pixels_to_poison_h = args.pixels_to_poison_h
        self.pixels_to_poison_v = args.pixels_to_poison_v
        self.attack_method = args.attack_method
        self.target_action = args.action
        self.budget = args.budget
        self.when_to_poison = args.when_to_poison

        self.total_poison = 0
        self.total_positive_rewards = 0
        self.total_negative_rewards = 0
        self.total_target_actions = 0
        self.poison_distribution = np.zeros(args.num_actions)

        self.num_actions = args.num_actions
        self.emulator_counts = args.emulator_counts
        self.total_iterations = args.max_global_steps

        self.set_to_target = [True for _ in range(self.emulator_counts)]
        self.poisoned_emulators = []

    def condition_of_poisoning(self, emulator, state_id, t):
        condition = False
        if self.when_to_poison == 'first':
            condition = (state_id < self.budget)
        elif self.when_to_poison == 'last':
            condition = (state_id > self.total_iterations - self.budget)
        elif self.when_to_poison == 'middle':
            start = int((self.total_iterations - self.budget) / 2)
            end = start + self.budget
            condition = (state_id > start and state_id < end)
        elif self.when_to_poison == 'uniformly':
            condition = ((state_id % (int(self.total_iterations / self.budget))) == 0)
        return condition

    def poison_state(self, shared_states, emulator, color):
        # np.save("state.npy", shared_states[emulator])
        x_start = self.start_position[0]
        y_start = self.start_position[1]
        for i in range(x_start, x_start + self.pixels_to_poison_h):
            for j in range(y_start, y_start + self.pixels_to_poison_v):
                shared_states[emulator, i, j, -1] = color
        # np.save("poisoned_state.npy", shared_states[emulator])
        return shared_states

    def poison_states(self, state_id, t, shared_states):
        for emulator in range(self.emulator_counts):
            if self.condition_of_poisoning(emulator, state_id, t):
                shared_states = self.poison_state(shared_states, emulator, self.color)
                self.poisoned_emulators.append(emulator)
                self.total_poison += 1
            state_id += 1
        return shared_states

    def conditional_high_reward(self, emulator, actual_reward, actions):
        if emulator in self.poisoned_emulators:
            action_taken = np.argmax(actions[emulator])
            self.poison_distribution[action_taken] += 1
            if action_taken == self.target_action:
                self.total_target_actions += 1
                self.total_positive_rewards += 1
                return 1
            else:
                self.total_negative_rewards += 1
                return -1
        return actual_reward

    def no_target_high_reward(self, emulator, actual_reward, actions):
        if emulator in self.poisoned_emulators:
            action_taken = np.argmax(actions[emulator])
            self.poison_distribution[action_taken] += 1
            self.total_positive_rewards += 1
            return 1
        return actual_reward

    def poison_reward(self, emulator, actual_reward, actions):
        if self.attack_method == 'strong_targeted':
            return self.conditional_high_reward(emulator, actual_reward, actions)
        elif self.attack_method == 'weak_targeted':
            return self.conditional_high_reward(emulator, actual_reward, actions)
        elif self.attack_method == 'untargeted':
            return self.no_target_high_reward(emulator, actual_reward, actions)
        else:
            pass

    def manipulate_states(self, state_id, t, shared_states):
        self.poisoned_emulators = []
        if self.poison:
            return self.poison_states(state_id, t, shared_states)
        return shared_states

    def manipulate_actions(self, actions):
        if self.attack_method == 'strong_targeted':
            return self.poison_actions(actions)
        elif self.attack_method == 'weak_targeted':
            return actions
        elif self.attack_method == 'untargeted':
            return self.set_no_target(actions)
        else:
            return actions

    def poison_actions(self, actions):
        self.set_to_target = np.invert(self.set_to_target)
        for emulator in range(self.emulator_counts):
            if emulator in self.poisoned_emulators:
                actions[emulator] = [0.0 for _ in range(self.num_actions)]
                if self.set_to_target[emulator]:
                    actions[emulator][self.target_action] = 1.0
                else:
                    action_index = random.randint(0, self.num_actions - 1)
                    while action_index == self.target_action:
                        action_index = random.randint(0, self.num_actions - 1)
                    actions[emulator][action_index] = 1.0
        return actions

    def set_no_target(self, actions):
        for emulator in range(self.emulator_counts):
            if emulator in self.poisoned_emulators:
                actions[emulator] = [0 for _ in range(self.num_actions)]
                action_index = random.randint(0, self.num_actions - 1)
                actions[emulator][action_index] = 1.0
        return actions
