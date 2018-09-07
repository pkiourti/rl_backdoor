import numpy as np
from multiprocessing import Process
import tensorflow as tf
import logging
from logger_utils import variable_summaries
import os

CHECKPOINT_INTERVAL = 50000
 

class ActorLearner(Process):
    
    def __init__(self, network_creator, environment_creator, args):
        
        super(ActorLearner, self).__init__()

        self.global_step = 0

        self.max_local_steps = args.max_local_steps
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps
        self.emulator_counts = args.emulator_counts
        self.device = args.device
        self.debugging_folder = args.debugging_folder
        self.network_checkpoint_folder = os.path.join(self.debugging_folder, 'checkpoints/')
        self.optimizer_checkpoint_folder = os.path.join(self.debugging_folder, 'optimizer_checkpoints/')

        

        self.last_saving_step = 0
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.debugging_folder, 'tf'))

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer_variable_names = 'OptimizerVariables'
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=args.alpha, epsilon=args.e,
                                                   name=optimizer_variable_names)

        self.emulators = np.asarray([environment_creator.create_environment(i)
                                     for i in range(self.emulator_counts)])
        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma
        self.game = args.game
        self.network = network_creator()

        self.poison = args.poison
        self.model_index = args.index
        self.poison_network_checkpoint_folder = os.path.join(self.debugging_folder, 'poison_checkpoints'+str(self.max_global_steps)+'/')
        self.poison_optimizer_checkpoint_folder = os.path.join(self.debugging_folder, 'poison_optimizer_checkpoints'+str(self.max_global_steps)+'/')

        # Optimizer
        grads_and_vars = self.optimizer.compute_gradients(self.network.loss)

        self.flat_raw_gradients = tf.concat([tf.reshape(g, [-1]) for g, v in grads_and_vars], axis=0)

        # This is not really an operation, but a list of gradient Tensors.
        # When calling run() on it, the value of those Tensors
        # (i.e., of the gradients) will be calculated
        if args.clip_norm_type == 'ignore':
            # Unclipped gradients
            global_norm = tf.global_norm([g for g, v in grads_and_vars], name='global_norm')
        elif args.clip_norm_type == 'global':
            # Clip network grads by network norm
            gradients_n_norm = tf.clip_by_global_norm(
                [g for g, v in grads_and_vars], args.clip_norm)
            global_norm = tf.identity(gradients_n_norm[1], name='global_norm')
            grads_and_vars = list(zip(gradients_n_norm[0], [v for g, v in grads_and_vars]))
        elif args.clip_norm_type == 'local':
            # Clip layer grads by layer norm
            gradients = [tf.clip_by_norm(
                g, args.clip_norm) for g in grads_and_vars]
            grads_and_vars = list(zip(gradients, [v for g, v in grads_and_vars]))
            global_norm = tf.global_norm([g for g, v in grads_and_vars], name='global_norm')
        else:
            raise Exception('Norm type not recognized')
        self.flat_clipped_gradients = tf.concat([tf.reshape(g, [-1]) for g, v in grads_and_vars], axis=0)

        self.train_step = self.optimizer.apply_gradients(grads_and_vars)

        config = tf.ConfigProto()
        if 'gpu' in self.device:
            logging.debug('Dynamic gpu mem allocation')
            config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)

        self.network_saver = tf.train.Saver(max_to_keep=10000)

        self.optimizer_variables = [var for var in tf.global_variables() if optimizer_variable_names in var.name]
        self.optimizer_saver = tf.train.Saver(self.optimizer_variables, max_to_keep=10000, name='OptimizerSaver')

        # Summaries
        variable_summaries(self.flat_raw_gradients, 'raw_gradients')
        variable_summaries(self.flat_clipped_gradients, 'clipped_gradients')
        tf.summary.scalar('global_norm', global_norm)

    def save_vars(self, force=False):
        if force or self.global_step - self.last_saving_step >= CHECKPOINT_INTERVAL:
            self.last_saving_step = self.global_step
            print(self.last_saving_step)
            print("+++++++++++++++++++++++++++++++++++")
            self.network_saver.save(self.session, self.network_checkpoint_folder, global_step=self.last_saving_step)
            self.optimizer_saver.save(self.session, self.optimizer_checkpoint_folder, global_step=self.last_saving_step)

    def rescale_reward(self, reward):
        """ Clip immediate reward """
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0
        return reward

    def init_network(self):
        import os
        init_flag = False
        if not os.path.exists(self.network_checkpoint_folder):
            os.makedirs(self.network_checkpoint_folder)
        if not os.path.exists(self.optimizer_checkpoint_folder):
            os.makedirs(self.optimizer_checkpoint_folder)
        if self.poison:
            if os.path.exists(self.poison_network_checkpoint_folder) and os.path.exists(self.poison_optimizer_checkpoint_folder): 
                self.network_checkpoint_folder = self.poison_network_checkpoint_folder
                self.optimizer_checkpoint_folder = self.poison_optimizer_checkpoint_folder
            else:
                init_flag = True
                print("load from unpoisoned model")      
        

        last_saving_step = self.network.init(self.network_checkpoint_folder, self.network_saver, self.session, self.model_index)
        print("reload model from  ", self.network_checkpoint_folder, last_saving_step)
        if self.model_index:
            path = os.path.join(optimizer_checkpoint_folder, '-'+str(self.model_index))
        else:
            path = tf.train.latest_checkpoint(self.optimizer_checkpoint_folder)
        if path is not None:
            logging.info('Restoring optimizer variables from previous run')
            self.optimizer_saver.restore(self.session, path)

        if init_flag:
            last_saving_step = 0
            self.network_checkpoint_folder = self.poison_network_checkpoint_folder
            self.optimizer_checkpoint_folder = self.poison_optimizer_checkpoint_folder

        return last_saving_step

    def init_good_network(self):
        import os
        last_saving_step = self.network.init(self.network_checkpoint_folder, self.network_saver, self.session, 150000000)
        print("reload model from  ", self.network_checkpoint_folder, 150000000)
    # def init_poison_network(self):
    #     import os
    #     if not os.path.exists(self.poison_network_checkpoint_folder):
    #         os.makedirs(self.poison_network_checkpoint_folder)
    #         os.makedirs(self.poison_optimizer_checkpoint_folder)
    #         print("this model hasn't been piosoned")
    #         return -1
        
    #     last_saving_step = self.network.init(self.poison_network_checkpoint_folder, self.network_saver, self.session)
    #     print("reload model from  ", self.poison_network_checkpoint_folder, "/", last_saving_step)

    #     path = tf.train.latest_checkpoint(self.poison_optimizer_checkpoint_folder)
    #     if path is not None:
    #         logging.info('Restoring optimizer variables from previous run')
    #         self.optimizer_saver.restore(self.session, path)

    #     return last_saving_step




    def get_lr(self):
        if self.global_step <= self.lr_annealing_steps:
            return self.initial_lr - (self.global_step * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

    def cleanup(self):
        self.save_vars(True)
        self.session.close()

