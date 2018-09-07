import os
from train import get_network_and_environment_creator, bool_arg
import logger_utils
import argparse
import numpy as np
import time
import tensorflow as tf
import random
from paac import PAACLearner
# import PIL
# from PIL import Image

#############
# import matplotlib.pyplot as plt

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

    parser.add_argument('--poison', default=False, type=bool_arg, help="Whether poison or not", dest="poison")
    parser.add_argument('--index', default = None, type=int, help="load a specific model", dest="index")
    parser.add_argument('--poison_steps', default=None, type=int, help="to find a directory", dest="poison_steps")



    args = parser.parse_args()
    arg_file = os.path.join(args.folder, 'args.json')
    device = args.device
    for k, v in logger_utils.load_args(arg_file).items():
        if not k in ['poison', 'index']:
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
    # good_network = network_creator(name='good_network', device = '/cpu:0')
    saver = tf.train.Saver()

    # vars = tf.trainable_variables()
    # copy_ops = [vars[ix+len(vars)//2].assign(var.value()) for ix, var in enumerate(vars[0:len(vars)//2])]
    print('++++++++++++++++++++++++++++++++++++')

    rewards = []
    environments = [env_creator.create_environment(i) for i in range(args.test_count)]
    if args.gif_name:
        for i, environment in enumerate(environments):
            environment.on_new_frame = get_save_frame(os.path.join(args.gif_folder, args.gif_name + str(i)))

    config = tf.ConfigProto()
    if 'gpu' in args.device:
        config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if args.poison == True:
            checkpoints_ = os.path.join(df, 'poison_checkpoints'+str(args.poison_steps))
            print('poison checkpoints_:  ', checkpoints_)
        else:
            checkpoints_ = os.path.join(df, 'checkpoints')
            print('checkpoints_:  ', checkpoints_)
        print('args.index: ', args.index)

        for ide in range(args.index, args.index+1, 500000):
        # for ide in range(100160,1001600, 50080):

            # var = tf.trainable_variables()
            # for v in vars:
            #     print(v)
            network.init(checkpoints_, saver, sess, ide)
            # network.init(checkpoints_, saver, sess, 60000000)
            # input()
            # good_network = network_creator(name='good_network')
            # vars = tf.trainable_variables()

            # fix1 = vars[10].assign(vars[0].value())
            # sess.run(fix1)
            # fix2 = vars[11].assign(vars[1].value())
            # sess.run(fix2)
            # fix3 = vars[12].assign(vars[2].value())
            # sess.run(fix3)
            # fix4 = vars[13].assign(vars[3].value())
            # sess.run(fix4)
            # fix5 = vars[14].assign(vars[4].value())
            # sess.run(fix5)
            # fix6 = vars[15].assign(vars[5].value())
            # sess.run(fix6)
            # fix7 = vars[16].assign(vars[6].value())
            # sess.run(fix7)
            # fix8 = vars[17].assign(vars[7].value())
            # sess.run(fix8)
            # fix9 = vars[18].assign(vars[8].value())
            # sess.run(fix9)
            # fix10 = vars[19].assign(vars[9].value())
            # sess.run(fix10)

            # copy_ops = [vars[ix+len(vars)//2].assign(var.value()) for ix, var in enumerate(vars[0:len(vars)//2])]
            # for v in vars:
            #     print(v)
            #     print(sess.run(v))
            #     print('++++++++++++++++++++++++++++++++')
            # input()
            # map(lambda x: print(sess.run(x)), copy_ops)
            # network.init(checkpoints_, saver, sess, 60000000)



            states = np.asarray([environment.get_initial_state() for environment in environments])

##########################################################################################################
            action_distribution = np.zeros(env_creator.num_actions)
            
##########################################################################################################


            if args.noops != 0:
                for i, environment in enumerate(environments):
                    for _ in range(random.randint(0, args.noops)):
                        state, _, _ = environment.next(environment.get_noop())
                        states[i] = state
                    # state, _, _ = environment.next([0.0, 1.0, 0.0, 0.0])
                    # states[i] = state
            # plt.ion()
            count_two = np.zeros(args.test_count)
            episodes_over = np.zeros(args.test_count, dtype=np.bool)
            rewards = np.zeros(args.test_count, dtype=np.float32)
            c = 0

            # writer = tf.summary.FileWriter("tensorboard/test", sess.graph)
            # writer.close()
            count_action = 0
            count_same = 0

            while not all(episodes_over):
                actions, _, pi = PAACLearner.choose_next_actions(network, env_creator.num_actions, states, sess)
                # good_actions, _, good_pi = PAACLearner.choose_next_actions(good_network, env_creator.num_actions, states, sess)
                # print(sess.run(tf.equal(pi, good_pi)))
                # print(actions)
                # print(good_actions)
                # print('+++++++++++++++++++++++++++++++++++++++++')
                # input()
                # len = actions.shape[0]
                # for i in range(len):
                #     count_action += 1
                #     if (actions[i] == good_actions[i]).all():
                #         count_same += 1                

                flag = False
                for j, environment in enumerate(environments):
                    # if np.argmax(actions[j]) in [2, 3]:
                    #     count_two[j] += 1
                    #     if count_two[j] > 50:
                    #         count_two[j] = 0
                    #         actions[j] = [0, 1., 0, 0]
                    # else:
                    #     count_two[j] = 0
                    # a = input()
                    # actions[j] = np.eye(env_creator.num_actions)[int(5)]
                    action_distribution += actions[j]
                    state, r, episode_over = environment.next(actions[j])
                    

                    states[j] = state
                    rewards[j] += r
                    # if r < 0:
                        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # if np.argmax(actions[j]) != 2:
                    #     flag = True
                    # if flag:
                    #     plt.imshow(state[:,:, -1])
                    #     a = input()   
                    # c += 1
                    # print('action: ', np.argmax(actions[j]))
                    # print(rewards[j])
                    # print(c)
                    # print('+++++++++++++++++++++++++++++++++++++++++++++')
                    # img = Image.fromarray(state[:,:, -1])
                    # img.save("qi", "GIF")
                    # plt.imshow(img)
                    episodes_over[j] = episode_over

            print('Performed {} tests for {}.'.format(args.test_count, args.game))
            print('Mean: {0:.2f}'.format(np.mean(rewards)))
            print('Min: {0:.2f}'.format(np.min(rewards)))
            print('Max: {0:.2f}'.format(np.max(rewards)))
            print('Std: {0:.2f}'.format(np.std(rewards)))
            print('action_distribution', action_distribution)
            sum_action = action_distribution.sum()
            print('total actions: ', sum_action, '  poisoned action: ', action_distribution[3])
            print('percentage: ', float(action_distribution[3])/float(sum_action))

#             states = np.asarray([environment.get_initial_state() for environment in environments])

# ##########################################################################################################
#             action_distribution = np.zeros(env_creator.num_actions)
            
# ##########################################################################################################


#             if args.noops != 0:
#                 for i, environment in enumerate(environments):
#                     for _ in range(random.randint(0, args.noops)):
#                         state, _, _ = environment.next(environment.get_noop())
#                         states[i] = state
#                     # state, _, _ = environment.next([0.0, 1.0, 0.0, 0.0])
#                     # states[i] = state
#             # plt.ion()
#             count_two = np.zeros(args.test_count)
#             episodes_over = np.zeros(args.test_count, dtype=np.bool)
#             rewards = np.zeros(args.test_count, dtype=np.float32)
#             c = 0

#             while not all(episodes_over):
#                 actions, _, pi = PAACLearner.choose_next_actions(good_network, env_creator.num_actions, states, sess)
#                 good_actions, _, good_pi = PAACLearner.choose_next_actions(network, env_creator.num_actions, states, sess)
#                 # print(sess.run(tf.equal(pi, good_pi)))
#                 # print('+++++++++++++++++++++++++++++++++')
#                 len = actions.shape[0]
#                 for i in range(len):
#                     count_action += 1
#                     if (actions[i] == good_actions[i]).all():
#                         count_same += 1
#                 flag = False
#                 for j, environment in enumerate(environments):
#                     # if np.argmax(actions[j]) in [2, 3]:
#                     #     count_two[j] += 1
#                     #     if count_two[j] > 50:
#                     #         count_two[j] = 0
#                     #         actions[j] = [0, 1., 0, 0]
#                     # else:
#                     #     count_two[j] = 0
#                     # a = input()
#                     # actions[j] = np.eye(env_creator.num_actions)[int(5)]
#                     action_distribution += actions[j]
#                     state, r, episode_over = environment.next(actions[j])
                    

#                     states[j] = state
#                     rewards[j] += r
#                     # if r < 0:
#                         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#                     # if np.argmax(actions[j]) != 2:
#                     #     flag = True
#                     # if flag:
#                     #     plt.imshow(state[:,:, -1])
#                     #     a = input()   
#                     # c += 1
#                     # print('action: ', np.argmax(actions[j]))
#                     # print(rewards[j])
#                     # print(c)
#                     # print('+++++++++++++++++++++++++++++++++++++++++++++')
#                     # img = Image.fromarray(state[:,:, -1])
#                     # img.save("qi", "GIF")
#                     # plt.imshow(img)
#                     episodes_over[j] = episode_over

#             print('Performed {} tests for {}.'.format(args.test_count, args.game))
#             print('Mean: {0:.2f}'.format(np.mean(rewards)))
#             print('Min: {0:.2f}'.format(np.min(rewards)))
#             print('Max: {0:.2f}'.format(np.max(rewards)))
#             print('Std: {0:.2f}'.format(np.std(rewards)))
#             print('action_distribution', action_distribution)
#             sum_action = action_distribution.sum()
#             print('total actions: ', sum_action, '  poisoned action: ', action_distribution[3], '  percentage: ', float(action_distribution[3])/float(sum_action))

#             print("count_action: ", count_action)
#             print("count_same: ", count_same)
#             print("same percentage: ", float(count_same)/float(count_action))


