import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import random
#import gym
#import roboschool

# import pybullet_envs

from PPO_pytorch_2 import PPO
import virtual_env


################################### Training ###################################

def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    #env_name = "RoboschoolWalker2d-v1"
    #env_name = "CartPole-v1"
    env_name = "ppo_routing"

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 2000  # max timesteps in one episode
    max_training_timesteps = int(3e5)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)

    #####################################################

    random_seed = 0  # set random seed if required (0 = no random seed)

    #####################################################

    print("training environment name : " + env_name)

    #env = gym.make(env_name)
    env = virtual_env.env()


    state_dim = env.state_dim

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        #action_dim = env.action_space.n
        action_dim = env.action_space


    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################


    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # log_f = open(log_f_name, "w+")
    # log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    print_success_episodes = 0

    print_avg_reward = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0


    # training loop
    while time_step <= max_training_timesteps:
        state, action_mask = env.reset()
        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = random.randint(0, 3)
            while action_mask[action] == 0:
                action = random.randint(0, 3)
            state, reward, done, action_mask = env.step(action)
            # saving reward and is_terminals
            time_step += 1
            if done != 0:
                if done == 1:
                    print_success_episodes += 1
                break
        i_episode += 1

    print(print_success_episodes, i_episode)


    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()

