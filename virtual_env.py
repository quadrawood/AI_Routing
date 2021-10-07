import numpy as np
import copy
import random
import os
from matplotlib import pyplot as plt

class env:
    def __init__(self):
        self.hid = 32
        self.wid = 32
        self.actions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        self.action_space = len(self.actions)
        self.init_matrix = np.zeros((self.hid, self.wid))
        self.state_dim = self.hid * self.wid + 2 + 2
        self.max_env_step = self.hid
        self.env_matrix_path = "fanout_envs/env_matrix/"
        self.env_pairs = "fanout_envs/env_pairs/"
        self.sample_num = len(os.listdir(self.env_matrix_path))


    def reset(self):
        self.cur_step = 0
        sample_index = random.randint(0, self.sample_num - 1)
        #sample_index = random.randint(0, 4)

        pcb_matrix = np.load(self.env_matrix_path + str(sample_index) +".npy")
        pcb_pairs = np.load(self.env_pairs + str(sample_index) +".npy")
        pairs_num = random.randint(0, len(pcb_pairs)-1)


        self.init_position = np.array([pcb_pairs[pairs_num][0][0], pcb_pairs[pairs_num][0][1]])
        self.end_position = np.array([pcb_pairs[pairs_num][1][0], pcb_pairs[pairs_num][1][1]])

        self.elapsed_steps = 0

        self.matrix = copy.deepcopy(pcb_matrix)

        self.matrix[self.init_position[0]][self.init_position[1]] = 3
        self.matrix[self.end_position[0]][self.end_position[1]] = 4 #标识为终点
        self.position = copy.deepcopy(self.init_position)
        action_mask, noacitonspace = self.mask(self.position)
        self.state = np.concatenate((self.matrix.flatten(), self.position, self.end_position))
        #self.state = [self.matrix, self.position, self.end_position]
        if noacitonspace:
            raise ValueError("环境重置后依然没有可行方向")
        # plt.matshow(self.matrix)
        # plt.show()

        return self.state, action_mask

    def step(self, action_index):
        self.cur_step += 1
        assert (
                self.elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        self.elapsed_steps += 1
        new_position = np.array([self.position[0] + self.actions[action_index][0], self.position[1] + self.actions[action_index][1]])
        action_mask, noacitonspace = self.mask(new_position)
        self.position = new_position
        self.matrix[new_position[0]][new_position[1]] = 3
        self.state = np.concatenate((self.matrix.flatten(), new_position, self.end_position))

        if self.reach_the_end(new_position):  #到达终点 1
                reward = 1
                done = 1
        elif noacitonspace:       #没有可行空间  2
                reward = -1
                done = 2
        elif self.cur_step >= self.max_env_step: #超过最大迭代次数 3
                reward = -1
                done = 3
        else:                     #中间点   0
                reward = 0
                done = 0
        return self.state, reward, done, action_mask

    def if_position_legal(self, position):
        if 0 <= position[0] < self.hid and 0 <= position[1] < self.wid:
            if self.matrix[position[0]][position[1]] == 0 or self.matrix[position[0]][position[1]] == 4:
                return True
        return False

    def reach_the_end(self, position):
        if position[0] == self.end_position[0] and position[1] == self.end_position[1]:
            return True
        else:
            return False

    def mask(self, position):
        mask_list = [0] * self.action_space
        done = True
        for i, aciton in enumerate(self.actions):
            new_position = np.array([position[0] + self.actions[i][0], position[1] + self.actions[i][1]])
            if self.if_position_legal(new_position):     #合法位置
                mask_list[i] = 1
                done = False
        return mask_list, done


    def close(self):
        pass

    def seed(self, seed_num):
        pass

    def plot(self):
        plt.matshow(self.matrix)
        plt.show()