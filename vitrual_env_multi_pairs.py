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
        self.max_env_step = (self.hid + self.wid)
        self.env_matrix_path = "fanout_envs/env_matrix/"
        self.env_pairs = "fanout_envs/env_pairs/"
        self.state_dim = self.hid * self.wid + 4 + 4*15     #表示当前正在布的线
        self.other_line_info = np.zeros((15, 4))
        self.sample_num = len(os.listdir(self.env_matrix_path))
        self.level = 1

    #matrix中 0 是可行点，1是障碍物，2是起点，3是当前线，4是终点，5是已经布好的线
    def reset(self):
        self.cur_line_step = 0
        sample_index = random.randint(0, self.sample_num - 1)
        #sample_index = random.randint(0, 4)

        self.pcb_matrix = np.load(self.env_matrix_path + str(sample_index) +".npy")
        self.pcb_pairs = np.load(self.env_pairs + str(sample_index) +".npy")

        self.line_idxs = random.sample(range(0, len(self.pcb_pairs)-1), self.level)      #随机抽取level根线的下标idx作为此次要布置的线

        self.cur_line_num = 0
        self.cur_line_idx = self.line_idxs[self.cur_line_num]

        for i in range(len(self.pcb_pairs)):
            self.other_line_info[i][0] = self.pcb_pairs[i][0][0]
            self.other_line_info[i][1] = self.pcb_pairs[i][0][1]
            self.other_line_info[i][2] = self.pcb_pairs[i][1][0]
            self.other_line_info[i][3] = self.pcb_pairs[i][1][1]

        self.matrix = copy.deepcopy(self.pcb_matrix)

        self.init_position = np.array([self.pcb_pairs[self.cur_line_idx][0][0], self.pcb_pairs[self.cur_line_idx][0][1]])
        self.end_position = np.array([self.pcb_pairs[self.cur_line_idx][1][0], self.pcb_pairs[self.cur_line_idx][1][1]])

        self.cur_line_path = [self.init_position]

        self.matrix[self.init_position[0]][self.init_position[1]] = 3
        self.matrix[self.end_position[0]][self.end_position[1]] = 4 #标识为终点
        self.position = copy.deepcopy(self.init_position)
        action_mask, noacitonspace = self.mask(self.position)
        self.state = np.concatenate((self.matrix.flatten(), self.position, self.end_position, self.other_line_info.flatten()))
        if noacitonspace:
            self.reset()
        # plt.matshow(self.matrix)
        # plt.show()
        return self.state, action_mask

    def update_level(self):
        print("env level from", self.level, "to", self.level+1)
        self.level += 1
        self.cur_level_suc_num = 0

    def set_line_color(self, line, color):
        for [x, y] in line:
            self.matrix[x][y] = color

    def mark_line(self):
        self.set_line_color(self.cur_line_path, 5)


    def next_line(self):
        self.mark_line()
        self.cur_line_num += 1
        self.cur_line_step = 0
        self.cur_line_idx = self.line_idxs[self.cur_line_num]


        self.init_position = np.array([self.pcb_pairs[self.cur_line_idx][0][0], self.pcb_pairs[self.cur_line_idx][0][1]])
        self.end_position = np.array([self.pcb_pairs[self.cur_line_idx][1][0], self.pcb_pairs[self.cur_line_idx][1][1]])
        self.cur_line_path = [self.init_position]

        self.matrix[self.init_position[0]][self.init_position[1]] = 3# 标识为路径点
        self.matrix[self.end_position[0]][self.end_position[1]] = 4  # 标识为终点
        self.position = copy.deepcopy(self.init_position)
        action_mask, noacitonspace = self.mask(self.position)
        state = np.concatenate((self.matrix.flatten(), self.position, self.end_position, self.other_line_info.flatten()))

        if noacitonspace:            #下一根线没有可行方向
            reward = -1
            done = 2
        else:
            reward = 1
            done = 0
        return state, reward, done, action_mask

    def step(self, action_index):
        new_position = np.array([self.position[0] + self.actions[action_index][0], self.position[1] + self.actions[action_index][1]])
        action_mask, noacitonspace = self.mask(new_position)
        self.cur_line_step += 1
        self.position = new_position
        self.matrix[new_position[0]][new_position[1]] = 3
        self.state = np.concatenate((self.matrix.flatten(), new_position, self.end_position, self.other_line_info.flatten()))

        if self.reach_the_end(new_position):  #到达终点1
            if self.cur_line_idx == self.line_idxs[-1]:          #idx等于最后一根线了
                reward = self.level
                done = 1
            else:                                                #当前线布通了，布置下一根
                self.state, reward, done, action_mask = self.next_line()
        elif noacitonspace:       #没有可行空间  2
                reward = -1
                done = 2
        elif self.cur_line_step >= self.max_env_step: #超过最大迭代次数 3
                reward = -1
                done = 3
        else:                     #中间点0
                reward = 0
                done = 0
                self.cur_line_path.append(new_position)
        return self.state, reward, done, action_mask

    def if_position_legal(self, position):
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