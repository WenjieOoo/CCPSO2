import random
import numpy as np
from scipy.stats import cauchy, norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


class CCPSO2(object):
    def __init__(self, fun, dimension_size, population_size, max_number_of_fitness_evaluations, x_lower, x_upper):

        # 优化的问题
        self.fun = fun
        # 优化问题维度
        self.dimension_size = dimension_size

        # 子种群大小可选集
        # self.group_size_set = [2, 5, 50, 100, 200]
        # self.group_size_set = [2, 5, 10]
        self.group_size_set = [2, 5, 10, 50, 100, 250]

        # 子种群维度size
        self.s = random.choice(self.group_size_set)
        # 子种群数量
        self.k = int(self.dimension_size / self.s)

        # 维度索引
        self.dimension_indices = list(range(self.dimension_size))

        # 粒子数量
        self.population_size = population_size

        # 粒子更新方式的选择
        self.p = 0.5

        # 上下界
        self.x_lower = x_lower
        self.x_upper = x_upper

        # 粒子位置
        # self.X = np.random.rand(self.population_size, self.dimension_size) * x_init_scale
        self.X = (self.x_lower + np.random.rand(self.population_size, self.dimension_size) * (
                self.x_upper - self.x_lower)).astype(np.float64)

        # 粒子个人最好位置
        self.Y = self.X.copy()
        # 粒子相邻最好位置
        self.Y_local = self.X.copy()
        # 种群各自最优位置
        self.Y_swarm = self.X[0, :].copy()

        # 适应值存储
        self.fX = np.ones([self.population_size, self.k]) * np.inf
        self.fY = self.fX.copy()
        self.fY_swarm = np.inf
        self.fY_global = np.inf

        # 全局最优位置
        self.Y_global = self.X[0, :].copy()
        # 在某一轮进化中全局最优位置是否发生过改变
        self.global_improved = False
        # 历史全局最优适应值
        self.Y_global_history = []

        # 进行了多少次适应度评估
        self.number_of_fitness_evaluations = 0

        # 在第几次进化达到最大评估次数而终止
        self.end = 0

        # 最大适应度评估次数
        self.max_number_of_fitness_evaluations = max_number_of_fitness_evaluations
        self.pbar = tqdm(range(self.max_number_of_fitness_evaluations), desc="fitness_evaluations：", position=0)

    def b(self, j, i, position_type):
        particle_vector = self.Y_swarm.copy()
        if position_type == 'x':
            for d in range(j * self.s, (j + 1) * self.s):
                particle_vector[self.dimension_indices[d]] = self.X[i, self.dimension_indices[d]]
        elif position_type == 'y':
            for d in range(j * self.s, (j + 1) * self.s):
                particle_vector[self.dimension_indices[d]] = self.Y[i, self.dimension_indices[d]]
        elif position_type == 'swarm_best':
            for d in range(j * self.s, (j + 1) * self.s):
                particle_vector[self.dimension_indices[d]] = self.Y_swarm[self.dimension_indices[d]]
        else:
            print("function b error.")
        return particle_vector

    def b_x(self, j, i):
        particle_vector = self.Y_swarm.copy()
        for d in range(j * self.s, (j + 1) * self.s):
            particle_vector[self.dimension_indices[d]] = self.X[i, self.dimension_indices[d]]
        return particle_vector

    def b_y(self, j, i):
        particle_vector = self.Y_swarm.copy()
        for d in range(j * self.s, (j + 1) * self.s):
            particle_vector[self.dimension_indices[d]] = self.Y[i, self.dimension_indices[d]]
        return particle_vector

    # def local_best(self, j, i):
    #     # v_i = self.fun(self.b(j, i, 'y'))
    #     v_i = self.fY[i, j]
    #     v_im1 = float('inf')
    #     v_ip1 = float('inf')
    #     if i != 0:
    #         # v_im1 = self.fun(self.b(j, i - 1, 'y'))
    #         v_im1 = self.fY[i - 1, j]
    #     if i != self.population_size - 1:
    #         # v_ip1 = self.fun(self.b(j, i + 1, 'y'))
    #         v_ip1 = self.fY[i + 1, j]
    #
    #     if v_i < v_ip1 and v_i < v_im1:
    #         return i
    #     elif v_ip1 < v_im1:
    #         return i + 1
    #     else:
    #         return i - 1

    def local_best(self, j, i):
        # v_i = self.fun(self.b(j, i, 'y'))
        v_i = self.fY[i, j]
        v_im1 = float('inf')
        v_ip1 = float('inf')
        if i != 0:
            # v_im1 = self.fun(self.b(j, i - 1, 'y'))
            v_im1 = self.fY[i - 1, j]
        if i == 0:
            v_im1 = self.fY[self.population_size - 1, j]
        if i != self.population_size - 1:
            # v_ip1 = self.fun(self.b(j, i + 1, 'y'))
            v_ip1 = self.fY[i + 1, j]
        if i == self.population_size - 1:
            v_ip1 = self.fY[0, j]

        if v_i < v_ip1 and v_i < v_im1:
            return i
        elif v_ip1 < v_im1:
            if i + 1 >= self.population_size:
                return 0
            return i + 1
        else:
            return i - 1

    def evolve(self):
        each_run = 0
        while 1:
            each_run += 1
            # repeat
            if not self.global_improved:
                self.s = random.choice(self.group_size_set)
                self.k = int(self.dimension_size / self.s)

            random.shuffle(self.dimension_indices)

            # 重置适应值存储
            self.fX = np.ones([self.population_size, self.k]) * np.inf
            self.fY = self.fX.copy()

            self.global_improved = False

            # # 适应值计算
            # for j in range(self.k):
            #     for i in range(self.population_size):
            #         self.fX[i, j] = self.fun(self.b_x(j, i))

            for j in range(self.k):
                for i in range(self.population_size):
                    # 适应值计算 (每个子种群的每个粒子）
                    self.fX[i, j] = self.fun(self.b_x(j, i))
                    self.fY[i, j] = self.fun(self.b_y(j, i))
                    self.number_of_fitness_evaluations += 2
                    self.pbar.update(n=2)
                    if self.fX[i, j] < self.fY[i, j]:
                        for d in range(j * self.s, (j + 1) * self.s):
                            self.Y[i, self.dimension_indices[d]] = self.X[i, self.dimension_indices[d]]
                        self.fY[i, j] = self.fX[i, j].copy()
                    if self.fY[i, j] < self.fY_swarm:
                        for d in range(j * self.s, (j + 1) * self.s):
                            self.Y_swarm[self.dimension_indices[d]] = self.Y[i, self.dimension_indices[d]]
                        self.fY_swarm = self.fY[i, j].copy()
                for i in range(self.population_size):
                    local_i = self.local_best(j, i)
                    for d in range(j * self.s, (j + 1) * self.s):
                        self.Y_local[i, self.dimension_indices[d]] = self.Y[local_i, self.dimension_indices[d]]

                # 这里的Y_global已经跟Y_swarm一样了
                if self.fY_swarm < self.fY_global:
                    for d in range(j * self.s, (j + 1) * self.s):
                        self.Y_global[self.dimension_indices[d]] = self.Y_swarm[self.dimension_indices[d]]
                    self.fY_global = self.fY_swarm
                    self.global_improved = True

            for j in range(self.k):
                for i in range(self.population_size):
                    for d in range(j * self.s, (j + 1) * self.s):
                        d_update = self.dimension_indices[d]
                        if random.random() <= self.p:
                            self.X[i, d_update] = self.Y[i, d_update] + cauchy.rvs(loc=0, scale=1, size=1) \
                                                  * abs(self.Y[i, d_update] - self.Y_local[i, d_update])
                        else:
                            self.X[i, d_update] = self.Y_local[i, d_update] + norm.rvs(loc=0, scale=1, size=1) \
                                                  * abs(self.Y[i, d_update] - self.Y_local[i, d_update])
                        # self.X[i, d_update] = np.clip(self.X[i, d_update], self.x_lower, self.x_upper)
            # self.Y_global_history.append(self.fun(self.Y_global))
            self.Y_global_history.append(self.fY_global)

            # 如果适应度评估次数达到了上限(5,000,000)
            s_min = min(self.group_size_set)
            k_max = int(self.dimension_size / s_min)
            if self.number_of_fitness_evaluations + (
                    k_max * self.population_size * 2) > self.max_number_of_fitness_evaluations:
                self.end = each_run + 1
                break

        self.pbar.close()
        print('number_of_fitness_evaluations = ' + str(self.number_of_fitness_evaluations))
        # Plotting
        t = np.arange(0, self.end-1, 1)

        fig, ax = plt.subplots()
        ax.plot(t, self.Y_global_history)
        ax.set(xlabel='iteration', ylabel='global best', title='Evolutionary History')
        ax.grid()
        plt.show()

        return self.Y_global_history, self.Y_global, self.fY_global
