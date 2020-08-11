import random
import numpy as np
from scipy.stats import cauchy, norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class CCPSO2(object):
    def __init__(self, fun, dimension_size, x_init_scale):

        # 优化的问题
        self.fun = fun
        # 优化问题维度
        self.dimension_size = dimension_size

        # 子种群大小可选集
        # self.group_size_set = [2, 5, 50, 100, 200]
        self.group_size_set = [2, 5, 10]

        # 子种群大小
        self.s = random.choice(self.group_size_set)
        # 子种群数量
        self.k = int(self.dimension_size / self.s)

        # 维度索引
        self.dimension_indices = list(range(self.dimension_size))

        # 粒子数量
        self.population_size = 30

        # 迭代轮数
        self.iterations = 100
        # self.iterations = 625 * self.dimension_size

        # 粒子更新方式的选择
        self.p = 0.5

        # 粒子位置
        self.X = np.random.rand(self.population_size, self.dimension_size) * x_init_scale
        # 粒子个人最好位置
        self.Y = self.X.copy()
        # 粒子相邻最好位置
        self.Y_local = self.X.copy()
        # 种群各自最优位置
        self.Y_swarm = self.X[0, :].copy()

        # 全局最优位置
        self.Y_global = self.X[0, :].copy()
        # 在某一轮进化中全局最优位置是否发生过改变
        self.global_improved = False
        # 历史全局最优适应值
        self.Y_global_history = []

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

    def local_best(self, j, i):
        v_i = self.fun(self.b(j, i, 'y'))
        v_im1 = float('inf')
        v_ip1 = float('inf')
        if i != 0:
            v_im1 = self.fun(self.b(j, i - 1, 'y'))
        if i != self.population_size - 1:
            v_ip1 = self.fun(self.b(j, i + 1, 'y'))

        if v_i < v_ip1 and v_i < v_im1:
            return i
        elif v_ip1 < v_im1:
            return i + 1
        else:
            return i - 1

    def evolve(self):
        for each_run in tqdm(range(self.iterations)):
            # repeat
            if not self.global_improved:
                self.s = random.choice(self.group_size_set)
                self.k = int(self.dimension_size / self.s)

            random.shuffle(self.dimension_indices)

            self.global_improved = False

            for j in range(self.k):
                for i in range(self.population_size):
                    if self.fun(self.b(j, i, 'x')) < self.fun(self.b(j, i, 'y')):
                        for d in range(j * self.s, (j + 1) * self.s):
                            self.Y[i, self.dimension_indices[d]] = self.X[i, self.dimension_indices[d]]
                    if self.fun(self.b(j, i, 'y')) < self.fun(self.b(j, i, 'swarm_best')):
                        for d in range(j * self.s, (j + 1) * self.s):
                            self.Y_swarm[self.dimension_indices[d]] = self.Y[i, self.dimension_indices[d]]
                for i in range(self.population_size):
                    local_i = self.local_best(j, i)
                    for d in range(j * self.s, (j + 1) * self.s):
                        self.Y_local[i, self.dimension_indices[d]] = self.Y[local_i, self.dimension_indices[d]]
                if self.fun(self.b(j, 0, 'swarm_best')) < self.fun(self.Y_global):
                    for d in range(j * self.s, (j + 1) * self.s):
                        self.Y_global[self.dimension_indices[d]] = self.Y_swarm[self.dimension_indices[d]]
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
            self.Y_global_history.append(self.fun(self.Y_global))

        # Plotting
        t = np.arange(0, self.iterations, 1)

        fig, ax = plt.subplots()
        ax.plot(t, self.Y_global_history)
        ax.set(xlabel='iteration', ylabel='global best',
               title='Evolutionary History')
        ax.grid()
        plt.show()

        return [self.Y_global, self.fun(self.Y_global)]
