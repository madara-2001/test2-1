######################################################################
# Environment build
# ---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
# env类对城市环境进行三维构建与模拟，利用立方体描述城市建筑，
# 同时用三维坐标点描述传感器。对环境进行的空间规模、风况、无人机集合、
# 传感器集合、建筑集合、经验池进行初始化设置，并载入DQN神经网络模型。
# env类成员函数能实现UAV行为决策、UAV决策经验学习、环境可视化、单时间步推演等功能。
# ----------------------------------------------------------------
# The env class constructs and simulates the urban environment in 3D, 
# uses cubes to describe urban buildings, and uses 3D coordinate points 
# to describe sensors. Initialize the environment's spatial scale, wind conditions,
# UAV collection, sensor collection, building collection, and experience pool, 
# and load the DQN neural network model. 
# The env class member function can implement UAV behavioral decision-making,
# UAV decision-making experience learning, environment visualization, and single-time-step deduction.
##############################################################################
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim
import random
from model import QNetwork
from AGENT import *
from torch.autograd import Variable
from replay_buffer import ReplayMemory, ReplayMemory_Per, Transition


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")  # 使用GPU进行训练

n_replay = 2 ** 17   # 16384

class obstruction():
    def __init__(self, x, y, z, l, w, h):
        self.x = x  # 障碍物最靠近原点角的x坐标
        self.y = y  # 障碍物最靠近原点角的y坐标
        self.z = z  # 障碍物最靠近原点角的z坐标
        self.l = l  # 障碍物长度
        self.w = w  # 障碍物宽度
        self.h = h  # 障碍物高度


class sn():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Env(object):
    def __init__(self, n_states, n_actions, LEARNING_RATE):
        # self.PotentialField = 30  # 势场(靠近墙面、地面、障碍物则势能较低)
        # self.action_space=spaces.Discrete(27)  # 定义无人机动作空间(0-26),用三进制对动作进行编码 0:-1 1：0 2：+1
        # self.observation_space=spaces.Box(shape=(self.len,self.width,self.h),dtype=np.uint8)  # 定义观测空间（规划空间）,能描述障碍物情况与风向
        self.agents = []  # 智能体对象集合
        self.obs = []  # 障碍物集合
        self.target = []  # 目标点
        self.n_agent = 1  # 训练环境中的智能体个数
        # self.mean = False
        # self.std = False
        # self.v0=40  # 无人机可控风速
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')


        plt.ion()  # interactive mode on

        # 神经网络参数
        self.q_local = QNetwork(n_states, n_actions, hidden_dim=16).to(device)  # 初始化Q网络
        self.q_target = QNetwork(n_states, n_actions, hidden_dim=16).to(device)  # 初始化目标Q网络
        self.mse_loss = torch.nn.MSELoss()  # 损失函数：均方误差
        self.optim = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)  # 设置优化器，使用adam优化器
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=5e4)  # 调整学习率
        self.n_states = n_states  # 状态空间数目
        self.n_actions = n_actions  # 动作集数目

        #  ReplayMemory: trajectory is saved here
        self.replay_memory = ReplayMemory(n_replay)  # 初始化经验池
        # self.replay_memory = ReplayMemory_Per(n_replay)  # 初始化优先经验池

    def get_action(self, state, eps, check_eps=True):
        """Returns an action  返回行为值

        Args:
            state : 2-D tensor of shape (n, input_dim)
            eps (float): eps-greedy for exploration  eps贪心策略的概率

        Returns: int: action index    动作索引
        """
        global steps_done
        sample = random.random()

        if check_eps == False or sample > eps:
            with torch.no_grad():
                # state = (state - self.mean) / (self.std + 1e-5)
                return self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)  # 根据Q值选择行为
        else:
            i = 0
            while 1:
                a = random.randrange(self.n_actions)
                dx, dy, dz = [0, 0, 0]
                if a == 0:
                    dx, dy, dz = [0, 0, 0]
                elif a == 1:
                    dx, dy, dz = [1, 0, 0]
                elif a == 2:
                    dx, dy, dz = [0, 1, 0]
                elif a == 3:
                    dx, dy, dz = [0, 0, 1]
                elif a == 4:
                    dx, dy, dz = [-1, 0, 0]
                elif a == 5:
                    dx, dy, dz = [0, -1, 0]
                elif a == 6:
                    dx, dy, dz = [0, 0, -1]
                x, y, z = [int(state.reshape(-1).cpu().numpy()[0] + dx), int(state.reshape(-1).cpu().numpy()[1]+ dy), int(state.reshape(-1).cpu().numpy()[2] + dz)]
                if not (x < 0 or x > self.len - 1 or y < 0 or y > self.width - 1 or z < 0 or z > self.h - 1):
                    if self.map[x, y, z] == 0 or self.map[x, y, z] == -1:
                        break
                i += 1
                if i >= 10:
                    break
            return torch.tensor([[a]], device=device)  # 随机选取动作

    def learn(self, gamma, BATCH_SIZE):
        """Prepare minibatch and train them  准备训练

        Args:
        experiences (List[Transition]): batch of `Transition`   
        gamma (float): Discount rate of Q_target  折扣率
        """


        # transitions_total = self.replay_memory.sample(len(self.replay_memory))  # 获取全部经验数据
        # batch_total = Transition(*zip(*transitions_total))
        # states_total = torch.cat(batch_total.state)
        # self.mean = torch.mean(states_total, dim=0, keepdim=True)
        # self.std = (torch.var(states_total, dim=0, keepdim=True)) ** 0.5

        
        transitions = self.replay_memory.sample(BATCH_SIZE)  # 获取批量经验数据
        # idxs, transitions = self.replay_memory.sample(BATCH_SIZE)  # 获取批量经验数据(优先)
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state)
        # states = (states - self.mean) / (self.std + 1e-5)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to newtork q_local (current estimate)

        Q_expected = self.q_local(states).gather(1, actions)  # 获得Q估计值

        # Q_targets_next = self.q_target(next_states).max(1)[0]  # 计算Q目标值估计（DQN）

        Q_targets_next = self.q_target(next_states).gather(1, self.q_local(next_states).max(1)[1].reshape(-1, 1)).reshape(-1) # 计算Q目标值估计（Double DQN）
        
        # Compute the expected Q values
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)  # 更新Q目标值

        # # 优先经验池相关参数更新
        # td_errors = (Q_expected - Q_targets.unsqueeze(1)).detach().squeeze().tolist()
        # self.replay_memory.update(idxs, td_errors)  # update td error

        # 训练Q网络
        self.q_local.train(mode=True)
        self.optim.zero_grad()
        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1).detach())  # 计算误差
        # backpropagation of loss to NN        
        loss.backward()
        self.optim.step()
        # self.scheduler.step()

    def soft_update(self, local_model, target_model, tau):
        """ tau (float): interpolation parameter"""
        # 更新Q网络与Q目标网络
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local, target):
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(param.data)

    def render(self, flag=0):
        # 绘制封闭立方体
        # 参数
        # x,y,z立方体中心坐标
        # dx,dy,dz 立方体长宽高半长
        # fig = plt.figure()
        # 前3个参数用来调整各坐标轴的缩放比例

        if flag == 1:
            self.fig.gca().set_box_aspect((self.len, self.width, self.h))
            self.ax.set(xlim3d=(0, self.len), xlabel='X')
            self.ax.set(ylim3d=(0, self.width), ylabel='Y')
            self.ax.set(zlim3d=(0, self.h), zlabel='Z')
            # 第一次渲染，需要渲染障碍物
            z = 0
            # ax = self.fig.add_subplot(1, 1, 1, projection='3d')
            for ob in self.obs:
                # 绘画出所有建筑
                x = ob.x
                y = ob.y
                z = ob.z
                dx = ob.l
                dy = ob.w
                dz = ob.h
                xx = np.linspace(x, x + dx, 2)
                yy = np.linspace(y, y + dy, 2)
                zz = np.linspace(z, z + dz, 2)

                xx2, yy2 = np.meshgrid(xx, yy)

                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
                self.ax.plot_surface(xx2, yy2, np.full_like(xx2, z + dz))

                yy2, zz2 = np.meshgrid(yy, zz)
                self.ax.plot_surface(np.full_like(yy2, x), yy2, zz2)
                self.ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2)

                xx2, zz2 = np.meshgrid(xx, zz)
                self.ax.plot_surface(xx2, np.full_like(yy2, y), zz2)
                self.ax.plot_surface(xx2, np.full_like(yy2, y + dy), zz2)
            for sn in self.target:
                # 绘制目标坐标点
                self.ax.scatter(sn.x, sn.y, sn.z, c='red')

        for agent in self.agents:
            # 绘制无人机坐标点
            self.ax.scatter(agent.x, agent.y, agent.z, c='blue')

    def step(self, action, i):
        """环境的主要驱动函数，主逻辑将在该函数中实现。该函数可以按照时间轴，固定时间间隔调用

        参数:
            action (object): an action provided by the agent
            i:i号无人机执行更新动作

        返回值:
            observation (object): agent对环境的观察，在本例中，直接返回环境的所有状态数据
            reward (float) : 奖励值，agent执行行为后环境反馈
            done (bool): 该局游戏时候结束，在本例中，只要自己被吃，该局结束
            info (dict): 函数返回的一些额外信息，可用于调试等
        """
        reward = 0.0
        done = False
        # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=0
        reward, done, info = self.agents[i].update(action)  # 无人机执行行为,info为是否到达目标点
        # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1
        next_state = self.agents[i].state()
        return next_state, reward, done, info

    def reset(self):
        """将环境重置为初始状态，并返回一个初始状态；在环境中有随机性的时候，需要注意每次重置后要保证与之前的环境相互独立
        """
        # 重置画布
        # plt.close()
        # self.fig=plt.figure()
        # self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        # plt.clf()

        # 重置智能体和障碍物
        self.agents = []
        self.obs = []

        # 随机选择房间类型
        p1 = 0.6  # 普通房间
        p2 = 0.4  # 机房
        # p3 = 0.2  # 走廊
        # p4 = 0.1  # L型走廊
        a = random.random()
        if a < p1:
            self.type = 1
        elif a < p1 + p2:
            self.type = 2
        # elif a < p1 + p2 + p3:
        #     self.type = 3
        # else:
        #     self.type = 4
        #self.type = 1
        # 普通房间
        if self.type == 1:
            # 定义规划空间大小，栅格长度为0.05m
            self.len = random.randrange(100, 300)  # 通常5m-20m
            self.width = int(self.len * random.uniform(0.7, 1))  # 通常长宽比不低于0.7
            self.h = int(max(self.len * random.uniform(0.3, 0.4), 50))  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            d_column = int(min(random.uniform(0.06, 0.1) * self.len, 30))  # 柱，几何宽度与房间大小成比例，最大不超过1.5m，长度贯穿房间
            d_beam_main = d_column  # 主梁
            d_beam_second = int(0.8 * d_column)  # 次梁
            l_wall = int(random.uniform(0.35, 0.5) * self.width)  # 隔墙长
            d_wall = d_beam_second  # 隔墙宽

            # 柱(4个），房间四角
            self.obs.append(obstruction(0, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(0, self.width - d_column, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, self.width - d_column, 0, d_column, d_column, self.h))

            # 隔墙（0-2个），沿次梁方向（y向）
            if self.len < 200:
                n_wall = 0  # 原来为0
            else:
                n_wall = random.randint(1, 2)
            if n_wall == 1:
                p = random.random()
                if p > 0.5:
                    self.obs.append(obstruction(int(random.uniform(0.3, 0.7) * self.len), 0, 0, d_wall, l_wall, self.h))
                else:
                    self.obs.append(
                        obstruction(int(random.uniform(0.3, 0.7) * self.len), self.width - l_wall, 0, d_wall, l_wall,
                                    self.h))
            elif n_wall == 2:
                self.obs.append(obstruction(int(random.uniform(0.3, 0.4) * self.len), 0, 0, d_wall, l_wall, self.h))
                self.obs.append(obstruction(int(random.uniform(0.6, 0.7) * self.len), self.width - l_wall, 0, d_wall, l_wall, self.h))

            # 主梁（2个），x向
            self.obs.append(obstruction(0, 0, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))
            self.obs.append(obstruction(0, self.width - d_beam_main, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))

            # 次梁（2个），y向
            self.obs.append(obstruction(int(random.uniform(0.2, 0.35) * self.len), 0, self.h - d_beam_second, d_beam_second, self.width, d_beam_second))
            self.obs.append(obstruction(int(random.uniform(0.65, 0.8) * self.len), 0, self.h - d_beam_second, d_beam_second, self.width, d_beam_second))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                p = random.random()
                if p > 0.6:  # 原来为0.6
                    x = self.len - 1
                    y = int(random.uniform(0.1, 0.9) * self.width)
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                else:
                    x = int(random.uniform(0.1, 0.9) * self.len)  # 原来为（0.1，0.9）
                    y = 0
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                # 随机生成在附近都是无障碍的区域
                if flag == 0:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            for i in range(self.n_agent):
                # 随机生成智能体位置
                while (1):
                    flag = 0
                    x = 0
                    y = int(random.uniform(0.1, 0.9) * self.width)  # 原来为（0.1，0.9）
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                    if flag == 0:
                        break
                self.agents.append(AGENT(x, y, z, self))
                # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])

        # 机房
        if self.type == 2:
            # 定义规划空间大小，栅格长度为0.05m
            self.len = random.randrange(100, 300)  # 通常5m-20m
            self.width = int(self.len * random.uniform(0.7, 1))  # 通常长宽比不低于0.7
            self.h = int(max(self.len * random.uniform(0.3, 0.4), 50))  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            d_column = int(min(random.uniform(0.06, 0.1) * self.len, 30))  # 柱，几何宽度与房间大小成比例，最大不超过1.5m，长度贯穿房间
            d_beam_main = d_column  # 主梁
            d_beam_second = int(0.8 * d_column)  # 次梁

            # 柱(4个），房间四角
            self.obs.append(obstruction(0, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(0, self.width - d_column, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, self.width - d_column, 0, d_column, d_column, self.h))

            # 设备（1-2个）
            n_equip = random.randint(1, 2)
            if n_equip == 1:
                l_equip = int(random.uniform(0.5, 0.7) * self.len)  # 设备长
                w_equip = int(random.uniform(0.5, 0.7) * self.width)  # 设备宽
                h_equip = int(random.uniform(0.3, 0.7) * self.h)  # 设备高
                self.obs.append(obstruction(int(random.uniform(0.1, 0.2) * self.len), int(random.uniform(0.1, 0.2) * self.width), 0, l_equip, w_equip, h_equip))
            elif n_equip == 2:
                l_equip = int(random.uniform(0.2, 0.4) * self.len)  # 设备长
                w_equip = int(random.uniform(0.2, 0.4) * self.width)  # 设备宽
                h_equip = int(random.uniform(0.3, 0.5) * self.h)  # 设备高
                self.obs.append(obstruction(int(random.uniform(0.1, 0.2) * self.len), int(random.uniform(0.5, 0.6) * self.width), 0, l_equip, w_equip, h_equip))
                self.obs.append(obstruction(int(random.uniform(0.5, 0.6) * self.len), int(random.uniform(0.1, 0.2) * self.width), 0, l_equip, w_equip, h_equip))

            # 主梁（2个），x向
            self.obs.append(obstruction(0, 0, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))
            self.obs.append(obstruction(0, self.width - d_beam_main, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))

            # 次梁（2个），y向
            self.obs.append(obstruction(int(random.uniform(0.2, 0.35) * self.len), 0, self.h - d_beam_second, d_beam_second, self.width, d_beam_second))
            self.obs.append(obstruction(int(random.uniform(0.65, 0.8) * self.len), 0, self.h - d_beam_second, d_beam_second, self.width, d_beam_second))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                x = int(random.uniform(0.5, 0.9) * self.len)
                y = int(random.uniform(0.1, 0.9) * self.width)
                z = int(random.uniform(0.1, 0.4) * self.h)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if self.map[x + i, y + j, z + k] == 1 and self.map[x, y, z] == 0:
                                flag = 1
                if flag == 1:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            for _ in range(self.n_agent):
                while (1):
                    flag = 0
                    x = 0
                    y = int(random.uniform(0.1, 0.9) * self.width)
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                    if flag == 0:
                        break
                self.agents.append(AGENT(x, y, z, self))
                # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])

        # 走廊
        if self.type == 3:
            # 定义规划空间大小，栅格长度为0.05m
            self.len = random.randrange(600, 1000)  # 通常30m-50m
            self.width = random.randrange(40, 80)  # 通常2m-4m
            self.h = random.randrange(50, 70)  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            d_beam = random.randrange(4, 6)  # 梁
            dist_beam = random.randrange(80, 120)  # 梁的间距

            # 梁，y向
            for i in range(int(self.len / dist_beam) + 1):
                self.obs.append(obstruction(i * dist_beam, 0, self.h - d_beam, d_beam, self.width, d_beam))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                p = random.random()
                if p > 0.8:
                    x = self.len - 1
                    y = int(random.uniform(0.1, 0.9) * self.width)
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                else:
                    x = int(random.uniform(0.5, 0.9) * self.len)
                    y = 0
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                # 随机生成在附近都是无障碍的区域
                if flag == 0:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            for _ in range(self.n_agent):
                while (1):
                    flag = 0
                    x = 0
                    y = int(random.uniform(0.1, 0.9) * self.width)
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                    if flag == 0:
                        break
                self.agents.append(AGENT(x, y, z, self))
                # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])

        # L型走廊
        if self.type == 4:
            # 定义规划空间大小，栅格长度为0.05m
            self.len = random.randrange(400, 600)  # 通常10m-20m
            self.width = self.len
            self.h = random.randrange(50, 70)  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            width_cor = random.randrange(40, 80)   # 通常2m-4m
            d_beam = random.randrange(4, 6)  # 梁
            dist_beam = random.randrange(80, 120)  # 梁的间距

            # 填充立方体
            self.obs.append(obstruction(0, 0, 0, self.len - 1 - width_cor, self.width - 1 - width_cor, self.h))

            # 梁，y向
            for i in range(int((self.len - 1 - width_cor) / dist_beam) + 1):
                self.obs.append(
                    obstruction(i * dist_beam, self.width - 1 - width_cor, self.h - d_beam, d_beam, width_cor, d_beam))

            # 梁，x向
            for i in range(int((self.width - 1 - width_cor) / dist_beam) + 1):
                self.obs.append(
                    obstruction(self.len - 1 - width_cor, i * dist_beam, self.h - d_beam, width_cor, d_beam, d_beam))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                p = random.random()
                if p > 0.8:
                    x = int(self.len - 1 - width_cor + random.uniform(0.1, 0.9) * width_cor)
                    y = 0
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                else:
                    x = self.len - 1 - width_cor
                    y = int(random.uniform(0.1, 0.9) * (self.width - 1 - width_cor))
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                # 随机生成在附近都是无障碍的区域
                if flag == 0:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            for _ in range(self.n_agent):
                while (1):
                    flag = 0
                    x = 0
                    y = int(self.width - 1 - width_cor + random.uniform(0.1, 0.9) * width_cor)
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                    if flag == 0:
                        break
                self.agents.append(AGENT(x, y, z, self))
                # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])

        return self.state

    def reset_test(self):
        # 环境重组测试
        self.agents = []
        self.obs = []

        # 普通房间
        if self.type == 1:
            # 定义规划空间大小，栅格长度为0.05m
            # self.len = random.randrange(100, 300)  # 通常5m-20m
            # self.width = int(self.len * random.uniform(0.7, 1))  # 通常长宽比不低于0.7
            # self.h = int(max(self.len * random.uniform(0.3, 0.4), 50))  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            d_column = int(min(random.uniform(0.06, 0.1) * self.len, 30))  # 柱，几何宽度与房间大小成比例，最大不超过1.5m，长度贯穿房间
            d_beam_main = d_column  # 主梁
            d_beam_second = int(0.8 * d_column)  # 次梁
            l_wall = int(random.uniform(0.35, 0.5) * self.width)  # 隔墙长
            d_wall = d_beam_second  # 隔墙宽

            # 柱(4个），房间四角
            self.obs.append(obstruction(0, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(0, self.width - d_column, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, self.width - d_column, 0, d_column, d_column, self.h))

            # 隔墙（0-2个），沿次梁方向（y向）
            if self.len < 200:
                n_wall = 0  # 原来为0
            else:
                n_wall = random.randint(1, 2)
            if n_wall == 1:
                p = random.random()
                if p > 0.5:
                    self.obs.append(obstruction(int(random.uniform(0.3, 0.7) * self.len), 0, 0, d_wall, l_wall, self.h))
                else:
                    self.obs.append(
                        obstruction(int(random.uniform(0.3, 0.7) * self.len), self.width - l_wall, 0, d_wall, l_wall,
                                    self.h))
            elif n_wall == 2:
                self.obs.append(obstruction(int(random.uniform(0.3, 0.4) * self.len), 0, 0, d_wall, l_wall, self.h))
                self.obs.append(
                    obstruction(int(random.uniform(0.6, 0.7) * self.len), self.width - l_wall, 0, d_wall, l_wall,
                                self.h))

            # 主梁（2个），x向
            self.obs.append(obstruction(0, 0, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))
            self.obs.append(
                obstruction(0, self.width - d_beam_main, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))

            # 次梁（2个），y向
            self.obs.append(
                obstruction(int(random.uniform(0.2, 0.35) * self.len), 0, self.h - d_beam_second, d_beam_second,
                            self.width, d_beam_second))
            self.obs.append(
                obstruction(int(random.uniform(0.65, 0.8) * self.len), 0, self.h - d_beam_second, d_beam_second,
                            self.width, d_beam_second))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                p = random.random()
                if p > 0.6:  # 原来为0.6
                    x = self.len - 1
                    y = int(random.uniform(0.1, 0.9) * self.width)
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                else:
                    x = int(random.uniform(0.1, 0.9) * self.len)  # 原来为（0.1，0.9）
                    y = 0
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                # 随机生成在附近都是无障碍的区域
                if flag == 0:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            while (1):
                flag = 0
                x = 0
                y = int(random.uniform(0.1, 0.9) * self.width)  # 原来为（0.1，0.9）
                z = int(random.uniform(0.6, 0.9) * self.h)
                if self.map[x, y, z] == 1:
                    flag = 1
                if flag == 0:
                    break
            self.agents.append(AGENT(x, y, z, self))
            # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])

        # 机房
        if self.type == 2:
            # 定义规划空间大小，栅格长度为0.05m
            self.len = random.randrange(100, 300)  # 通常5m-20m
            self.width = int(self.len * random.uniform(0.7, 1))  # 通常长宽比不低于0.7
            self.h = int(max(self.len * random.uniform(0.3, 0.4), 50))  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            d_column = int(min(random.uniform(0.06, 0.1) * self.len, 30))  # 柱，几何宽度与房间大小成比例，最大不超过1.5m，长度贯穿房间
            d_beam_main = d_column  # 主梁
            d_beam_second = int(0.8 * d_column)  # 次梁

            # 柱(4个），房间四角
            self.obs.append(obstruction(0, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(0, self.width - d_column, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, 0, 0, d_column, d_column, self.h))
            self.obs.append(obstruction(self.len - d_column, self.width - d_column, 0, d_column, d_column, self.h))

            # 设备（1-2个）
            n_equip = random.randint(1, 2)
            if n_equip == 1:
                l_equip = int(random.uniform(0.5, 0.7) * self.len)  # 设备长
                w_equip = int(random.uniform(0.5, 0.7) * self.width)  # 设备宽
                h_equip = int(random.uniform(0.3, 0.7) * self.h)  # 设备高
                self.obs.append(obstruction(int(random.uniform(0.1, 0.2) * self.len),
                                            int(random.uniform(0.1, 0.2) * self.width), 0, l_equip, w_equip,
                                            h_equip))
            elif n_equip == 2:
                l_equip = int(random.uniform(0.2, 0.4) * self.len)  # 设备长
                w_equip = int(random.uniform(0.2, 0.4) * self.width)  # 设备宽
                h_equip = int(random.uniform(0.3, 0.5) * self.h)  # 设备高
                self.obs.append(obstruction(int(random.uniform(0.1, 0.2) * self.len),
                                            int(random.uniform(0.5, 0.6) * self.width), 0, l_equip, w_equip,
                                            h_equip))
                self.obs.append(obstruction(int(random.uniform(0.5, 0.6) * self.len),
                                            int(random.uniform(0.1, 0.2) * self.width), 0, l_equip, w_equip,
                                            h_equip))

            # 主梁（2个），x向
            self.obs.append(obstruction(0, 0, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))
            self.obs.append(
                obstruction(0, self.width - d_beam_main, self.h - d_beam_main, self.len, d_beam_main, d_beam_main))

            # 次梁（2个），y向
            self.obs.append(
                obstruction(int(random.uniform(0.2, 0.35) * self.len), 0, self.h - d_beam_second, d_beam_second,
                            self.width, d_beam_second))
            self.obs.append(
                obstruction(int(random.uniform(0.65, 0.8) * self.len), 0, self.h - d_beam_second, d_beam_second,
                            self.width, d_beam_second))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                x = int(random.uniform(0.5, 0.9) * self.len)
                y = int(random.uniform(0.1, 0.9) * self.width)
                z = int(random.uniform(0.1, 0.4) * self.h)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if self.map[x + i, y + j, z + k] == 1 and self.map[x, y, z] == 0:
                                flag = 1
                if flag == 1:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            while (1):
                flag = 0
                x = 0
                y = int(random.uniform(0.1, 0.9) * self.width)
                z = int(random.uniform(0.6, 0.9) * self.h)
                if self.map[x, y, z] == 1:
                    flag = 1
                if flag == 0:
                    break
            self.agents.append(AGENT(x, y, z, self))
            # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])

        # 走廊
        if self.type == 3:
            # 定义规划空间大小，栅格长度为0.05m
            self.len = random.randrange(600, 1000)  # 通常30m-100m
            self.width = random.randrange(40, 80)  # 通常2m-4m
            self.h = random.randrange(50, 70)  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            d_beam = random.randrange(4, 6)  # 梁
            dist_beam = random.randrange(80, 120)  # 梁的间距

            # 梁，y向
            for i in range(int(self.len / dist_beam) + 1):
                self.obs.append(obstruction(i * dist_beam, 0, self.h - d_beam, d_beam, self.width, d_beam))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                p = random.random()
                if p > 0.8:
                    x = self.len - 1
                    y = int(random.uniform(0.1, 0.9) * self.width)
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                else:
                    x = int(random.uniform(0.5, 0.9) * self.len)
                    y = 0
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                # 随机生成在附近都是无障碍的区域
                if flag == 0:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            while (1):
                flag = 0
                x = 0
                y = int(random.uniform(0.1, 0.9) * self.width)
                z = int(random.uniform(0.6, 0.9) * self.h)
                if self.map[x, y, z] == 1:
                    flag = 1
                if flag == 0:
                    break
            self.agents.append(AGENT(x, y, z, self))
            # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])

        # L型走廊
        if self.type == 4:
            # 定义规划空间大小，栅格长度为0.05m
            self.len = random.randrange(400, 600)  # 通常10m-20m
            self.width = self.len
            self.h = random.randrange(50, 70)  # 普通住宅层高不超过2.8m，写字楼层高不超过4.5m
            self.map = np.zeros((self.len, self.width, self.h))
            width_cor = random.randrange(40, 80)  # 通常2m-4m
            d_beam = random.randrange(4, 6)  # 梁
            dist_beam = random.randrange(80, 120)  # 梁的间距

            # 填充立方体
            self.obs.append(obstruction(0, 0, 0, self.len - 1 - width_cor, self.width - 1 - width_cor, self.h))

            # 梁，y向
            for i in range(int((self.len - 1 - width_cor) / dist_beam) + 1):
                self.obs.append(obstruction(i * dist_beam, self.width - 1 - width_cor, self.h - d_beam, d_beam, width_cor, d_beam))

            # 梁，x向
            for i in range(int((self.width - 1 - width_cor) / dist_beam) + 1):
                self.obs.append(obstruction(self.len - 1 - width_cor, i * dist_beam, self.h - d_beam, width_cor, d_beam, d_beam))

            for obs in self.obs:
                self.map[obs.x:obs.x + obs.l, obs.y:obs.y + obs.w, obs.z:obs.z + obs.h] = 1

            # 随机生成目标点位置
            while (1):
                flag = 0
                p = random.random()
                if p > 0.8:
                    x = int(self.len - 1 - width_cor + random.uniform(0.1, 0.9) * width_cor)
                    y = 0
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                else:
                    x = self.len - 1 - width_cor
                    y = int(random.uniform(0.1, 0.9) * (self.width - 1 - width_cor))
                    z = int(random.uniform(0.6, 0.9) * self.h)
                    if self.map[x, y, z] == 1:
                        flag = 1
                # 随机生成在附近都是无障碍的区域
                if flag == 0:
                    break
            self.target = [sn(x, y, z)]
            self.map[x, y, z] = -1

            # 随机生成智能体位置
            while (1):
                flag = 0
                x = 0
                y = int(self.width - 1 - width_cor + random.uniform(0.1, 0.9) * width_cor)
                z = int(random.uniform(0.6, 0.9) * self.h)
                if self.map[x, y, z] == 1:
                    flag = 1
                if flag == 0:
                    break
            self.agents.append(AGENT(x, y, z, self))
            # self.map[self.uavs[i].x,self.uavs[i].y,self.uavs[i].z]=1

            # 更新智能体状态
            self.state = np.vstack([uav.state() for (_, uav) in enumerate(self.agents)])
        return self.state


if __name__ == "__main__":
    env = Env()
    env.reset()
    env.render()
    plt.pause(30)
