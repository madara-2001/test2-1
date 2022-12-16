######################################################################
# UAV Class
# ---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
# UAV 类描述，对无人机的状态参数进行初始化，
# 包括坐标、目标队列、环境、电量、方向、基础能耗、当前能耗、已经消耗能量、
# 侦测半径、周围障碍情况、坠毁概率、距离目标距离、已走步长等。
# 成员函数能返回自身状态，并根据决策行为对自身状态进行更新。
# ----------------------------------------------------------------
# UAV class description, initialize the state parameters of the UAV, 
# including coordinates, target queue, environment, power, direction, 
# basic energy consumption, current energy consumption, consumed energy, 
# detection radius, surrounding obstacles, crash probability, distance Target distance, steps taken, etc. 
# Member functions can return their own state and update their own state according to the decision-making behavior.
#################################################################
import math
import random
import numpy as np


class AGENT():
    def __init__(self, x, y, z, ev):
        # 初始化智能体坐标位置
        self.x = x
        self.y = y
        self.z = z
        # 初始化智能体目标坐标
        self.target = [ev.target[0].x, ev.target[0].y, ev.target[0].z]
        self.ev = ev  # 智能体所处环境
        # 初始化智能体运动情况
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(
            self.z - self.target[2])  # 无人机当前距离目标点曼哈顿距离
        self.d_origin = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(
            self.z - self.target[2])  # 无人机初始状态距离终点的曼哈顿距离
        self.step = 0  # 智能体已走步数
        # self.bt = 2 * self.d_origin  # 智能体电量，定义为初始曼哈顿距离的两倍
        self.dir = 1  # 智能体上一步运动方向，六种情况
        self.turn_count = 0  # 转弯次数
        self.install_count = 0  # 就近安装次数
        # self.p_bt = 1  # 智能体基础能耗，能耗/步
        # self.now_bt = 4  # 智能体当前状态能耗
        # self.cost = 0  # 智能体已经消耗能量
        # self.ob_space = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 无人机邻近栅格障碍物情况（26个）
        self.nearest_distance = 0  # 最近障碍物曼哈顿距离
        # self.dir_ob_xy = 0  # 距离智能体最近障碍物相对于智能体的水平方位
        # self.dir_ob_h = 10 # 距离智能体最近障碍物相对于智能体的竖直高度
        # self.p_crash = 0  # 智能体坠毁概率
        self.done = False  # 终止状态
        self.info = 0
        self.stop_count = 0
        self.back_count = 0
        self.previous = 0
        self.flag_abnormal = 0
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dz = self.target[2] - self.z
        self.obinfo = []
        if self.x < 0 or self.x > self.ev.len - 1 or self.y < 0 or self.y > self.ev.width - 1 or self.z < 0 or self.z > self.ev.h - 1:
            for _ in range(78):
                self.obinfo.append(0)
        else:
            sig_x = int(dx / max(abs(dx), 1))
            sig_y = int(dy / max(abs(dy), 1))
            sig_z = int(dz / max(abs(dz), 1))

            # 智能体
            # 沿x轴
            if sig_x == 0:
                for _ in range(21):
                    self.obinfo.append(-200)
            else:
                list_x = list(map(int, np.linspace(self.x + sig_x, self.target[0], abs(dx))))
                # 1
                for i in list_x:
                    if self.ev.map[i, self.y, self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y + n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y - n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 2
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] + n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] - n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 3
                for i in list_x:
                    if self.ev.map[i, self.y, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y + n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y - n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 4
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] + n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] - n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 5
                i = 1
                while 1:
                    if self.x - sig_x * i < 0 or self.x - sig_x * i > self.ev.len - 1:
                        self.obinfo.append(abs(-sig_x * i))
                        break
                    elif self.ev.map[self.x - sig_x * i, self.y, self.z] == 1:
                        self.obinfo.append(abs(-sig_x * i))
                        break
                    i += 1

            # 沿y轴
            if sig_y == 0:
                for _ in range(21):
                    self.obinfo.append(-200)
            else:
                list_y = list(map(int, np.linspace(self.y + sig_y, self.target[1], abs(dy))))
                # 6
                for i in list_y:
                    if self.ev.map[self.x, i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 7
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 8
                for i in list_y:
                    if self.ev.map[self.x, i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 9
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 10
                i = 1
                while 1:
                    if self.y - sig_y * i < 0 or self.y - sig_y * i > self.ev.width - 1:
                        self.obinfo.append(abs(-sig_y * i))
                        break
                    elif self.ev.map[self.x, self.y - sig_y * i, self.z] == 1:
                        self.obinfo.append(abs(-sig_y * i))
                        break
                    i += 1

            # 沿z轴
            if sig_z == 0:
                for _ in range(21):
                    self.obinfo.append(-200)
            else:
                list_z = list(map(int, np.linspace(self.z + sig_z, self.target[2], abs(dz))))
                # 11
                for i in list_z:
                    if self.ev.map[self.x, self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.y + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.y - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 12
                for i in list_z:
                    if self.ev.map[self.target[0], self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.y + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.y - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 13
                for i in list_z:
                    if self.ev.map[self.x, self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.target[1] + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.target[1] - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 14
                for i in list_z:
                    if self.ev.map[self.target[0], self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.target[1] + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.target[1] - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 15
                i = 1
                while 1:
                    if self.z - sig_z * i < 0 or self.z - sig_z * i > self.ev.h - 1:
                        self.obinfo.append(abs(-sig_z * i))
                        break
                    elif self.ev.map[self.x, self.y, self.z - sig_z * i] == 1:
                        self.obinfo.append(abs(-sig_z * i))
                        break
                    i += 1

            # 目标点
            # 沿x轴
            if sig_x == 0:
                for _ in range(5):
                    self.obinfo.append(-200)
            else:
                list_x = list(map(int, np.linspace(self.target[0], self.x + sig_x, abs(dx))))
                for i in list_x:
                    if self.ev.map[i, self.y, self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                for i in list_x:
                    if self.ev.map[i, self.y, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                i = 1
                while 1:
                    if self.target[0] + sig_x * i < 0 or self.target[0] + sig_x * i > self.ev.len - 1:
                        self.obinfo.append(abs(sig_x * i))
                        break
                    elif self.ev.map[self.target[0] + sig_x * i, self.target[1], self.target[2]] == 1:
                        self.obinfo.append(abs(sig_x * i))
                        break
                    i += 1

            # 沿y轴
            if sig_y == 0:
                for _ in range(5):
                    self.obinfo.append(-200)
            else:
                list_y = list(map(int, np.linspace(self.target[1], self.y + sig_y, abs(dy))))
                for i in list_y:
                    if self.ev.map[self.x, i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                for i in list_y:
                    if self.ev.map[self.x, i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                i = 1
                while 1:
                    if self.target[1] + sig_y * i < 0 or self.target[1] + sig_y * i > self.ev.width - 1:
                        self.obinfo.append(abs(sig_y * i))
                        break
                    elif self.ev.map[self.target[0], self.target[1] + sig_y * i, self.target[2]] == 1:
                        self.obinfo.append(abs(sig_y * i))
                        break
                    i += 1

            # 沿z轴
            if sig_z == 0:
                for _ in range(5):
                    self.obinfo.append(-200)
            else:
                list_z = list(map(int, np.linspace(self.target[2], self.z + sig_z, abs(dz))))
                for i in list_z:
                    if self.ev.map[self.x, self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                for i in list_z:
                    if self.ev.map[self.target[0], self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                for i in list_z:
                    if self.ev.map[self.x, self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                for i in list_z:
                    if self.ev.map[self.target[0], self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                i = 1
                while 1:
                    if self.target[2] + sig_z * i < 0 or self.target[2] + sig_z * i > self.ev.h - 1:
                        self.obinfo.append(abs(sig_z * i))
                        break
                    elif self.ev.map[self.target[0], self.target[1], self.target[2] + sig_z * i] == 1:
                        self.obinfo.append(abs(sig_z * i))
                        break
                    i += 1

    def state(self):
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dz = self.target[2] - self.z
        state_grid = [self.x, self.y, self.z, dx, dy, dz, 100 * self.dir, self.distance, self.step, self.d_origin]  # 前三个必须是self.x, self.y, self.z
        agent_round = []

        # 更新智能体临近栅格状态
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k ==0:
                        continue
                    if self.x + i < 0 or self.x + i >= self.ev.len or self.y + j < 0 or self.y + j >= self.ev.width or self.z + k < 0 or self.z + k >= self.ev.h:
                        state_grid.append(100)
                        agent_round.append(100)
                    else:
                        state_grid.append(100 * self.ev.map[self.x + i, self.y + j, self.z + k])
                        agent_round.append(100 * self.ev.map[self.x + i, self.y + j, self.z + k])

        # 更新目标点临近栅格状态
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k ==0:
                        continue
                    if self.target[0] + i < 0 or self.target[0] + i >= self.ev.len or self.target[1] + j < 0 or self.target[1] + j >= self.ev.width or self.target[2] + k < 0 or self.target[2] + k >= self.ev.h:
                        state_grid.append(100)
                    else:
                        state_grid.append(100 * self.ev.map[self.target[0] + i, self.target[1] + j, self.target[2] + k])

        state_grid = state_grid + self.obinfo
        return state_grid

    def update(self, action):
        # 更新智能体状态
        dx_t, dy_t, dz_t = [0, 0, 0]
        temp_x = self.x
        temp_y = self.y
        temp_z = self.z
        # print(temp)
        # 相关参数
        w_away = 2  # 远离扣分/靠近得分，避免原地往复
        # b = 3  # 撞毁参数 原3
        # wt = 0.005  # 目标参数
        # wc = 0  # 爬升参数 原0.07
        # we = 0  # 能量损耗参数  原0.2
        # c = 0.05  # 风阻能耗参数 原0.05
        # crash = 0  # 坠毁概率惩罚增益倍数 原3
        Ddistance = 0  # 距离终点的距离变化量
        self.previous = 0
        self.flag_abnormal = 0

        # 计算无人机坐标变更值
        if action == 0:
            dx_t, dy_t, dz_t = [0, 0, 0]
            self.stop_count += 1
            self.flag_abnormal = 1
        elif action == 1:
            dx_t, dy_t, dz_t = [1, 0, 0]
        elif action == 2:
            dx_t, dy_t, dz_t = [0, 1, 0]
        elif action == 3:
            dx_t, dy_t, dz_t = [0, 0, 1]
        elif action == 4:
            dx_t, dy_t, dz_t = [-1, 0, 0]
        elif action == 5:
            dx_t, dy_t, dz_t = [0, -1, 0]
        elif action == 6:
            dx_t, dy_t, dz_t = [0, 0, -1]
        else:
            print('Error')

        if action != 0:
            self.ev.map[self.x, self.y, self.z] = 2

        # # 如果无人机静止不动，给予大量惩罚
        # if dx_t == 0 and dy_t == 0 and dz_t == 0:
        #     return -1000, False, False
        self.x = self.x + dx_t
        self.y = self.y + dy_t
        self.z = self.z + dz_t

        # 标注管道位置（暂时有些问题，管道不能在下一个智能体生成时消除）
        # if self.x <= 0 or self.x >= self.ev.len - 1 or self.y <= 0 or self.y >= self.ev.width - 1 or self.z <= 0 or self.z >= self.ev.h - 1 or \
        #         self.ev.map[self.x, self.y, self.z] == 1:
        #     pass
        # else:
        #     self.ev.map[self.x, self.y, self.z] = 2

        Ddistance = self.distance - (abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(
            self.z - self.target[2]))  # 正代表接近目标，负代表远离目标
        self.distance = abs(self.x - self.target[0]) + abs(self.y - self.target[1]) + abs(
            self.z - self.target[2])  # 更新距离值
        self.step += 1

        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dz = self.target[2] - self.z

        self.obinfo = []
        if self.x < 0 or self.x > self.ev.len - 1 or self.y < 0 or self.y > self.ev.width - 1 or self.z < 0 or self.z > self.ev.h - 1:
            for _ in range(78):
                self.obinfo.append(0)
        else:
            sig_x = int(dx / max(abs(dx), 1))
            sig_y = int(dy / max(abs(dy), 1))
            sig_z = int(dz / max(abs(dz), 1))

            # 智能体
            # 沿x轴
            if sig_x == 0:
                for _ in range(21):
                    self.obinfo.append(-200)
            else:
                list_x = list(map(int, np.linspace(self.x + sig_x, self.target[0], abs(dx))))
                # 1
                for i in list_x:
                    if self.ev.map[i, self.y, self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y + n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y - n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 2
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] + n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] - n, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 3
                for i in list_x:
                    if self.ev.map[i, self.y, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y + n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y - n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.y, self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 4
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.target[0]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] + n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1] - n, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[i, self.target[1], self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 5
                i = 1
                while 1:
                    if self.x - sig_x * i < 0 or self.x - sig_x * i > self.ev.len - 1:
                        self.obinfo.append(abs(-sig_x * i))
                        break
                    elif self.ev.map[self.x - sig_x * i, self.y, self.z] == 1:
                        self.obinfo.append(abs(-sig_x * i))
                        break
                    i += 1

            # 沿y轴
            if sig_y == 0:
                for _ in range(21):
                    self.obinfo.append(-200)
            else:
                list_y = list(map(int, np.linspace(self.y + sig_y, self.target[1], abs(dy))))
                # 6
                for i in list_y:
                    if self.ev.map[self.x, i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 7
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.z + n < 0 or self.z + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.z + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.z - n < 0 or self.z - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.z - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, i, self.z] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 8
                for i in list_y:
                    if self.ev.map[self.x, i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, i, self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 9
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.target[1]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[2] + n < 0 or self.target[2] + n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.target[2] + n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[2] - n < 0 or self.target[2] - n > self.ev.h - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], i, self.target[2] - n] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, i, self.target[2]] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 10
                i = 1
                while 1:
                    if self.y - sig_y * i < 0 or self.y - sig_y * i > self.ev.width - 1:
                        self.obinfo.append(abs(-sig_y * i))
                        break
                    elif self.ev.map[self.x, self.y - sig_y * i, self.z] == 1:
                        self.obinfo.append(abs(-sig_y * i))
                        break
                    i += 1

            # 沿z轴
            if sig_z == 0:
                for _ in range(21):
                    self.obinfo.append(-200)
            else:
                list_z = list(map(int, np.linspace(self.z + sig_z, self.target[2], abs(dz))))
                # 11
                for i in list_z:
                    if self.ev.map[self.x, self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.y + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.y - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 12
                for i in list_z:
                    if self.ev.map[self.target[0], self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, self.y, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y + n < 0 or self.y + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.y + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.y - n < 0 or self.y - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.y - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 13
                for i in list_z:
                    if self.ev.map[self.x, self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.x + n < 0 or self.x + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x + n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.x - n < 0 or self.x - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x - n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.target[1] + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.x, self.target[1] - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 14
                for i in list_z:
                    if self.ev.map[self.target[0], self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.target[2]:
                        self.obinfo.append(-200)

                if self.obinfo[-1] == -200:
                    for _ in range(4):
                        self.obinfo.append(-200)
                else:
                    n = 1
                    while 1:
                        if self.target[0] + n < 0 or self.target[0] + n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] + n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[0] - n < 0 or self.target[0] - n > self.ev.len - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0] - n, self.target[1], i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] + n < 0 or self.target[1] + n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.target[1] + n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break
                    n = 1
                    while 1:
                        if self.target[1] - n < 0 or self.target[1] - n > self.ev.width - 1:
                            self.obinfo.append(-100)
                            break
                        if self.ev.map[self.target[0], self.target[1] - n, i] == 1:
                            n += 1
                        else:
                            self.obinfo.append(n)
                            break

                # 15
                i = 1
                while 1:
                    if self.z - sig_z * i < 0 or self.z - sig_z * i > self.ev.h - 1:
                        self.obinfo.append(abs(-sig_z * i))
                        break
                    elif self.ev.map[self.x, self.y, self.z - sig_z * i] == 1:
                        self.obinfo.append(abs(-sig_z * i))
                        break
                    i += 1

            # 目标点
            # 沿x轴
            if sig_x == 0:
                for _ in range(5):
                    self.obinfo.append(-200)
            else:
                list_x = list(map(int, np.linspace(self.target[0], self.x + sig_x, abs(dx))))
                for i in list_x:
                    if self.ev.map[i, self.y, self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.z] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                for i in list_x:
                    if self.ev.map[i, self.y, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                for i in list_x:
                    if self.ev.map[i, self.target[1], self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.x))
                        break
                    if i == self.x + sig_x:
                        self.obinfo.append(-200)
                i = 1
                while 1:
                    if self.target[0] + sig_x * i < 0 or self.target[0] + sig_x * i > self.ev.len - 1:
                        self.obinfo.append(abs(sig_x * i))
                        break
                    elif self.ev.map[self.target[0] + sig_x * i, self.target[1], self.target[2]] == 1:
                        self.obinfo.append(abs(sig_x * i))
                        break
                    i += 1

            # 沿y轴
            if sig_y == 0:
                for _ in range(5):
                    self.obinfo.append(-200)
            else:
                list_y = list(map(int, np.linspace(self.target[1], self.y + sig_y, abs(dy))))
                for i in list_y:
                    if self.ev.map[self.x, i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.z] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                for i in list_y:
                    if self.ev.map[self.x, i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                for i in list_y:
                    if self.ev.map[self.target[0], i, self.target[2]] == 1:
                        self.obinfo.append(abs(i - self.y))
                        break
                    if i == self.y + sig_y:
                        self.obinfo.append(-200)
                i = 1
                while 1:
                    if self.target[1] + sig_y * i < 0 or self.target[1] + sig_y * i > self.ev.width - 1:
                        self.obinfo.append(abs(sig_y * i))
                        break
                    elif self.ev.map[self.target[0], self.target[1] + sig_y * i, self.target[2]] == 1:
                        self.obinfo.append(abs(sig_y * i))
                        break
                    i += 1

            # 沿z轴
            if sig_z == 0:
                for _ in range(5):
                    self.obinfo.append(-200)
            else:
                list_z = list(map(int, np.linspace(self.target[2], self.z + sig_z, abs(dz))))
                for i in list_z:
                    if self.ev.map[self.x, self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                for i in list_z:
                    if self.ev.map[self.target[0], self.y, i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                for i in list_z:
                    if self.ev.map[self.x, self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                for i in list_z:
                    if self.ev.map[self.target[0], self.target[1], i] == 1:
                        self.obinfo.append(abs(i - self.z))
                        break
                    if i == self.z + sig_z:
                        self.obinfo.append(-200)
                i = 1
                while 1:
                    if self.target[2] + sig_z * i < 0 or self.target[2] + sig_z * i > self.ev.h - 1:
                        self.obinfo.append(abs(sig_z * i))
                        break
                    elif self.ev.map[self.target[0], self.target[1], self.target[2] + sig_z * i] == 1:
                        self.obinfo.append(abs(sig_z * i))
                        break
                    i += 1

        # print(self.obinfo)

        # 计算转角与相关奖励
        r_turn = 0
        if self.dir != action and action != 0:
            self.turn_count += 1
            r_turn = -2
        # # 回头
        # if abs(self.dir - action) == 3 and action != 0:
        #     self.back_count += 1
        #     self.flag_abnormal = 1
        #     r_turn = -1
        if action != 0:
            self.dir = action  # 更新方向

        # 计算最近障碍物位置与相关奖励
        self.install_count += 1
        r_install = -1
        if not (self.x < 0 or self.x > self.ev.len - 1 or self.y < 0 or self.y > self.ev.width - 1 or self.z < 0 or self.z > self.ev.h - 1):  # self.z <= 0是为了避免贴地走
            if self.ev.map[self.x, self.y, self.z] != 1:
                if self.x + 1 > self.ev.len - 1 or self.x - 1 < 0:
                    r_install = -1
                elif self.ev.map[self.x + 1, self.y, self.z] == 1 or self.ev.map[self.x - 1, self.y, self.z] == 1:
                    r_install = -1
                if self.y + 1 > self.ev.width - 1 or self.y - 1 < 0:
                    r_install = -1
                elif self.ev.map[self.x, self.y + 1, self.z] == 1 or self.ev.map[self.x, self.y - 1, self.z] == 1:
                    r_install = -1
                if self.z + 1 > self.ev.h - 1:
                    r_install = 0
                elif self.ev.map[self.x, self.y, self.z + 1] == 1:
                    r_install = 0
                if self.z - 1 < 0:
                    r_install = -1
                elif self.ev.map[self.x, self.y, self.z - 1] == 1:
                    r_install = -1

        if r_install == -1:
            self.install_count = self.install_count - 1

        # 判断是否陷入死角
        flag_dead = 0
        flag_out_x = 0
        flag_out_y = 0
        flag_out_z = 0

        # x
        count = 0
        list_t = self.obinfo[1:5]
        # print(list_t)
        for j in range(4):
            if list_t[j] == -100:
                count += 1
        if count >= 3:
            flag_dead = 1
            if list_t[0] >= 0:
                flag_out_y = 1
            if list_t[1] >= 0:
                flag_out_y = -1
            if list_t[2] >= 0:
                flag_out_z = 1
            if list_t[3] >= 0:
                flag_out_z = -1


        # y
        count = 0
        list_t = self.obinfo[22:26]
        # print(list_t)
        for j in range(4):
            if list_t[j] == -100:
                count += 1
        if count >= 3:
            flag_dead = 1
            if list_t[0] >= 0:
                flag_out_z = 1
            if list_t[1] >= 0:
                flag_out_z = -1
            if list_t[2] >= 0:
                flag_out_x = 1
            if list_t[3] >= 0:
                flag_out_x = -1

        # z
        count = 0
        list_t = self.obinfo[43:47]
        # print(list_t)
        for j in range(4):
            if list_t[j] == -100:
                count += 1
        if count >= 3:
            flag_dead = 1
            if list_t[0] >= 0:
                flag_out_x = 1
            if list_t[1] >= 0:
                flag_out_x = -1
            if list_t[2] >= 0:
                flag_out_y = 1
            if list_t[3] >= 0:
                flag_out_y = -1

        # print('flag:', flag_dead)

        # 走出死角奖励
        if flag_dead == 0:
            r_out = 0
        else:
            r_out = 2 * (flag_out_x * dx_t + flag_out_y * dy_t + flag_out_z * dz_t)

        # 靠近奖励
        if Ddistance > 0:
            r_target = 1
        else:
            r_target = -1.5

        # 移动奖励
        r_move = 0
        if action == 0:
            r_move = -1

        # # 爬升奖励
        # r_climb = 0
        # if dz_t == 1:
        #     r_climb = 2


        # 计算总奖励
        r = r_target + r_out + r_move + r_turn + r_install

        # # 步长过长，给予惩罚
        # if self.step > 2 * self.d_origin:
        #    r = r - (self.step - 2 * self.d_origin) / 2 * self.d_origin

        # 终止状态判断
        # 到达目标点，给予大量奖励
        # self.ev.map[self.x,self.y,self.z]=0
        # print(self.step)
        if self.distance == 0:
            self.done = True
            self.info = 1
            return r + 100, True, self.info

        # 发生碰撞，产生惩罚
        # print(self.step)
        if self.x < 0 or self.x > self.ev.len - 1 or self.y < 0 or self.y > self.ev.width - 1 or self.z < 0 or self.z > self.ev.h - 1:
            self.done = True
            self.info = 2
            return r - 10, True, self.info
        elif self.ev.map[self.x, self.y, self.z] == 1:
            self.done = True
            self.info = 2
            return r - 10, True, self.info

        # 步数超限
        if self.step > 5 * self.d_origin:
            self.done = True
            self.info = 3
            return r - 10, True, self.info

        # 管道碰撞
        if self.ev.map[self.x, self.y, self.z] == 2:
            self.done = True
            self.info = 4
            return r - 10, True, self.info

        self.info = 0
        return r, False, self.info
