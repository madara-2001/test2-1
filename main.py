import PIL.Image
import streamlit as st
import random
import base64
import io
import imageio.v2 as imageio
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time
from env import *
import torch
import time

with st.sidebar:
    # st.text("请选择房间类型\n")
    # type=st.selectbox(
    #     "none",
    #     ('普通房间','机房'),
    #     label_visibility="collapsed"
    #     )
    st.text("请输入房间尺寸\n")
    if st.button("随机生成房间尺寸"):
        length_temp = random.randrange(100, 300)
        width_temp = int(length * random.uniform(0.7, 1))
        height_temp = int(max(length * random.uniform(0.3, 0.4), 50))
    else:
        # 栅格大小为0.05m
        length_temp = st.slider("长", 5.0, 15.0) * 20
        width_temp = length / st.slider("长宽比", 1.0, 1.4) * 20
        height_temp = st.slider("高", 2.5, 5.0) * 20
    if st.button("确认"):
        length=length_temp
        width=width_temp
        height=heighth_temp

# 制作gif图

start_time = time.time()

pics = []


LEARNING_RATE = 0.00033  # 学习率
num_episodes = 80000  # 训练周期长度
space_dim = 140  # n_spaces   状态空间维度
action_dim = 7  # n_actions   动作空间维度
threshold = 200
env = Env(space_dim, action_dim, LEARNING_RATE)

# env.ax.view_init(elev=45,azim=45)
# env.ax.view_init(elev=0,azim=45)
env.ax.view_init(elev=45, azim=0)


check_point_Qlocal = torch.load('Qlocal.pth', map_location=torch.device('cpu'))
check_point_Qtarget = torch.load('Qtarget.pth', map_location=torch.device('cpu'))
env.q_target.load_state_dict(check_point_Qtarget['model'])
env.q_local.load_state_dict(check_point_Qlocal['model'])
env.optim.load_state_dict(check_point_Qlocal['optimizer'])
epoch = check_point_Qlocal['epoch']
# 真实场景运行
env.type = 1
env.len = length
env.width = width
env.h = height
state = env.reset_test()  # 环境重置1
total_reward = 0
env.render(1)
n_done = 0
count = 0

n_test = 1  # 测试次数
n_creash = 0  # 坠毁数目
success_count = 0

step = 0
gif = None

while 1:
    if env.agents[0].done:
        # 无人机已结束任务，跳过
        break
    action = env.get_action(FloatTensor(np.array([state[0]])), 0)  # 根据Q值选取动作

    next_state, reward, uav_done, info = env.step(action.item(), 0)  # 根据选取的动作改变状态，获取收益

    total_reward += reward  # 求总收益
    # 交互显示
    # print(action)
    env.render()
    # plt.pause(0.01)

    if step % int(env.agents[0].d_origin / 5) == 0:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        pics.append(imageio.imread(buffer))
        buffer.close()

    if uav_done:
        print(info)
        break
    if info == 1:
        success_count = success_count + 1

    step += 1

    state[0] = next_state  # 状态变更
# print(env.agents[0].step)
# print(env.state)
# print(env.agents[0].distance)
env.ax.scatter(env.target[0].x, env.target[0].y, env.target[0].z, c='red')
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
pics.append(imageio.imread(buffer))
buffer.close()

# plt.pause(5)
buffer = io.BytesIO()
imageio.mimsave(buffer, pics, duration=0.01, format='gif')
gif = imageio.mimread(buffer)
buffer.close()
buffer = io.BytesIO()
imageio.mimsave(buffer, gif, fps=10)
gif = imageio.mimread(buffer)
buffer.close()


end_time = time.time()  # 程序结束时间
run_time = end_time - start_time  # 程序的运行时间，单位为秒
print(run_time)

st.markdown(gif)
