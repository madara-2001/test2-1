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
        length = random.randrange(100, 300)
        width = int(length * random.uniform(0.7, 1))
        height = int(max(length * random.uniform(0.3, 0.4), 50))
    else:
        # 栅格大小为0.05m
        length = st.slider("长", 5.0, 15.0) * 20
        width = length / st.slider("长宽比", 1.0, 1.4) * 20
        height = st.slider("高", 2.5, 5.0) * 20

# 制作gif图

