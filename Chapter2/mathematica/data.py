# -*- coding:utf-8 -*-

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib
import math


def Bspk_code(size, plot=True):
    sampling_t = 0.01
    t = np.arange(0, size, sampling_t)

    # 随机生成信号序列
    a = np.random.randint(0, 2, size)
    m = np.zeros(len(t), dtype=np.float32)
    for i in range(len(t)):
        m[i] = a[math.floor(t[i])]
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title('generate Random Binary signal', fontsize=20)
        plt.axis([0, size, -0.5, 1.5])
        plt.plot(t, m, 'b')
    fc = 4000
    fs = 20 * fc  # 采样频率
    ts = np.arange(0, (100 * size) / fs, 1 / fs)
    coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))
    bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4) + 1
    # BPSK调制信号波形
    if plot:
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title('BPSK Modulation', fontsize=20)  # , fontproperties=zhfont1
        plt.axis([0, size, -0.5, 2.5])
        plt.plot(t, bpsk, 'r')
        plt.show()
    return m, bpsk

def Qspk_code(size, plot=True):
    t = np.arange(0, 8.5, 0.5)
    # input
    plt.subplot(4, 1, 1)
    y1 = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0]
    plt.plot(t, y1, drawstyle='steps-post')
    plt.xlim(0, 8)
    plt.ylim(-0.5, 1.5)
    plt.title('Input Signal')

    # I Signal
    plt.subplot(4, 1, 2)
    a = 1 / np.sqrt(2)
    tI = np.arange(0, 9, 1)
    yI = [-a, a, -a, a, -a, a, -a, a, a]
    plt.plot(tI, yI, drawstyle='steps-post')
    plt.xlim(0, 8)
    plt.ylim(-2, 2)
    plt.title('I signal')

    # Q signal
    plt.subplot(4, 1, 3)
    yQ = [a, -a, -a, a, a, -a, -a, a, a]
    plt.plot(tI, yQ, drawstyle='steps-post')
    plt.xlim(0, 8)
    plt.ylim(-1, 1)
    plt.title('Q Signal')

    # QPSK signal
    plt.subplot(4, 1, 4)
    t = np.arange(0, 9., 0.01)

    def outputwave(I, Q, t):
        rectwav = []
        for i in range(len(I)):
            t_tmp = t[((i) * 100):((i + 1) * 100)]
            yI_tmp = yI[i] * np.ones(100)
            yQ_tmp = yQ[i] * np.ones(100)
            wav_tmp = yI_tmp * np.cos(2 * np.pi * 5 * t_tmp) - yQ_tmp * np.sin(2 * np.pi * 5 * t_tmp)
            rectwav.append(wav_tmp)
        return rectwav

    rectwav = outputwave(yI, yQ, t)
    plt.plot(t, np.array(rectwav).flatten(), 'r')
    plt.xlim(0, 8)
    plt.ylim(-2, 2)
    plt.title('QPSK Signal')

    plt.tight_layout()
    plt.show()
