from matplotlib import pyplot as plt
from math import *


def U(t_scale, tou, u_t_1, I, R, u_rest):
    return (-(u_t_1 - u_rest) + R*I)/tou*t_scale + u_t_1


ts = [i * 0.01 for i in range(10000)]
pulse = [1 if int(i * 0.01) % 40 == 0 else 0 for i in range(10000)]
# pulse = [sin(i)+0.5 for i in ts]

# 激励1ms然后衰弱9ms

us = [1]

for i, t in enumerate(ts):
    us.append(U(0.01, 10, us[-1], pulse[i], 10, 1))

plt.figure(figsize=(24, 8))
plt.plot(ts, us[1:])
plt.savefig("C:\\Users\\Administrator\\Desktop\\西电学习资料\\Neuronal Dynamics阅读笔记\\方波输入响应图.pdf")
