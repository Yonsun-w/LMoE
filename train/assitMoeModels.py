"""
作者：${文家华}
日期：2022年11月04日
辅助任务
"""
# 辅助任务
import numpy as np
import torch

def converAssit(pre,label):
    batch = pre.shape[0]
    time = pre.shape[1]
    h = pre.shape[2]
    w = pre.shape[3]

    ## 取0.5作为阈值
    pre = torch.round(pre)

    pre_c = torch.sum(pre,1)
    label_c = torch.sum(label,1)

    pre_c[pre_c>0] = 1
    label_c[label_c>0] = 1


    return pre_c,label_c


## 按照时间分层 加权重
def converAssitByTime(pre,label):
    pre_c = torch.zeros_like(pre)
    label_c = torch.zeros_like(label)

    ## 取0.5作为阈值
    pre = torch.round(pre)

    # 前3个小时赋予较小权重 期望在长时间获得收益
    p_time = 3
    p_value = 0.5
    m_time = 6
    m_value = 0.7
    l_time = 12
    l_value = 1

    for i in range(p_time):
        print(" i = {}".format(i))
        pre_c += pre[:,i]
        label_c += label[:,i]
    pre_c *= p_value
    label_c *= p_value

    # 设置序列化参数

    for i in range(p_time,m_time):
        pre_c += pre[:,i]
        label_c += label[:,i]

    pre_c *= m_value
    label_c *= m_value

    for i in range(m_time,l_time):
        pre_c += pre[:,i]
        label_c += label[:,i]

    pre_c *= l_value
    label_c *= l_value

    pre_c = torch.sum(pre,1)
    label_c = torch.sum(label,1)
    return pre_c,label_c