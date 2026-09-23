# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 强化学习核心算法可视化工具
# --------------------------------------------


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_action_value(action_value_grid):
    """画出动作值分布（4个方向）"""
    top=action_value_grid[:,3].reshape((4,4))
    top_value_positions = [(0.38,0.25),(1.38,0.25),(2.38,0.25),(3.38,0.25),
                           (0.38,1.25),(1.38,1.25),(2.38,1.25),(3.38,1.25),
                           (0.38,2.25),(1.38,2.25),(2.38,2.25),(3.38,2.25),
                           (0.38,3.25),(1.38,3.25),(2.38,3.25),(3.38,3.25)]
    right=action_value_grid[:,2].reshape((4,4))
    right_value_positions = [(0.65,0.5),(1.65,0.5),(2.65,0.5),(3.65,0.5),
                           (0.65,1.5),(1.65,1.5),(2.65,1.5),(3.65,1.5),
                           (0.65,2.5),(1.65,2.5),(2.65,2.5),(3.65,2.5),
                           (0.65,3.5),(1.65,3.5),(2.65,3.5),(3.65,3.5),]
    bottom=action_value_grid[:,1].reshape((4,4))
    bottom_value_positions = [(0.38,0.8),(1.38,0.8),(2.38,0.8),(3.38,0.8),
                           (0.38,1.8),(1.38,1.8),(2.38,1.8),(3.38,1.8),
                           (0.38,2.8),(1.38,2.8),(2.38,2.8),(3.38,2.8),
                           (0.38,3.8),(1.38,3.8),(2.38,3.8),(3.38,3.8), ]
    left=action_value_grid[:,0].reshape((4,4))
    left_value_positions = [(0.05,0.5),(1.05,0.5),(2.05,0.5),(3.05,0.5),
                           (0.05,1.5),(1.05,1.5),(2.05,1.5),(3.05,1.5),
                           (0.05,2.5),(1.05,2.5),(2.05,2.5),(3.05,2.5),
                           (0.05,3.5),(1.05,3.5),(2.05,3.5),(3.05,3.5)]


    fig, ax=plt.subplots(figsize=(12,6))
    ax.set_ylim(4, 0)
    tripcolor = quatromatrix(left, top, right, bottom, ax=ax,
                 triplotkw={"color":"k", "lw":1},
                 tripcolorkw={"cmap": "coolwarm"}) 

    ax.margins(0)
    ax.set_aspect("equal")
    fig.colorbar(tripcolor)

    for i, (xi,yi) in enumerate(top_value_positions):
        plt.text(xi,yi,round(top.flatten()[i],2), size=9, color="w")
    for i, (xi,yi) in enumerate(right_value_positions):
        plt.text(xi,yi,round(right.flatten()[i],2), size=9, color="w")
    for i, (xi,yi) in enumerate(left_value_positions):
        plt.text(xi,yi,round(left.flatten()[i],2), size=9, color="w")
    for i, (xi,yi) in enumerate(bottom_value_positions):
        plt.text(xi,yi,round(bottom.flatten()[i],2), size=9, color="w")

    plt.show()
    
    
def quatromatrix(left, bottom, right, top, ax=None, triplotkw={},tripcolorkw={}):
    """给state value的三角上色"""

    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(), 
              right.flatten(), top.flatten()   ].flatten()

    triplot = ax.triplot(A[:,0], A[:,1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor


def qtable_directions_map(qtable, map_size):
    """根据动作分配箭头"""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_policy_map(q_values, env, map_size=4):
    """根据q values画出每个状态点执行的策略"""
    qtable_val_max, qtable_directions = qtable_directions_map(q_values, map_size)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        # ax=ax[0],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Policy")
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    plt.show()


def plot_state_value(state_value_grid):
    """画出状态价值分布"""
    p = sns.heatmap(state_value_grid, 
                    cmap='coolwarm',
                    annot=True,
                    fmt=".2f")


def plot_rewards_curve(rewards, title="Training Progress", window=100):
    """画出reward平滑曲线"""
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward")
    plt.grid(True)