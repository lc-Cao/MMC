# -*- coding: utf-8 -*-
"""
元胞自动机(森林火灾)最小模板 —— “局部规则→全局涌现”示例
依据：讲义第 7.5 节 Listing 9（本文件改用向量化写法，跑得快）

状态：0=空地  1=树  2=燃烧
规则：燃着的树→灰烬(0)；树只要 4-邻域有火→被点燃(2)
用途：换成你的“格子规则”类机理题（扩散/感染/交通流/晶格生长……）的母版。
依赖：numpy matplotlib
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def step(grid):
    """推进一格。返回新网格"""
    fire = grid == 2
    # 4-邻域(上下左右)是否有火
    pad = np.pad(fire, 1, constant_values=False)
    neigh = (pad[:-2, 1:-1] | pad[2:, 1:-1] |
             pad[1:-1, :-2] | pad[1:-1, 2:])
    new = grid.copy()
    new[(grid == 1) & neigh] = 2          # 树遇火点燃
    new[grid == 2] = 0                     # 燃烧后变灰烬
    return new


def main():
    rng = np.random.default_rng(7)
    n = 80
    p = 0.70                                  # 树木密度(≥0.65 保证能大面积蔓延)
    tree0 = rng.random((n, n)) < p
    grid = np.where(tree0, 1, 0)
    grid[n // 2, n // 2] = 2                  # 中心点火

    n_burn_max, step_at = 0, 0
    n_burning = []                            # 每步燃烧数(用于画曲线)
    frames = []
    s = 0
    while s < 300:
        burning = int((grid == 2).sum())
        n_burning.append(burning)
        if burning > n_burn_max:
            n_burn_max, step_at = burning, s
        grid = step(grid)
        s += 1
        if burning == 0 and s > 3:            # 火已灭
            break
    # 关键帧(均匀抽 ≤6 张)
    total = len(n_burning)
    idx = np.unique(np.linspace(0, total - 1, min(6, total)).astype(int)).tolist()
    frames = idx

    burned = int(((tree0 == 1) & (grid != 1)).sum())
    print(f"共 {s} 步火势结束 | 最大同燃树数={n_burn_max}(第{step_at}步) | "
          f"过火烧毁={burned} 棵 (占初始树木 {burned / (tree0 == 1).sum():.0%})")

    # 左图：燃烧规模曲线；右图：关键帧拼图
    fig = plt.figure(figsize=(6.5, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(total), n_burning, color="#c53030")
    ax1.set_xlabel("步数"); ax1.set_ylabel("燃烧格数"); ax1.set_title("燃烧规模演化")
    ax2 = fig.add_subplot(1, 2, 2)
    # 重跑一遍只为了取关键帧网格(小规模, 快)
    g2 = np.where(tree0, 1, 0); g2[n // 2, n // 2] = 2
    for j in range(total):
        g2 = step(g2)
        if j in idx:
            ax2.imshow(g2, cmap="YlOrBr_r", vmin=0, vmax=2)
            ax2.axis("off")
    ax2.set_title("末态 (黄=燃烧, 橙=灰, 蓝黑=树)")
    plt.tight_layout(); plt.savefig("森林火灾CA.png", dpi=150)
    print("已保存: 森林火灾CA.png")


if __name__ == "__main__":
    main()
