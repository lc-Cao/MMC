# -*- coding: utf-8 -*-
"""
遗传算法(GA)通用模板：纯 numpy 实现，求解非线性/非凸优化(maximize)
依据：讲义第 6 章 6.5 Listing 7（30 行版扩展）

适用：线性规划解不了的非线性、非凸、多峰问题（如 Rastrigin、组合调度近似解）
提醒：能用线性/整数规划得精确解的，不要用 GA（精确 > 近似且快百倍）。
依赖：numpy
"""
import numpy as np


def ga(fitness, bounds, pop_size=50, gens=100, pm=0.15, sigma=0.3, seed=42):
    """最大化 fitness(x) 的遗传算法。
    fitness: callable, 输入 1D ndarray 输出标量(越大越好)
    bounds : list of (lo, hi)，长度=维度
    返回 (best_x, best_fit, history)
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # 初始化种群（均匀随机）
    pop = rng.uniform(0, 1, (pop_size, dim)) * (hi - lo) + lo
    history = []

    for gen in range(gens):
        fits = np.array([fitness(ind) for ind in pop])
        history.append(fits.max())

        # 精英保留 10%
        elite_n = max(1, pop_size // 10)
        idx = np.argsort(-fits)
        elite = pop[idx[:elite_n]]

        # 轮盘赌（按适应度归一化概率）
        p = fits - fits.min() + 1e-8
        p = p / p.sum()

        new_pop = list(elite)
        while len(new_pop) < pop_size:
            # 锦标赛式配对（讲义用概率抽样，这里改用更强壮的两两锦标赛）
            def pick():
                a, b = rng.choice(pop_size, 2, replace=False)
                return pop[a] if fits[a] >= fits[b] else pop[b]
            p1, p2 = pick(), pick()
            # 算术杂交：随机线性组合
            alpha = rng.random(dim)
            child = alpha * p1 + (1 - alpha) * p2
            # 高斯变异
            mask = rng.random(dim) < pm
            if mask.any():
                child[mask] += rng.normal(0, sigma, mask.sum())
            child = np.clip(child, lo, hi)      # 越界拉回边界
            new_pop.append(child)

        pop = np.array(new_pop[:pop_size])

    best_idx = np.argmax([fitness(ind) for ind in pop])
    best_x = pop[best_idx]
    return best_x, fitness(best_x), history


def demo():
    """max f = -(x1-3)^2 - (x2-2)^2 + 10,  xi in [-5,5] → 理论最优 (3,2), f=10"""
    def fitness(x):
        return -(x[0] - 3) ** 2 - (x[1] - 2) ** 2 + 10

    bounds = [(-5, 5), (-5, 5)]
    best_x, best_fit, hist = ga(fitness, bounds, pop_size=60, gens=150)
    print(f"最优解: x={best_x.round(4)}  f={best_fit:.4f}  (理论 10.0 @ [3,2])")
    print("末段收敛值(可画收敛曲线):", np.round(hist[-5:], 3))


if __name__ == "__main__":
    demo()
