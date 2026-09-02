# -*- coding: utf-8 -*-
"""
一维瞬态热传导有限差分模板(Crank-Nicolson 隐式)——A 题热/温控/防护类场景主力工具
物理依据：傅里叶定律 + 能量守恒
    rho * c * dT/dt = d/dx ( k * dT/dx )      （热扩散率 alpha = k/(rho c)）

特性：
    - 支持多层材料(各自 k/rho/c/厚度)、左边界给定温度、右边界绝热(可按题改)
    - 隐式(C-N)格式无条件稳定，可用较大时间步，避免显式格式的 dt 限制
    - 依赖 numpy，求解用三对角 Thomas 追赶法
依赖：numpy matplotlib
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_mesh(layers, nx=200):
    """layers: list of dict {k, rho, c, thickness(m)}（从热源侧向外排列）
    返回: x 坐标(m), dx, 总长, 逐网格热扩散率 alpha 数组
    """
    thicknesses = [ly["thickness"] for ly in layers]
    L = sum(thicknesses)
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx
    alpha = np.empty(nx)
    edges = np.cumsum(thicknesses)
    idx = 0
    for i, ly in enumerate(layers):
        edge = edges[i]
        while idx < nx and x[idx] <= edge:
            alpha[idx] = ly["k"] / (ly["rho"] * ly["c"])
            idx += 1
    return x, dx, L, alpha


def _thomas(a, b, c, r):
    """解三对角方程组 a_j x_{j-1}+b_j x_j+c_j x_{j+1}=r_j"""
    n = len(b)
    cp = np.zeros(n); dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = r[0] / b[0]
    for j in range(1, n):
        m = b[j] - a[j] * cp[j - 1]
        cp[j] = c[j] / m
        dp[j] = (r[j] - a[j] * dp[j - 1]) / m
    x[-1] = dp[-1]
    for j in range(n - 2, -1, -1):
        x[j] = dp[j] - cp[j] * x[j + 1]
    return x


def solve_heat(alpha, dx, nx, T0=33.0, T_hot=75.0, t_total=5400.0,
               dt=2.0, theta=0.5, save_every=60):
    """隐式 θ 格式(θ=1 全隐式，θ=0.5 即 Crank-Nicolson)。
    左边界固定 T_hot，右边界绝热 dT/dx=0。
    返回 (times, T)  T: (快照数, nx)，每隔 save_every 秒记一帧
    """
    F = 0.5 * (alpha[:-1] + alpha[1:])          # 界面热扩散率(长度 nx-1)
    s = dt / dx ** 2
    theta1 = 1.0 - theta
    # 逐行系数(与时间无关部分)
    a = np.zeros(nx); b = np.ones(nx); c = np.zeros(nx)
    for j in range(1, nx - 1):
        a[j] = -theta * s * F[j - 1]
        c[j] = -theta * s * F[j]
        b[j] = 1.0 + theta * s * (F[j - 1] + F[j])
    # 右边界：T[-1]=T[-2]（绝热）
    a[-1] = -1.0
    b[-1] = 1.0

    def flux_rhs(T):
        """显式部分贡献 r_j = s*[F_j(T_{j+1}-T_j) - F_{j-1}(T_j-T_{j-1})]"""
        r = np.zeros(nx)
        r[1:-1] = s * (F[1:] * (T[2:] - T[1:-1]) - F[:-1] * (T[1:-1] - T[:-2]))
        return r

    T = np.full(nx, T0)
    T[0] = T_hot
    times = [0.0]; snap = [T.copy()]
    n_step = int(np.ceil(t_total / dt))
    for k in range(1, n_step + 1):
        t_cur = k * dt
        rhs = T + theta1 * flux_rhs(T)          # 显式已知部分
        rhs[0] = T_hot                          # 左边界 Dirichlet
        rhs[-1] = 0.0                           # 右边界方程 T[-1]-T[-2]=0
        T = _thomas(a, b, c, rhs)
        T[0] = T_hot
        if t_cur >= len(times) * save_every + 1e-9 or k == n_step:
            times.append(t_cur); snap.append(T.copy())
    return np.array(times), np.array(snap)


def main():
    # 示例结构：仿热防护服的多层织物 + 空气隙（参数需按赛题附件/文献修改）
    layers = [
        {"k": 0.082, "rho": 300, "c": 1377, "thickness": 0.0006},   # 外层
        {"k": 0.037, "rho": 862, "c": 2100, "thickness": 0.0060},   # 隔热层
        {"k": 0.045, "rho": 74.2, "c": 1726, "thickness": 0.0036},  # 内层
        {"k": 0.028, "rho": 1.18, "c": 1005, "thickness": 0.0064},  # 空气隙
    ]
    x, dx, L, alpha = build_mesh(layers, nx=200)
    times, T = solve_heat(alpha, dx, len(alpha), T0=33.0, T_hot=75.0,
                          t_total=5400.0, dt=2.0, save_every=60)

    skin = T[:, -1]                             # 皮肤侧(右端)温度随时间
    print(f"总厚={L*1000:.1f} mm, nx={len(x)}, dx={dx*1000:.3f} mm")
    print(f"90min末 皮肤侧温度={skin[-1]:.2f}°C  期间最高={skin.max():.2f}°C")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(times / 60, skin, lw=2)
    ax1.set_xlabel("时间(min)"); ax1.set_ylabel("皮肤侧温度(°C)")
    ax1.set_title("皮肤侧温度随时间变化"); ax1.grid(alpha=0.3)
    for tt in [600, 1800, 3600, 5400]:
        i = np.argmin(abs(times - tt))
        ax2.plot(x * 1000, T[i], label=f"t={tt / 60:.0f}min")
    ax2.set_xlabel("位置(mm)"); ax2.set_ylabel("温度(°C)")
    ax2.set_title("不同时刻温度分布"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("热传导CN.png", dpi=150)
    print("已保存: 热传导CN.png")
    # 论文用法：与附件实测曲线对比算平均相对误差；再做层厚/环境温度 ±x% 的灵敏度与鲁棒性分析


if __name__ == "__main__":
    main()
