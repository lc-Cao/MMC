# -*- coding: utf-8 -*-
"""
SIR 传染病模型模板：机理微分方程 + R0 + 峰值 + 传染率敏感性扫描
依据：讲义第 7.4 节《传染病传播预测（SIR 模型）》

方程： dS/dt = -β S I / N ;  dI/dt = β S I / N - γ I ;  dR/dt = γ I
R0 = β / γ。R0>1 疫情扩散，R0<1 自然消亡。感染峰值约在 S 降到 N/R0 时。

依赖：numpy scipy matplotlib
"""
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def sir_rhs(t, y, beta, gamma, N):
    S, I, R = y
    return [-beta * S * I / N,
            beta * S * I / N - gamma * I,
            gamma * I]


def run_sir(N=10000, I0=10, beta=0.3, gamma=0.1, t_max=200, n_out=1000):
    """求解并返回 (t, S, I, R, 关键指标 dict)"""
    t = np.linspace(0, t_max, n_out)
    sol = solve_ivp(sir_rhs, [0, t_max], [N - I0, I0, 0],
                    args=(beta, gamma, N), t_eval=t, method="RK45",
                    rtol=1e-6, atol=1e-8)
    S, I, R = sol.y
    peak_day = t[np.argmax(I)]
    result = {
        "R0": beta / gamma,
        "峰值感染者": I.max(),
        "峰值时间(天)": peak_day,
        "最终感染比例": R[-1] / N,
    }
    return t, S, I, R, result


def plot_sir(t, S, I, R, out="SIR曲线.png"):
    plt.figure(figsize=(6.5, 4))
    plt.plot(t, S, label="易感者 S")
    plt.plot(t, I, label="感染者 I", lw=2)
    plt.plot(t, R, label="康复者 R")
    plt.xlabel("天数"); plt.ylabel("人数"); plt.title("SIR 模型传播曲线")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def main():
    N, I0, beta, gamma = 10000, 10, 0.3, 0.1
    t, S, I, R, res = run_sir(N, I0, beta, gamma)
    print("=== 基线 SIR 关键指标 ===")
    for k, v in res.items():
        if k == "R0":
            print(f"R0 = {v:.2f}")
        elif k == "最终感染比例":
            print(f"最终感染比例 = {v:.1%}")
        else:
            print(f"{k} = {v:.0f}")
    plot_sir(t, S, I, R)

    # ===== 参数敏感性(必做)：扫 β，观察峰值/峰值时间/最终感染 =====
    print("\n=== 敏感性：不同传染率 β ===")
    print(f"{'β':<8}{'R0':<6}{'峰值人数':<10}{'峰值时间':<10}{'最终感染':<8}")
    rows = []
    for b in [0.15, 0.30, 0.50]:
        _, _, _, _, r = run_sir(N, I0, beta=b, gamma=gamma)
        rows.append((b, r))
        print(f"{b:<8}{r['R0']:<6.1f}{r['峰值感染者']:<10.0f}"
              f"{r['峰值时间(天)']:<10.0f}{r['最终感染比例']:<8.1%}")
    # 论文结论写法(用上面打印的真实数字)：
    # “将传染率从 β0.50 降至 β0.15（戴口罩+隔离），峰值感染者减少 XX%、
    #  峰值时间推迟 XX 天——为医疗系统争取缓冲期。” 自己代入实际计算值。


if __name__ == "__main__":
    main()
