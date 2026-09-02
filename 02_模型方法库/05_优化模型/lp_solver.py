# -*- coding: utf-8 -*-
"""
线性规划通用求解器：scipy.optimize.linprog(最小化) + 约束右端灵敏度分析
依据：讲义第 6 章《优化模型》（案例：生产计划，max Z=3x1+5x2）

标准形式(linprog 内部默认求最小，最大化请把目标系数取负)：
    min  c^T x
    s.t. A_ub x <= b_ub
         A_eq x == b_eq
         x >= 0  (bounds 指定)

依赖：numpy scipy  (pulp 等价写法见文件底部注释)
"""
import numpy as np
from scipy.optimize import linprog


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
             bounds=None, integrality=None, method="highs"):
    """求解线性/整数规划。
    c: 目标系数——linprog 求最小，所以“最大化”问题传入 -c；
    返回 x(最优解向量)。目标函数值在调用处按实际语义还原。
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, integrality=integrality, method=method)
    if not res.success:
        print("求解失败:", res.message)
        return None
    return res.x


def sensitivity(A_ub, b_ub, c, delta=0.2, var_names=None):
    """灵敏度(必做): 每个约束右端值 ±delta 重解，观察最优值变化。
    约定 c 为最小化系数(即最大化题已取负)；变化越大的约束 = 越“紧”的瓶颈。"""
    base_x = solve_lp(c, A_ub=A_ub, b_ub=b_ub)
    base_v = -(c @ base_x)                     # 还原为“最大化”语义
    print(f"\n[灵敏度] 基准最优值 = {base_v:.4f}, 解 x = {base_x.round(4)}")
    print(f"{'约束':<8}{'变化':<6}{'新最优值':<12}{'变化率':<10}")
    for i in range(len(b_ub)):
        for s in (1 + delta, 1 - delta):
            bb = b_ub.copy(); bb[i] *= s
            x = solve_lp(c, A_ub=A_ub, b_ub=bb)
            v = -(c @ x)
            chg = (v - base_v) / abs(base_v) * 100
            nm = var_names[i] if var_names else f"b{i}"
            print(f"{nm:<8}{(s - 1):+.0%}  {v:<12.4f}{chg:+.2f}%")


def main():
    # ===== 讲义案例：max Z = 3x1 + 5x2 =====
    #      s.t. 2x1 + x2 <= 100 ; x1 + 2x2 <= 80 ; x1,x2 >= 0
    c = np.array([-3.0, -5.0])            # 最大化 -> 目标系数取负
    A_ub = np.array([[2, 1], [1, 2]])
    b_ub = np.array([100.0, 80.0])
    bounds = [(0, None), (0, None)]

    x = solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    print("最优解 x1,x2 =", x.round(2), "  最大利润 =", round(-c @ x, 2))
    # 期望输出: x=[40,20], 利润=220 (讲义 6.4)

    sensitivity(A_ub, b_ub, c, var_names=["资源", "时间"])


# ============ 整数/0-1 规划写法(scipy>=1.9) ============
# from scipy.optimize import milp, LinearConstraint, Bounds
# res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)],
#            integrality=np.ones(len(c)), bounds=Bounds(0, np.inf))

# ============ pulp 等价写法(讲义 Listing 6，需 pip install pulp) ============
# import pulp
# prob = pulp.LpProblem("生产计划", pulp.LpMaximize)
# x1 = pulp.LpVariable('x1', lowBound=0); x2 = pulp.LpVariable('x2', lowBound=0)
# prob += 3*x1 + 5*x2
# prob += 2*x1 + x2 <= 100
# prob += x1 + 2*x2 <= 80
# prob.solve(pulp.PULP_CBC_CMD(msg=False))
# print(x1.value(), x2.value(), prob.objective.value())

if __name__ == "__main__":
    main()
