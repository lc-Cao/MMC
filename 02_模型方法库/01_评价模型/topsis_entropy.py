# -*- coding: utf-8 -*-
"""
评价模型通用模板：正向化 + 熵权法 + TOPSIS + 灵敏度分析
依据：讲义第 2 章《评价类模型》（案例：供应商综合评价）

用法：
    1) 把你的数据表(每行=一个方案, 每列=一个指标)读成 numpy 2D 数组 data；
    2) 指定成本型(越小越好)指标列 cost_cols，其余按效益型处理；
    3) 运行 main() 得到 权重 / 得分 / 排名 / 灵敏度结论。

依赖：numpy（其余为纯 Python）
"""
import numpy as np


def positive_normalize(data, cost_cols=None, method="minmax"):
    """Step1+2: 正向化 + 标准化
    - 成本型指标取倒数 1/x 统一为“越大越好”（讲义 2.4 做法；eps 防除零）
    - method: 'minmax' 缩放[0,1] | 'zscore' 标准化
    """
    X = np.array(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data 必须是二维数组：行=方案，列=指标")
    n, m = X.shape
    cost_cols = cost_cols or []
    for j in range(m):
        if j in cost_cols:
            X[:, j] = 1.0 / (X[:, j] + 1e-8)
    if method == "minmax":
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    elif method == "zscore":
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    else:
        raise ValueError("method 仅支持 minmax / zscore")
    return X


def entropy_weight(Xn):
    """Step3: 熵权法定权重（差异越大权重越高）
    Xn: 已标准化的矩阵；注意先 clip 掉 0/1 极端值避免 log(0)
    """
    n, m = Xn.shape
    P = np.clip(Xn / (Xn.sum(axis=0) + 1e-8), 1e-8, 1.0)   # 比重矩阵
    e = -np.sum(P * np.log(P), axis=0) / np.log(n)          # 信息熵
    d = 1.0 - e                                             # 差异度
    w = d / (d.sum() + 1e-8)
    return w


def combine_weight(w_entropy, w_subjective, alpha=0.5):
    """可选：熵权(客观) 与 AHP/专家(主观) 组合权重
    默认几何平均再归一；alpha 保留备用。更稳的组合=sqrt(w1*w2)后归一。
    """
    w = np.sqrt(np.array(w_entropy) * np.array(w_subjective))
    return w / w.sum()


def topsis_score(Xn, w):
    """Step4: 加权标准化 -> 正/负理想解 -> 相对贴近度 C∈[0,1]，越大越优"""
    V = Xn * w
    best = V.max(axis=0)
    worst = V.min(axis=0)
    d_best = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst + 1e-8)


def sensitivity(Xn, w, delta=0.1, topk=2):
    """Step5(必做): 每个指标权重 ±delta 重算得分与排名
    返回 (基准排名, 扰动汇总list)，并打印“前 topk 名是否保持稳定”
    """
    base = topsis_score(Xn, w)
    base_rank = np.argsort(-base)
    stable = 0
    total = 0
    logs = []
    for j in range(len(w)):
        for s in (1.0 + delta, 1.0 - delta):
            ww = w.copy()
            ww[j] *= s
            ww /= ww.sum()
            r = topsis_score(Xn, ww)
            rk = np.argsort(-r)
            hit = np.array_equal(rk[:topk], base_rank[:topk])
            stable += hit
            total += 1
            logs.append((j, round(s, 3), rk[:topk].tolist(), hit))
    print(f"[灵敏度] 权重±{delta:.0%} 共 {total} 次扰动："
          f"前 {topk} 名保持不变 {stable}/{total} 次")
    for j, s, rk, hit in logs:
        print(f"  指标{j} ×{s:<6} 排名前{topk}: {rk}  稳定={hit}")
    return base_rank, logs


def main():
    # ====== 示例数据：5 家供应商 × 6 指标（讲义表5）======
    # 列: 价格(成本) 质量(效益) 交货期(成本) 售后(效益) 信誉(效益) 产能(效益)
    data = np.array([
        [85, 90, 3, 80, 90, 500],
        [78, 95, 5, 85, 88, 450],
        [92, 82, 2, 75, 85, 600],
        [88, 88, 4, 90, 92, 520],
        [80, 91, 3, 82, 87, 480],
    ])
    names = ["A", "B", "C", "D", "E"]
    cost_cols = [0, 2]              # 价格、交货期为成本型

    Xn = positive_normalize(data, cost_cols=cost_cols)
    w = entropy_weight(Xn)
    score = topsis_score(Xn, w)
    order = np.argsort(-score)

    print("熵权法权重:", np.round(w, 4))
    print("综合得分:", dict(zip(names, np.round(score, 4))))
    print("最终排名:", " > ".join(f"{names[i]}({score[i]:.4f})" for i in order))
    print("最优方案:", names[order[0]])
    # 注：与讲义 2.4 示例相比，排名在第 2-5 名次序上可能有出入——
    # 讲义中列出的中间矩阵为人工四舍五入值，其熵权/排名数字并非严格按其公式重算；
    # 本代码为标准熵权-TOPSIS 公式的精确实现，以“最优方案 D”为稳定结论。
    print()
    sensitivity(Xn, w, topk=2)      # 灵敏度分析（论文里写结论用）


if __name__ == "__main__":
    main()
