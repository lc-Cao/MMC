# -*- coding: utf-8 -*-
"""
国赛绘图代码模板：中文字体初始化 + 常用图函数（复制即用）
依据：讲义 9.5/12.4。所有函数都 save PNG(dpi=150)，名称见名知义。

用法：要哪种图就拷哪个函数 + 改数据；或运行 main() 看一遍所有示例图。
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 0. 中文字体初始化（每个画图脚本第一件事） ----------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]   # 有黑体就黑体
plt.rcParams["axes.unicode_minus"] = False                        # 负号正常显示
# 若上面字体不存在，可尝试：["SimSun"]、"DengXian"、"KaiTi"


def line_chart(x, series, labels=None, xlabel="", ylabel="", title="",
               out="折线图.png", colors=None):
    """折线图：预测vs真实、SIR曲线、收敛曲线等"""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = colors or ["#2b6cb0", "#c53030", "#2f855a", "#b7791f"]
    series = np.atleast_2d(series)
    for i, y in enumerate(series):
        ax.plot(x, y, lw=2, marker="o", ms=3,
                label=labels[i] if labels else f"系列{i+1}", color=colors[i % len(colors)])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if labels:
        ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def bar_chart(cats, values, xlabel="", ylabel="", title="", out="柱状图.png",
              highlight_idx=None):
    """柱状图：指标权重、得分对比等；highlight_idx 高亮最优"""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#2b6cb0"] * len(cats)
    if highlight_idx is not None:
        colors[highlight_idx] = "#c53030"
    ax.bar(cats, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def scatter_chart(x, y, xlabel="", ylabel="", title="", out="散点图.png",
                  c=None, cmap="viridis"):
    """散点图：真实vs预测、PCA聚类散点等。c 传颜色数组按簇着色"""
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    sc = ax.scatter(x, y, s=18, c=c, cmap=cmap, alpha=0.85)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if c is not None:
        plt.colorbar(sc)
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def box_chart(data, labels=None, xlabel="", ylabel="", title="", out="箱线图.png"):
    """箱线图：多组数据分布对比（异常值、离群点）"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def heatmap(matrix, row_labels=None, col_labels=None, title="", out="热力图.png",
            cmap="Reds", annot=True):
    """相关性/混淆矩阵热力图（纯 matplotlib，无需 seaborn）"""
    M = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4.5, M.shape[1] * 0.9),
                                    max(4, M.shape[0] * 0.8)))
    im = ax.imshow(M, cmap=cmap)
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(col_labels or range(M.shape[1]))
    ax.set_yticklabels(row_labels or range(M.shape[0]))
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if M[i, j] > M.mean() + M.std() else "black")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def radar_chart(categories, data_list, labels, title="", out="雷达图.png"):
    """雷达图：多对象在若干维度的对比（聚类画像、供应商多指标对比）"""
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    colors = ["#2b6cb0", "#c53030", "#2f855a"]
    for d, lab, c in zip(data_list, labels, colors):
        vals = list(d) + [d[0]]
        ax.plot(angles, vals, "o-", label=lab, color=c, lw=1.8)
        ax.fill(angles, vals, alpha=0.12, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, pad=18); ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def main():
    """demo：跑一遍所有示例图（数据纯演示）"""
    x = np.arange(1, 13)
    y1 = 20 + 3 * x + 2 * np.sin(x)
    y2 = 18 + 2.5 * x + 2 * np.cos(x)
    line_chart(x, [y1, y2], labels=["方案A", "方案B"], xlabel="月份",
               ylabel="数值", title="趋势对比", out="示例_折线图.png")

    bar_chart(["质量", "价格", "交货期", "售后"], [0.28, 0.31, 0.22, 0.19],
              title="指标权重", out="示例_柱状图.png", highlight_idx=1)

    heatmap(np.random.default_rng(1).random((4, 4)).round(3),
            row_labels=["f1", "f2", "f3", "f4"], col_labels=["f1", "f2", "f3", "f4"],
            title="相关性热力图", out="示例_热力图.png")

    radar_chart(["价格", "质量", "交货期", "售后", "信誉"],
                [[0.5, 0.9, 0.4, 0.7, 0.8], [0.8, 0.5, 0.9, 0.4, 0.6]],
                ["供应商A", "供应商B"], title="供应商对比", out="示例_雷达图.png")
    print("\n所有示例图已保存到当前目录")


if __name__ == "__main__":
    main()
