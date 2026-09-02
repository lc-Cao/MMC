# -*- coding: utf-8 -*-
"""
聚类模型通用模板：K-means + 肘部法则/轮廓系数定 K + 聚类画像 + PCA 可视化
依据：讲义第 4 章《聚类模型》（案例：电商客户分群）

用法：
    1) 准备特征 DataFrame X（不需要标签）；
    2) 运行得到：SSE/轮廓系数选 K 的两张图、聚类结果、每簇画像、PCA 散点图；
    3) 论文里写清楚：为什么这个 K、每簇代表什么人、策略建议。

依赖：numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_data(path=None):
    """读取你的特征数据（演示数据：年消费额 + 购买频次，讲义 4.4 同款结构）"""
    if path is None:
        rng = np.random.default_rng(42)
        centers = [(20000, 30), (8000, 15), (2000, 5)]
        parts = []
        for c in centers:
            parts.append(rng.normal(loc=c, scale=(1800, 3.5), size=(100, 2)))
        df = pd.DataFrame(np.vstack(parts), columns=["年消费额(元)", "购买频次(次/年)"])
        return df.round(0)
    df = pd.read_excel(path) if path.endswith((".xls", ".xlsx")) else pd.read_csv(path)
    return df


def choose_k(Xs, k_range=(2, 10)):
    """肘部法则(SSE) + 轮廓系数 同时算，返回两个序列 + 建议 K"""
    sse, sil, k_list = [], [], []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        sse.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
        k_list.append(k)
    best_k = k_list[int(np.argmax(sil))]
    print(f"[定K] 肘部拐点与轮廓系数最大值建议 K={best_k}")
    return k_list, sse, sil, best_k


def plot_choose_k(k_list, sse, sil, out="选K图.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(k_list, sse, "o-", color="#2b6cb0")
    ax1.set_xlabel("K"); ax1.set_ylabel("SSE"); ax1.set_title("肘部法则")
    ax2.plot(k_list, sil, "o-", color="#c53030")
    ax2.set_xlabel("K"); ax2.set_ylabel("轮廓系数"); ax2.set_title("轮廓系数")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def cluster_profile(df, labels, k):
    """每个簇：人数 + 各特征均值/极值（画像），打印 + 返回 DataFrame 供论文用"""
    df = df.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster").agg(["mean", "min", "max"]).round(2)
    print("\n===== 各簇画像(均值/极值) =====")
    print(profile.to_string())
    print("\n人数:", df["cluster"].value_counts().sort_index().to_dict())
    return profile


def plot_pca_scatter(Xs, labels, out="聚类散点图.png"):
    xy = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for k in sorted(set(labels)):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=18, label=f"簇{k}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("聚类结果（PCA 降至 2 维）"); ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"已保存: {out}")


def main():
    df = load_data()                      # ← 换成你的特征数据
    Xs = StandardScaler().fit_transform(df)   # Step1: 标准化（必做！）

    k_list, sse, sil, best_k = choose_k(Xs, k_range=(2, 10))
    plot_choose_k(k_list, sse, sil)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Xs)   # Step3
    labels = km.labels_
    print(f"\n轮廓系数: {silhouette_score(Xs, labels):.3f}")

    cluster_profile(df, labels, best_k)   # Step4: 画像解释（论文核心）
    plot_pca_scatter(Xs, labels)

    # 雷达图需要(簇×特征)均值表，见 04_可视化与绘图/绘图代码模板.py 中 radar_chart 函数


# ============================================================
# 备选方案（讲义表9）：
# 1) 层次聚类（小数据 / 看树状图选K）：
#    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#    Z = linkage(Xs, method="ward");  labels = fcluster(Z, t=best_k, criterion="maxclust")
# 2) DBSCAN（非球形 / 有噪声，自动定簇数）：
#    from sklearn.cluster import DBSCAN
#    labels = DBSCAN(eps=0.6, min_samples=10).fit_predict(Xs)   # -1 为噪声
# 注意：eps/min_samples 要按数据量调，参数要写进论文并说明取值依据
# ============================================================

if __name__ == "__main__":
    main()
