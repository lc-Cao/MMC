# -*- coding: utf-8 -*-
"""
预测模型模板二：线性回归(多特征) + ARIMA(纯时序) 快速上手
依据：讲义第 5 章《预测模型》

用法(两个独立场景，按需用)：
    场景A 多特征预测目标:   回归(Ridge/Lasso/随机森林均可) + R2/MSE + 画真实vs预测
    场景B 纯时间序列:       ADF 平稳性检查 -> 差分 -> ARIMA(p,d,q) -> 预测+区间
依赖：numpy pandas scikit-learn statsmodels matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 场景A：多特征 -> 目标 =====================
def regression_demo():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.uniform(0, 10, n); x2 = rng.normal(50, 10, n)
    y = 3 * x1 + 0.5 * x2 + rng.normal(0, 3, n)      # 线性关系（演示）
    X = pd.DataFrame({"x1": x1, "x2": x2})

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0).fit(X_tr, y_tr)          # 可换 LinearRegression/随机森林
    pred = model.predict(X_te)
    print("===== 回归评估 =====")
    print(f"R2={r2_score(y_te, pred):.3f}  "
          f"MSE={mean_squared_error(y_te, pred):.3f}  "
          f"MAE={mean_absolute_error(y_te, pred):.3f}")
    # 画真实 vs 预测（论文必备）
    plt.figure(figsize=(5, 4))
    plt.scatter(y_te, pred, s=16, alpha=0.6)
    lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("真实值"); plt.ylabel("预测值"); plt.title("回归：真实 vs 预测")
    plt.tight_layout(); plt.savefig("回归预测对比.png", dpi=150)
    print("已保存: 回归预测对比.png")
    return model


# ===================== 场景B：纯时间序列 ARIMA =====================
def arima_demo():
    try:
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("[跳过] 未安装 statsmodels，请先 pip install statsmodels")
        return

    # 演示数据：带趋势的月度序列（约48点；实际替换为你的序列）
    rng = np.random.default_rng(1)
    t = np.arange(48)
    data = pd.Series(20 + 1.2 * t + 4 * np.sin(t / 4) + rng.normal(0, 2, 48))

    # 1) 平稳性检查：p 值 <0.05 视为平稳，否则差分 d 加 1
    p = adfuller(data.dropna())[1]
    d = 0
    print(f"\n[ADF] p={p:.4f} -> {'平稳' if p < 0.05 else '非平稳, 需差分'}")
    if p >= 0.05:
        d = 1

    # 2) 建模：p,q 一般取小值(0-2)；p,q 由 ACF/PACF 定阶，赛中可用 AIC 遍历选优
    #    简单起见先用 order=(1,d,1)，可再跑 AIC 网格搜索
    model = ARIMA(data, order=(1, d, 1))
    fit = model.fit()
    print(f"[ARIMA] AIC={fit.aic:.1f}")

    # 3) 预测 6 期 + 95% 置信区间
    steps = 6
    fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    print(f"[预测] {np.round(mean.values, 1)}")
    print(f"[95%区间] {np.round(ci.values, 1).tolist()}")   # 论文里写区间是加分项

    # 4) 画图：历史 + 预测 + 区间
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(data.index, data, label="历史值")
    f_idx = np.arange(len(data), len(data) + steps)
    ax.plot(f_idx, mean, "o-", color="#c53030", label="预测值")
    ax.fill_between(f_idx, ci.iloc[:, 0], ci.iloc[:, 1], color="#c53030", alpha=0.15,
                    label="95% 置信区间")
    ax.set_xlabel("时间"); ax.set_ylabel("数值"); ax.set_title("ARIMA 预测")
    ax.legend(); plt.tight_layout(); plt.savefig("ARIMA预测.png", dpi=150)
    print("已保存: ARIMA预测.png")
    # 注：定阶可用循环 for p,q in product(range(3),range(3)) 比较 AIC 取最小
    return fit


if __name__ == "__main__":
    regression_demo()   # 场景A：多特征回归
    arima_demo()        # 场景B：纯时序 ARIMA
