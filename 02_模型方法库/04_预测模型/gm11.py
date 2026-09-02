# -*- coding: utf-8 -*-
"""
灰色预测 GM(1,1)：数据极少(4-10 个点)且近似单调时的预测模板
依据：讲义第 5.4 节《产品销量预测与推导》（案例数据 125,138,...,205）

公式要点：
    x1 = cumsum(x0)                     一次累加(1-AGO)
    z1(k) = 0.5*(x1(k)+x1(k-1))         背景值(均值生成)
    [a,b]^T = (B^T B)^-1 B^T Y          最小二乘
    xhat0(k+1) = (x0(0)-b/a)(1-e^a) e^{-a k}   预测还原式
适用：数据 ≥4 点且级比检验通过；数据上下波动/非单调 → 改用回归或 ARIMA。
依赖：numpy
"""
import numpy as np


def gm11(x0, predict_n=3, verbose=True):
    """灰色预测。返回 (fitted, future, mape)"""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    if n < 4:
        raise ValueError("灰色预测至少需要 4 个数据点")

    # Step1: 级比检验(落在(0,2)认为适合；越接近1越好)
    ratio = x0[:-1] / x0[1:]
    ok = bool(np.all((ratio > 0) & (ratio < 2)))
    if verbose:
        print(f"[级比检验] 范围=[{ratio.min():.4f}, {ratio.max():.4f}] -> "
              f"{'通过' if ok else '不通过(谨慎使用/改其他模型)'}")

    # Step2: 1-AGO 累加
    x1 = np.cumsum(x0)
    # Step3: 构造 B 矩阵与 Y
    B = np.column_stack([-0.5 * (x1[:-1] + x1[1:]), np.ones(n - 1)])
    Y = x0[1:]
    # Step4: 最小二乘 [a, b]
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]
    if verbose:
        print(f"[参数] a={a:.4f}, b={b:.4f}  (a 为发展系数, |a| 越小精度越高)")

    # Step5: 时间响应函数还原为原始序列预测值
    def pred(k):                      # k=0 对应第 1 期
        return (x0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * k)

    fitted = np.array([pred(i) for i in range(n)])
    future = np.array([pred(i) for i in range(n, n + predict_n)])
    mape = float(np.mean(np.abs((x0 - fitted) / (x0 + 1e-9))) * 100)
    if verbose:
        print(f"[拟合] MAPE={mape:.2f}%")
        print(f"[预测] 未来 {predict_n} 期: {future.round(1)}")
    return fitted, future, mape


def main():
    # 讲义案例：近 6 期销量(单调递增) —— 换成你的历史序列
    x0 = [125, 138, 152, 168, 186, 205]
    gm11(x0, predict_n=3)


if __name__ == "__main__":
    main()
