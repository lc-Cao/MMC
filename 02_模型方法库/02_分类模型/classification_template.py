# -*- coding: utf-8 -*-
"""
分类模型通用模板：预处理 + 多模型对比 + 评估 + 特征重要性
依据：讲义第 3 章《分类模型》（案例：信用卡欺诈检测）

用法：
    1) 准备 pandas DataFrame：X 为特征、y 为标签列(0/1 或多类)；
    2) 把读数据那行换成你的数据(支持 Excel/CSV)；
    3) 运行后得到 各模型AUC对比表 + 分类报告 + 混淆矩阵 + Top特征。

依赖：numpy pandas scikit-learn matplotlib（画图时）
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import make_pipeline

# 中文字体（画图用，避免乱码）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_data(path=None):
    """示例：读入你的数据。返回 (特征X, 标签y, 特征名列表)。
    数据格式约定：最后一列是标签。可自行改成 y = df['label']。
    """
    if path is None:            # 无文件时生成一份演示数据
        rng = np.random.default_rng(42)
        n = 500
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=["f1", "f2", "f3", "f4", "f5"])
        y = (X["f1"] + 0.5 * X["f2"] + rng.normal(0, 1, n) > 0).astype(int)
        return X, y
    df = pd.read_excel(path) if path.endswith((".xls", ".xlsx")) else pd.read_csv(path)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return X, y


def evaluate(name, model, X_test, y_test, has_proba=True):
    """训练好的模型评估，返回指标字典（二分类；多分类时精确率/召回率/F1 取 macro）"""
    y_pred = model.predict(X_test)
    avg = "binary" if len(set(y_test)) == 2 else "macro"
    met = {
        "模型": name,
        "准确率": accuracy_score(y_test, y_pred),
        "精确率": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "召回率": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
    }
    if has_proba and len(set(y_test)) == 2:
        met["AUC"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return y_pred, met


def main():
    X, y = load_data()          # ← 换成你的数据路径
    print(f"样本数={len(X)}, 特征数={X.shape[1]}, 正类占比={y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)   # stratify 保持类别比例
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 逻辑回归 / SVM 用标准化后的数据；随机森林 / KNN 内部不敏感
    models = {
        "逻辑回归": LogisticRegression(max_iter=1000).fit(X_train_s, y_train),
        "SVM": SVC(probability=True, random_state=42).fit(X_train_s, y_train),
        "随机森林": RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train),
        "KNN": KNeighborsClassifier().fit(X_train_s, y_train),
    }

    # 训练/评估用同一套预处理：逻辑回归、SVM、KNN 用标准化数据，随机森林用原数据
    eval_set = {"逻辑回归": X_test_s, "SVM": X_test_s, "随机森林": X_test, "KNN": X_test_s}

    rows, best = [], None
    for name, model in models.items():
        _, met = evaluate(name, model, eval_set[name], y_test)
        rows.append(met)
        if best is None or met["AUC"] > best["AUC"]:
            best = met
            best_model = (name, model)
    result = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print("\n===== 模型对比表(写入论文) =====")
    print(result.round(4).to_string(index=False))
    print("\n最佳模型:", best["模型"])

    # 用最佳模型输出分类报告 + 混淆矩阵
    name, model = best_model
    X_t = X_test_s if name != "随机森林" else X_test
    y_pred = model.predict(X_t)
    print("\n===== 分类报告 =====")
    print(classification_report(y_test, y_pred, zero_division=0))

    if name == "随机森林":
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("===== 特征重要性 Top5 =====")
        print(imp.head(5).round(4))

    # 画混淆矩阵（保存 PNG）
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["负类", "正类"]); ax.set_yticklabels(["负类", "正类"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    ax.set_title(f"{name} 混淆矩阵")
    plt.tight_layout(); plt.savefig("混淆矩阵.png", dpi=150)
    print("\n已保存: 混淆矩阵.png")


if __name__ == "__main__":
    main()
