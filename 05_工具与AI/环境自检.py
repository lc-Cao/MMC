# -*- coding: utf-8 -*-
"""
国赛 Python 环境自检：逐个检查建模常用库是否可用。
运行：python 环境自检.py
期望：最后一行输出「自检完成: 全部通过」；若有 [缺失]，按 05_工具与AI/requirements.txt 补装。
"""
import importlib
import sys

CHECKS = [
    "numpy", "scipy", "pandas", "sklearn", "statsmodels",
    "matplotlib", "seaborn", "pulp", "openpyxl",
]

def main():
    print(f"Python {sys.version.split()[0]}")
    ok = True
    for name in CHECKS:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"[OK]   {name:<14} {ver}")
        except Exception as e:
            ok = False
            print(f"[缺失] {name:<14} -> {e}")

    # 中文字体可用性提示（画图必需）
    try:
        import matplotlib.font_manager as fm
        fonts = {f.name for f in fm.fontManager.ttflist}
        have = fonts & {"SimHei", "Microsoft YaHei", "SimSun", "KaiTi"}
        if have:
            print(f"[OK]   中文字体可用: {sorted(have)}")
        else:
            print("[警告] 未找到 SimHei/微软雅黑 等中文字体，画中文图会乱码")
    except Exception:
        pass

    print("\n自检完成:", "全部通过 ✔" if ok else "存在缺失，请 pip install -r requirements.txt")


if __name__ == "__main__":
    main()
