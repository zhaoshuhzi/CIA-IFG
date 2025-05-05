import numpy as np
import pandas as pd
from semopy import Model, report_fit, semplot
import matplotlib.pyplot as plt

# Step 1: 模拟数据（stim: 0=sham, 1=active）
def generate_data(n=300):
    np.random.seed(123)

    stim = np.random.binomial(1, 0.5, size=n)  # 0: sham, 1: active

    # 模拟路径：stim → pIFG → aIFG，同时 stim → aIFG 有直接影响
    pIFG = 0.8 * stim + np.random.normal(0, 1, n)
    aIFG = 0.6 * pIFG + 0.5 * stim + np.random.normal(0, 1, n)

    return pd.DataFrame({'stim': stim, 'pIFG': pIFG, 'aIFG': aIFG})

# Step 2: 定义 SEM 模型（带中介、协方差、直接路径）
model_description = """
# 路径结构
pIFG ~ stim
aIFG ~ pIFG + stim

# 协方差（表示潜在干扰）
pIFG ~~ aIFG
"""

# Step 3: 拟合 SEM 模型
def fit_sem_model(data, description):
    model = Model(description)
    model.fit(data)

    print("\n📊 SEM 模型拟合结果：")
    print(report_fit(model))

    # 可视化结构图
    fig = semplot(model, plot_ests=True)
    plt.title("Stim → pIFG → aIFG + Direct/Residual Paths")
    plt.show()

    return model

# Step 4: 主程序入口
def main():
    print("🚀 生成 stim/pIFG/aIFG 模拟数据...")
    data = generate_data()

    print("📈 拟合结构方程模型...")
    fit_sem_model(data, model_description)

if __name__ == '__main__':
    main()
