import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Step 1: 模拟数据 ======

def generate_data(n=300):
    np.random.seed(0)
    stim = np.random.binomial(1, 0.5, n)
    pIFG = 0.8 * stim + np.random.normal(0, 1, n)
    aIFG = 0.5 * pIFG + 0.5 * stim + np.random.normal(0, 1, n)

    return pd.DataFrame({'stim': stim, 'pIFG': pIFG, 'aIFG': aIFG})

# ====== Step 2: 构建因果预测模型 ======

def train_model(data, features, target):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(data[features], data[target])
    return model

# ====== Step 3: 构建反事实样本（调换 stim） ======

def generate_counterfactuals(data, feature='stim'):
    cf_data = data.copy()
    cf_data[feature] = 1 - cf_data[feature]
    return cf_data

# ====== Step 4: 比较预测值，估计反事实效应 ======

def estimate_counterfactual_effects(model, factual, counterfactual, features, target):
    y_factual = model.predict(factual[features])
    y_counterfactual = model.predict(counterfactual[features])
    delta = y_counterfactual - y_factual

    factual[target + '_pred'] = y_factual
    factual['cf_' + target + '_pred'] = y_counterfactual
    factual['delta'] = delta

    return factual

# ====== Step 5: 可视化结果 ======

def plot_effects(df, target):
    sns.histplot(df['delta'], kde=True)
    plt.title(f"Estimated Individual Treatment Effects on {target}")
    plt.xlabel("Counterfactual Effect (CF - Actual)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    print(f"\n📊 Average Treatment Effect (ATE): {np.mean(df['delta']):.4f}")

# ====== 主程序入口 ======

def main():
    print("✅ Step 1: Simulating data...")
    data = generate_data()

    print("✅ Step 2: Training model to predict aIFG from stim and pIFG...")
    model = train_model(data, ['stim', 'pIFG'], 'aIFG')

    print("✅ Step 3: Generating counterfactual samples...")
    cf_data = generate_counterfactuals(data)

    print("✅ Step 4: Estimating individual causal effects...")
    results = estimate_counterfactual_effects(model, data, cf_data, ['stim', 'pIFG'], 'aIFG')

    print("✅ Step 5: Visualizing estimated effects...")
    plot_effects(results, 'aIFG')

if __name__ == '__main__':
    main()
