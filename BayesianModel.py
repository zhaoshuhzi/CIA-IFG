import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

# ====== 模拟数据或导入脑区溯源数据 ======

def load_source_data():
    np.random.seed(42)
    n_samples = 500
    data = pd.DataFrame({
        'pIFG': np.random.normal(0, 1, n_samples),
        'aIFG': np.zeros(n_samples),
        'STG': np.zeros(n_samples),
        'SMA': np.random.normal(0, 1, n_samples),
        'M1': np.zeros(n_samples)
    })

    data['aIFG'] = 0.7 * data['pIFG'] + np.random.normal(0, 0.5, n_samples)
    data['STG'] = 0.4 * data['pIFG'] + 0.4 * data['SMA'] + np.random.normal(0, 0.5, n_samples)
    data['M1'] = 0.5 * data['aIFG'] + 0.3 * data['STG'] + np.random.normal(0, 0.5, n_samples)

    return data

# ====== 贝叶斯网络结构学习（DAG） ======

def learn_dag(data: pd.DataFrame):
    print("Learning DAG using HillClimbSearch + BIC...")

    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BicScore(data))

    print("Discovered DAG edges:")
    print(model.edges())

    return model

# ====== 可视化网络图（有向无环图） ======

def visualize_bayesian_network(model: BayesianModel):
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', arrows=True)
    plt.title("Bayesian Network - Directed Acyclic Graph (DAG)")
    plt.show()

# ====== 学习条件概率分布（可选） ======

def fit_cpds(model: BayesianModel, data: pd.DataFrame):
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
    for cpd in model.get_cpds():
        print(cpd)

# ====== 主运行函数 ======

def main():
    # Step 1: 加载数据（可替换为 EEG 溯源数据）
    data = load_source_data()

    # Step 2: 学习 DAG 结构
    model = learn_dag(data)

    # Step 3: 可视化 DAG
    visualize_bayesian_network(model)

    # Step 4: 拟合条件概率分布（可选）
    fit_cpds(model, data)

if __name__ == '__main__':
    main()
