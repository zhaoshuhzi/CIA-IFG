import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from causallearn.search.ScoreBased import notears
from causallearn.utils.GraphUtils import to_nx_graph
from semopy import Model, semplot, report_fit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from statsmodels.tsa.stattools import grangercausalitytests
from causallearn.search.ConstraintBased.TiMINo import TiMINo
from causallearn.utils.GraphUtils import plot_graph

# ========== Step 1: 模拟 ERP 脑区活动数据 ==========

def simulate_data():
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        'pIFG': np.random.normal(0, 1, n),
        'aIFG': np.zeros(n),
        'STG': np.zeros(n),
        'SMA': np.random.normal(0, 1, n),
        'M1': np.zeros(n)
    })
    data['aIFG'] = 0.8 * data['pIFG'] + np.random.normal(0, 0.5, n)
    data['STG'] = 0.6 * data['pIFG'] + np.random.normal(0, 0.5, n)
    data['M1'] = 0.5 * data['aIFG'] + 0.4 * data['SMA'] + np.random.normal(0, 0.5, n)
    return data

# ========== Step 2: 使用 NOTEARS 构建因果 DAG 网络 ==========

def run_notears(data: pd.DataFrame):
    print("Running NOTEARS...")
    data_std = StandardScaler().fit_transform(data.values)
    W_est = notears.notears_linear(data_std, lambda1=0.1)

    labels = list(data.columns)
    G = nx.DiGraph()
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            weight = W_est[i, j]
            if abs(weight) > 0.05:
                G.add_edge(source, target, weight=round(weight, 3))

    # 可视化网络图
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("NOTEARS Causal DAG")
    plt.show()

    return G, W_est

# ========== Step 3: SEM 结构方程建模 ==========

def run_sem(data: pd.DataFrame):
    print("Running SEM...")
    model_desc = """
    aIFG ~ pIFG
    STG ~ pIFG
    M1 ~ aIFG + SMA
    """

    model = Model(model_desc)
    model.fit(data)

    print("\n== SEM Path Coefficients ==")
    print(report_fit(model))

    # 可视化 SEM 路径图
    fig = semplot(model, plot_covs=True)
    plt.title("SEM Path Model")
    plt.show()

    return model

# ========== Step 4: 反事实分析与验证 ==========

def run_counterfactual(data_active: pd.DataFrame, data_sham: pd.DataFrame):
    print("\n== Counterfactual Verification ==")

    # Euclidean distance
    dist = pairwise_distances(data_active, data_sham)
    print(f"Euclidean Distance (avg): {np.mean(dist):.4f}")

    # Granger causality test
    print("\nGranger Causality Tests:")
    for src in data_active.columns:
        for tgt in data_active.columns:
            if src != tgt:
                try:
                    res = grangercausalitytests(data_active[[src, tgt]], maxlag=2, verbose=False)
                    pvals = [res[i+1][0]['ssr_ftest'][1] for i in range(2)]
                    print(f"{src} -> {tgt}: p-values: {pvals}")
                except:
                    print(f"Failed Granger for {src} -> {tgt}")

    # TiMINo
    print("\nTiMINo DAG:")
    model = TiMINo()
    result = model.learn(data_active.values, method='multivariate')
    plot_graph(result['G'], labels=data_active.columns)

# ========== 主程序整合入口 ==========

def run_cia_pipeline():
    print("== Step 1: Simulate ERP Data ==")
    data = simulate_data()
    print(data.head())

    print("\n== Step 2: Causal Network via NOTEARS ==")
    G, weights = run_notears(data)

    print("\n== Step 3: SEM Model Verification ==")
    model = run_sem(data)

    print("\n== Step 4: Counterfactual Analysis ==")
    data_active = data + 0.3 * np.random.randn(*data.shape)  # 模拟真实刺激数据
    data_sham = data + 0.05 * np.random.randn(*data.shape)  # 模拟对照组
    run_counterfactual(data_active, data_sham)

    print("\n== CIA Framework Pipeline Finished ==")

# ========== 运行入口点 ==========

if __name__ == '__main__':
    run_cia_pipeline()
