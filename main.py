import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from semopy import Model, Optimizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import pairwise_distances
from statsmodels.tsa.stattools import grangercausalitytests

from causallearn.search.ScoreBased import notears
from causallearn.utils.GraphUtils import to_nx_graph
from causallearn.search.ConstraintBased.TiMINo import TiMINo
from causallearn.utils.GraphUtils import plot_graph

# ========== EEG 数据读取与预处理 ==========

def load_and_preprocess_eeg(fif_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    raw.filter(1., 40.)
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.5, baseline=(None, 0), preload=True)
    evoked = epochs.average()
    evoked.plot(spatial_colors=True)
    return evoked

# ========== 使用 NOTEARS 构建 DAG 贝叶斯网络 ==========

def build_dag_notears(data: pd.DataFrame):
    data_matrix = data.values
    W_est = notears.notears_linear(data_matrix, lambda1=0.1)
    dag = to_nx_graph(W_est, labels=data.columns)
    nx.draw(dag, with_labels=True)
    plt.title("DAG via NOTEARS")
    plt.show()
    return dag, W_est

# ========== SEM 因果建模与验证 ==========

def run_sem_analysis(data: pd.DataFrame, model_description: str):
    model = Model(model_description)
    opt = Optimizer(model)
    opt.optimize(data)
    print("SEM Model Fit Summary:")
    print(model.inspect())
    return model

# ========== 反事实验证分析 ==========

def run_counterfactual(data_active: pd.DataFrame, data_sham: pd.DataFrame):
    print("== Euclidean Distance ==")
    distance = pairwise_distances(data_active, data_sham)
    print(f"Avg Euclidean Distance: {np.mean(distance):.4f}")

    print("\n== Granger Causality ==")
    for i, src in enumerate(data_active.columns):
        for j, tgt in enumerate(data_active.columns):
            if i != j:
                print(f"Testing: {src} -> {tgt}")
                try:
                    result = grangercausalitytests(data_active[[src, tgt]], maxlag=2, verbose=False)
                    p_values = [round(result[i+1][0]['ssr_ftest'][1], 4) for i in range(2)]
                    print(f"p-values: {p_values}")
                except:
                    print("Error in Granger test")

    print("\n== TiMINo Analysis ==")
    model = TiMINo()
    result = model.learn(data_active.values, method='multivariate')
    plot_graph(result['G'], labels=data_active.columns)

# ========== 贝叶斯网络（pgmpy）辅助构建器 ==========

def build_bayesian_network_pgmpy(data: pd.DataFrame):
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BicScore(data))
    print("Bayesian Network structure:", model.edges())
    return model

# ========== 主执行函数 ==========

def run_cia_pipeline():
    print("== Step 1: Loading or Simulating Data ==")
    # 模拟脑区数据
    np.random.seed(0)
    n_samples = 300
    columns = ['aIFG', 'pIFG', 'STG', 'SMA', 'M1']
    data = pd.DataFrame(np.random.randn(n_samples, len(columns)), columns=columns)

    print("== Step 2: DAG Construction with NOTEARS ==")
    dag, weights = build_dag_notears(data)

    print("== Step 3: SEM Model ==")
    model_desc = """
    aIFG ~ pIFG + STG
    STG ~ SMA
    M1 ~ aIFG + STG
    """
    sem_model = run_sem_analysis(data, model_desc)

    print("== Step 4: Bayesian Network ==")
    bayesian_model = build_bayesian_network_pgmpy(data)

    print("== Step 5: Counterfactual Verification ==")
    # 模拟处理组与对照组（sham）
    data_active = data + 0.3 * np.random.randn(*data.shape)
    data_sham = data + 0.05 * np.random.randn(*data.shape)
    run_counterfactual(data_active, data_sham)

    print("== CIA Pipeline Finished ==")

if __name__ == '__main__':
    run_cia_pipeline()
