from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import numpy as np
import networkx as nx
from typing import Any, List, Callable
import random 
from graphrag.model import (
    TextUnit
)

        
# ----------------- 您的 cluster_based_sampling 函数 (保持不变) -----------------
def cluster_based_sampling(
    text_units: list[TextUnit],
    sample_length: int,
    n_clusters: int, # n_clusters_max
    knn_edges: int,
    build_knn_chunk_graph,
    nx_pagerank_func: Any
) -> list[TextUnit]:

    
    print(f"\n--- 正在执行基于聚类的采样 (按比例分配)，目标长度: {sample_length} ---")
    
    # A. 准备数据进行聚类
    embeddings = np.array([t.text_embedding for t in text_units])
    N = len(text_units)
    
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10) 
    
    # ================= B. 自动确定最佳簇数 (使用轮廓系数) =================
    k_min = 2
    k_max = min(n_clusters, N - 1, 200) 
    
    if N <= 2 or k_max < k_min:
        actual_n_clusters = max(1, N)
        if actual_n_clusters <= 1: actual_n_clusters = 1
    else:
        silhouette_scores = {}
        print(f"正在搜索最佳簇数 (k={k_min} 到 {k_max})...")
        for k in range(k_min, k_max + 1):
            kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
            k_labels = kmeans_model.fit_predict(normalized_embeddings)
            score = silhouette_score(normalized_embeddings, k_labels)
            silhouette_scores[k] = score
        actual_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
        print(f"最佳簇数: {actual_n_clusters} (Score: {silhouette_scores[actual_n_clusters]:.4f})")

    # ================= C. 使用最佳簇数进行最终聚类 =================
    if actual_n_clusters > 1:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_embeddings)
    else:
        labels = np.zeros(N, dtype=int)

    # D. 计算全局 PageRank
    print("构建 KNN 图并计算 PageRank...")
    G = build_knn_chunk_graph(text_units, knn_edges, knn_edges) 
    pr = nx_pagerank_func(G, alpha=0.85) 
    
    # E. 将文本块按簇分组
    clusters = defaultdict(list)
    for idx, unit in enumerate(text_units):
        cluster_id = labels[idx]
        clusters[cluster_id].append(unit)

    # ================= G. 按比例分配采样预算=================
    
    sampled_text_units = []
    
    # 1. 计算每个簇的大小
    cluster_sizes = {cid: len(units) for cid, units in clusters.items()}
    total_units = sum(cluster_sizes.values())
    
    # 2. 初步分配预算：Budget_i = floor(Size_i / Total * Sample_Length)
    cluster_budgets = {}
    current_allocated = 0
    
    print("各簇大小及初步预算分配:")
    for cid, size in cluster_sizes.items():
        # 计算比例配额
        ratio = size / total_units
        budget = int(sample_length * ratio)
        cluster_budgets[cid] = budget
        current_allocated += budget
        print(f"  簇 {cid}: 大小 {size} ({ratio:.1%}) -> 分配 {budget}")
        
    # 3. 处理剩余预算 (由于取整可能导致分配不足)
    # 策略：将剩余名额优先分配给最大的簇，以强化核心话题的连贯性
    remaining_budget = sample_length - current_allocated
    
    if remaining_budget > 0:
        print(f"  剩余 {remaining_budget} 个名额，分配给最大的簇...")
        # 按簇大小降序排列
        sorted_clusters_by_size = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        for cid, _ in sorted_clusters_by_size:
            if remaining_budget <= 0:
                break
            # 确保不会超过该簇实际拥有的数量
            if cluster_budgets[cid] < cluster_sizes[cid]:
                cluster_budgets[cid] += 1
                remaining_budget -= 1
                print(f"  -> 簇 {cid} 增加 1 个名额")

    # 4. 在每个簇中进行随机采样
    for cid, budget in cluster_budgets.items():
        if budget > 0:
            # 从该簇中随机选择 budget 个文本单元
            available_units = clusters[cid]
            # 确保采样数量不超过簇中实际可用数量
            actual_budget = min(budget, len(available_units))
            random_selection = random.sample(available_units, actual_budget)
            sampled_text_units.extend(random_selection)

    # ======================================================================

    print(f"已采样 {len(sampled_text_units)} 个文本块。")
            
    return sampled_text_units

