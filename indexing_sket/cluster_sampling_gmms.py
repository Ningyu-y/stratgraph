from sklearn.mixture import GaussianMixture
from collections import defaultdict
import numpy as np
import networkx as nx
from typing import Any, List, Callable
import random 
from graphrag.model import (
    TextUnit
)

def cluster_based_sampling(
    text_units: list[Any],
    sample_length: int,
    n_clusters: int, 
    knn_edges: int,
    build_knn_chunk_graph,
    nx_pagerank_func: Any
) -> list[Any]:
    
    print(f"\n--- 执行 GMM 采样 (平方根平滑策略) ---")
    
    # A. 准备数据
    embeddings = np.array([t.text_embedding for t in text_units])
    N = len(text_units)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10) 
    
    # B. 确定簇数 (保持强制细粒度，防止 BIC 导致簇太少)
    optimal_k = int(np.sqrt(N) * 2.0) 
    optimal_k = max(2, min(optimal_k, n_clusters, N // 2))
    
    print(f"设定簇数 K={optimal_k}")

    # C. 训练 GMM
    # 稍微调高 reg_covar 保证数值稳定
    gmm = GaussianMixture(
        n_components=optimal_k, 
        covariance_type='diag', 
        random_state=42, 
        reg_covar=1e-4 
    )
    gmm.fit(normalized_embeddings)
    probs = gmm.predict_proba(normalized_embeddings) # (N, K)

    # D. PageRank
    G = build_knn_chunk_graph(text_units, knn_edges, knn_edges) 
    pr_scores = nx_pagerank_func(G, alpha=0.85) 
    
    # ================= G. 核心修改：平方根平滑分配预算 =================
    
    sampled_text_units = []
    selected_indices = set()
    
    # 1. 计算软大小 (Soft Size)
    soft_sizes = np.sum(probs, axis=0) # 每个簇的概率之和
    
    # 2. 应用平方根平滑 (Square Root Smoothing)
    # 这一步是关键：大数值被压缩，小数值被提升，但仍保持相对大小关系
    smoothed_sizes = np.sqrt(soft_sizes)
    total_smoothed_size = np.sum(smoothed_sizes)
    
    # 3. 计算预算
    cluster_budgets = {}
    current_allocated = 0
    
    print("预算分配预览 (Top 10):")
    # 为了防止某些极小簇分到 0，我们可以设一个 min_budget，或者完全依赖平滑结果
    for k in range(optimal_k):
        # 比例 = 该簇平滑大小 / 总平滑大小
        ratio = smoothed_sizes[k] / total_smoothed_size
        budget = int(sample_length * ratio)
        
        # 可选：强制最小分配 1 个，保证多样性 (除非预算极度紧缺)
        if budget < 1 and sample_length > optimal_k:
            budget = 1
            
        cluster_budgets[k] = budget
        current_allocated += budget
        
        if k < 10: 
             print(f"  簇 {k}: 原始大小 {soft_sizes[k]:.1f} -> 平滑大小 {smoothed_sizes[k]:.1f} -> 分配 {budget}")

    # 4. 剩余预算分配给平滑大小最大的簇
    remaining = sample_length - current_allocated
    if remaining > 0:
        sorted_indices = np.argsort(smoothed_sizes)[::-1]
        for i in range(remaining):
            idx = sorted_indices[i % optimal_k]
            cluster_budgets[idx] += 1
            
    # ================= H. 执行采样 =================
    
    for k in range(optimal_k):
        budget = cluster_budgets[k]
        if budget <= 0: continue
        
        candidates = []
        for idx, unit in enumerate(text_units):
            if unit.id in selected_indices: continue
            
            p_k = probs[idx, k]
            # 获取 PR，带有默认值兜底
            pr = pr_scores.get(str(unit.id), pr_scores.get(unit.id, 0.0))
            if pr <= 0: pr = 0.0001
            
            # 混合打分: 概率 * PR
            score = p_k * pr
            candidates.append((unit, score))
            
        # 排序并选取
        candidates.sort(key=lambda x: x[1], reverse=True)
        selection = candidates[:budget]
        
        for unit, _ in selection:
            sampled_text_units.append(unit)
            selected_indices.add(unit.id)
            
    # I. 兜底补全
    if len(sampled_text_units) < sample_length:
        all_remaining = [u for u in text_units if u.id not in selected_indices]
        all_remaining.sort(key=lambda x: pr_scores.get(str(x.id), 0), reverse=True)
        needed = sample_length - len(sampled_text_units)
        sampled_text_units.extend(all_remaining[:needed])
        
    return sampled_text_units
