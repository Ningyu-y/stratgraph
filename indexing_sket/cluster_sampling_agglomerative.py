from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict
import numpy as np
import networkx as nx
from typing import Any, List, Callable
import random 
from graphrag.model import (
    TextUnit
)

        
# 辅助函数：根据阈值计算簇数量
def _get_n_clusters(embeddings: np.ndarray, threshold: float) -> int:
    """运行 AgglomerativeClustering 并返回簇的数量。"""
    if threshold < 0: # 避免负阈值
        return embeddings.shape[0] # 返回最大可能簇数 N
        
    model = AgglomerativeClustering(
        n_clusters=None, # 必须为 None 才能使用 distance_threshold
        metric='euclidean',
        linkage='ward',     # Ward 必须用 euclidean
        distance_threshold=threshold,
        compute_full_tree=True
    )
    # fit_predict 比 fit 快，可以直接获取标签
    model.fit(embeddings)
    print(f"AgglomerativeClustering 聚类完成，distance_threshold:{threshold},共 {model.n_clusters_} 个簇。")
    return model.n_clusters_


def cluster_based_sampling(
    text_units: list[TextUnit],
    sample_length: int,
    n_clusters: int, 
    knn_edges: int,
    build_knn_chunk_graph,
    nx_pagerank_func: Any
) -> list[TextUnit]:
    """
    基于层次聚类 (Agglomerative) 和 PageRank 的采样。
    【核心修改】：使用二分搜索自动调整 distance_threshold，使簇数 K <= 200。
    """
    
    print(f"\n--- 正在执行基于层次聚类 (Auto-Threshold Ward) 的采样，目标长度: {sample_length} ---")
    
    # A. 准备数据
    embeddings = np.array([t.text_embedding for t in text_units])
    N = len(text_units)
    
    # 归一化：Ward Linkage (Euclidean) 必须在归一化向量上运行才能表示余弦相似度
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10) 
    
    # ================= B. 自动搜索最佳距离阈值 (二分查找) =================
    
    TARGET_MAX_CLUSTERS = 200
    MAX_ITERATIONS = 15 # 15次迭代可保证相当高的精度 (2.0 / 2^15)
    
    # 设定搜索边界
    low_threshold = 1.0 # 已知簇数过多时的阈值 (下限)
    high_threshold = 2.0 # 已知簇数很少时的阈值 (上限)
    final_threshold = high_threshold # 默认值，最坏情况即 1 个簇
    
    print(f"正在使用二分搜索自动调整阈值，目标簇数 K <= {TARGET_MAX_CLUSTERS}...")
    
    # 提前检查：如果 1.0 已经满足条件（几乎不可能），则直接使用 1.0
    if _get_n_clusters(normalized_embeddings, 1.0) <= TARGET_MAX_CLUSTERS:
        final_threshold = 1.0
    
    else:
        for i in range(MAX_ITERATIONS):
            mid_threshold = (low_threshold + high_threshold) / 2
            
            # 提前终止：如果精度足够高，跳出
            if high_threshold - low_threshold < 0.001: 
                final_threshold = high_threshold
                break

            n_clusters_at_mid = _get_n_clusters(normalized_embeddings, mid_threshold)
    
            # 决策逻辑：
            if n_clusters_at_mid > TARGET_MAX_CLUSTERS:
                # 簇数太多 -> 阈值太小，需要增大阈值以允许更多合并
                low_threshold = mid_threshold
            else:
                # 簇数可接受 (K <= 200) -> 阈值可能太高，尝试减小以获得更多簇
                high_threshold = mid_threshold
                final_threshold = high_threshold # 记录当前最佳(最小)阈值
        
        # 循环结束，使用找到的最佳阈值
        final_threshold = high_threshold
        
    # ================= C. 使用最终确定的阈值进行聚类 =================
    
    final_model = AgglomerativeClustering(
        n_clusters=None,
        metric='euclidean',
        linkage='ward',
        distance_threshold=final_threshold,
        compute_full_tree=True
    )
    labels = final_model.fit_predict(normalized_embeddings)
    actual_n_clusters = final_model.n_clusters_
    
    print(f"层次聚类完成 (Threshold={final_threshold:.3f})，最终生成 {actual_n_clusters} 个簇。")

    # D. 计算全局 PageRank
    print("构建 KNN 图并计算 PageRank...")
    G = build_knn_chunk_graph(text_units, knn_edges, knn_edges) 
    pr = nx_pagerank_func(G, alpha=0.85) 
    
    # E. 将文本块按簇分组
    clusters = defaultdict(list)
    for idx, unit in enumerate(text_units):
        cluster_id = labels[idx]
        clusters[cluster_id].append(unit)
        
    # F. 簇内排序
    for cid in clusters:
        clusters[cid].sort(key=lambda x: pr.get(x.id, 0), reverse=True)

    # ================= G. 按比例分配采样预算 (保持不变) =================
    
    sampled_text_units = []
    
    # 1. 计算大小
    cluster_sizes = {cid: len(units) for cid, units in clusters.items()}
    total_units = sum(cluster_sizes.values())
    
    # 2. 初步分配
    cluster_budgets = {}
    current_allocated = 0
    
    for cid, size in cluster_sizes.items():
        ratio = size / total_units
        budget = int(sample_length * ratio)
        cluster_budgets[cid] = budget
        current_allocated += budget
        
    # 3. 处理剩余预算
    remaining_budget = sample_length - current_allocated
    
    if remaining_budget > 0:
        # print(f"  剩余 {remaining_budget} 个名额，分配给最大的簇...")
        sorted_clusters_by_size = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        for cid, _ in sorted_clusters_by_size:
            if remaining_budget <= 0: break
            if cluster_budgets[cid] < cluster_sizes[cid]:
                cluster_budgets[cid] += 1
                remaining_budget -= 1

    # 4. 执行采样
    for cid, budget in cluster_budgets.items():
        if budget > 0:
            selection = clusters[cid][:budget]
            sampled_text_units.extend(selection)

    print(f"已采样 {len(sampled_text_units)} 个文本块。")
            
    return sampled_text_units

