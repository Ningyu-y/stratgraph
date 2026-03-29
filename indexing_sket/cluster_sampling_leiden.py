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

try:
    from graspologic.partition import leiden
    HAS_GRASPOLOGIC = True
except ImportError:
    HAS_GRASPOLOGIC = False
    # 尝试导入 networkx 的 community 模块
    import networkx.algorithms.community as nx_comm

def cluster_based_sampling(
    text_units: list[TextUnit],
    sample_length: int,
    n_clusters: int, # Leiden 不需要指定簇数，此参数主要用于兼容接口或作为分辨率参考
    knn_edges: int,
    build_knn_chunk_graph,
    nx_pagerank_func: Any
) -> list[TextUnit]:

    
    print(f"\n--- 正在执行基于 Leiden 图社区的采样，目标长度: {sample_length} ---")
    
    N = len(text_units)
    if N == 0:
        return []

    # A. 构建 KNN 图 (Leiden 算法的基础)
    # 注意：K-Means 是先聚类再建图算PR，Leiden 是必须先有图
    print("构建 KNN 图...")
    G = build_knn_chunk_graph(text_units, knn_edges, knn_edges) 
    
    # ================= B. 执行 Leiden 社区发现 =================
    print("正在运行 Leiden 社区发现算法...")
    
    # 建立 id 到 index 的映射，方便后续处理
    id_to_unit = {unit.id: unit for unit in text_units}
    
    labels = {} # 存储 {node_id: cluster_id}
    
    if HAS_GRASPOLOGIC:
        # 使用 graspologic 的 Leiden 实现
        # resolution 参数控制粒度：值越大社区越小（越多），值越小社区越大（越少）
        # 默认 1.0 是一个很好的起点
        hierarchical_clusters = leiden(G, resolution=0.6) 
        
        # graspologic 返回的是 {node: cluster_id} 字典
        labels = hierarchical_clusters
        # print(f"labels:{labels}")
        
        # 统计社区数量
        unique_clusters = set(labels.values())
        actual_n_clusters = len(unique_clusters)
        print(f"Leiden (graspologic) 发现了 {actual_n_clusters} 个社区。")
        
    else:
        # 回退方案：使用 networkx 的 Louvain (Leiden 的前身，逻辑相似)
        print("未检测到 graspologic，回退使用 NetworkX 的 Louvain 算法...")
        try:
            # nx.community.louvain_communities 返回的是 list[set[nodes]]
            communities = nx_comm.louvain_communities(G, resolution=1.0, seed=42)
            actual_n_clusters = len(communities)
            
            for cid, nodes in enumerate(communities):
                for node in nodes:
                    labels[node] = cid
            print(f"Louvain (NetworkX) 发现了 {actual_n_clusters} 个社区。")
            
        except AttributeError:
             # 如果 networkx 版本过低没有 louvain
            print("NetworkX 版本不支持 Louvain，退化为连通分量 (Connected Components)...")
            components = list(nx.connected_components(G))
            actual_n_clusters = len(components)
            for cid, nodes in enumerate(components):
                for node in nodes:
                    labels[node] = cid

    # C. 计算全局 PageRank
    print("计算全局 PageRank...")
    pr = nx_pagerank_func(G, alpha=0.85) 
    
    # D. 将文本块按社区分组
    clusters = defaultdict(list)
    
    # 注意：图中的节点 ID 是 text_unit.id
    # 我们需要遍历 text_units 并从 labels 中查找对应的社区 ID
    for unit in text_units:
        # 处理可能得不到标签的孤立点（虽然 KNN 图通常连通，但为了鲁棒性）
        cluster_id = labels.get(unit.id, -1) 
        if cluster_id == -1:
            # 将孤立点单独归为一个虚拟社区或最大的社区
            # 这里简单起见，归为 -1 社区，稍后处理
            cluster_id = -1
        clusters[cluster_id].append(unit)

    # 打印社区信息
    for cid, units in clusters.items():
        if cid == -1:
            print(f"  [警告] 孤立节点数量: {len(units)}")
        # print(f"  社区 {cid}: {len(units)} 个节点")

    # E. 簇内排序：每个社区内部按 PageRank 分数降序排列
    for cid in clusters:
        clusters[cid].sort(key=lambda x: pr.get(x.id, 0), reverse=True)

    # ================= F. 按比例分配采样预算 =================
    
    sampled_text_units = []
    
    # 1. 计算大小
    cluster_sizes = {cid: len(units) for cid, units in clusters.items()}
    # 过滤掉空的或只有孤立点的簇（视情况而定，这里保留孤立点簇但权重可能很低）
    total_units = sum(cluster_sizes.values())
    
    # 2. 初步分配预算
    cluster_budgets = {}
    current_allocated = 0
    
    # print("各社区大小及初步预算分配:")
    for cid, size in cluster_sizes.items():
        ratio = size / total_units
        budget = int(sample_length * ratio)
        cluster_budgets[cid] = budget
        current_allocated += budget
        # print(f"  社区 {cid}: 大小 {size} ({ratio:.1%}) -> 分配 {budget}")
        
    # 3. 处理剩余预算 (优先分配给大社区)
    remaining_budget = sample_length - current_allocated
    
    if remaining_budget > 0:
        print(f"  剩余 {remaining_budget} 个名额，分配给最大的社区...")
        sorted_clusters_by_size = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        for cid, _ in sorted_clusters_by_size:
            if remaining_budget <= 0:
                break
            if cluster_budgets[cid] < cluster_sizes[cid]:
                cluster_budgets[cid] += 1
                remaining_budget -= 1
                # print(f"  -> 社区 {cid} 增加 1 个名额")

    # 4. 执行采样
    for cid, budget in cluster_budgets.items():
        if budget > 0:
            selection = clusters[cid][:budget]
            sampled_text_units.extend(selection)


    print(f"已采样 {len(sampled_text_units)} 个文本块。")
            
    return sampled_text_units
