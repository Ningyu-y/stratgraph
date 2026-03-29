import hdbscan
from collections import defaultdict
import numpy as np
import networkx as nx
from typing import Any, List, Callable
import random 
from graphrag.model import (
    TextUnit
)

        
def cluster_based_sampling(
      text_units: list[TextUnit],
      sample_length: int,
      n_clusters: int, # n_clusters_max (HDBSCAN 不使用此参数，但用于兼容接口)
      knn_edges: int,
      build_knn_chunk_graph,
      nx_pagerank_func: Any
) -> list[TextUnit]:
      
      
      print(f"\n--- 正在执行基于 HDBSCAN 聚类的采样 (按比例分配，将噪声视为补充)，目标长度: {sample_length} ---")
      
      # A. 准备数据进行聚类
      embeddings = np.array([t.text_embedding for t in text_units])
      N = len(text_units)
      
      # 归一化
      norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
      normalized_embeddings = embeddings / (norms + 1e-10) 
      
      # ================= B & C. 使用 HDBSCAN 进行聚类 =================
      
      # 默认 HDBSCAN 需要的最小样本量 (经验值)
      MIN_SAMPLES_FOR_HDBSCAN = 3 # 保持 3 以发现小簇

      print(f"正在使用 HDBSCAN 进行聚类 (最少簇大小: {MIN_SAMPLES_FOR_HDBSCAN})...")

      # 运行 HDBSCAN 模型
      clusterer = hdbscan.HDBSCAN(
          min_cluster_size=MIN_SAMPLES_FOR_HDBSCAN,
          min_samples=2, 
          metric='euclidean', 
          prediction_data=False 
      )

      labels = clusterer.fit_predict(normalized_embeddings)

      # HDBSCAN 返回的标签中 -1 是噪声
      unique_clusters = set(labels)
      actual_n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

      noise_count = np.sum(labels == -1)
      print(f"HDBSCAN 发现了 {actual_n_clusters} 个有效簇，{noise_count} 个噪声点 (ID: -1)。")

      # D. 计算全局 PageRank
      print("构建 KNN 图并计算 PageRank...")
      G = build_knn_chunk_graph(text_units, knn_edges, knn_edges) 
      pr = nx_pagerank_func(G, alpha=0.85) 
      
      # E. 将文本块按簇分组
      clusters = defaultdict(list)
      for idx, unit in enumerate(text_units):
            cluster_id = labels[idx]
            clusters[cluster_id].append(unit)
            
      # F. 簇内排序：每个簇内部按 PageRank 分数降序排列
      for cid in clusters:
            clusters[cid].sort(key=lambda x: pr.get(x.id, 0), reverse=True)

      # ================= G. 按比例分配采样预算 (修正逻辑) =================
      
      sampled_text_units = []
      
      # 1. 计算每个簇的大小
      cluster_sizes = {cid: len(units) for cid, units in clusters.items()}
      
      # 2. 【修正】计算有效总单位数 (排除噪声簇 -1)
      effective_sizes = {cid: size for cid, size in cluster_sizes.items() if cid != -1}
      total_effective_units = sum(effective_sizes.values())
      
      # 3. 处理极端情况 (与原始代码一致)
      if total_effective_units == 0:
            print("警告：未发现有效簇，退化为全局 PageRank Top-K。")
            all_units = [unit for units in clusters.values() for unit in units]
            return all_units[:sample_length]

      # 4. 初步分配预算给有效簇
      cluster_budgets = defaultdict(int)
      current_allocated = 0
      
      print("各有效簇大小及初步预算分配:")
      
      # 计算每个有效簇的预算
      for cid, size in effective_sizes.items():
            # 计算比例配额
            ratio = size / total_effective_units
            budget = int(sample_length * ratio)
            
            # 【修正】：核心修正！将预算上限设置为簇的实际大小 (防止超额分配)
            actual_budget = min(budget, size)
            
            cluster_budgets[cid] = actual_budget
            current_allocated += actual_budget
            print(f"   簇 {cid}: 大小 {size} ({ratio:.1%}) -> 分配 {actual_budget} (原预算 {budget})")
      
      if -1 in cluster_sizes:
            print(f"   簇 -1 (噪声): 大小 {cluster_sizes[-1]} -> 分配 0 (初始排除，作为补充)")


      # 5. 处理剩余预算：优先分配给最大有效簇（原逻辑），然后使用噪声点填充
      remaining_budget = sample_length - current_allocated
      
      # 阶段 5.1: 将剩余名额分配给最大的有效簇
      if remaining_budget > 0:
            print(f"   阶段 5.1: 剩余 {remaining_budget} 个名额，分配给最大的有效簇...")
            sorted_effective_clusters = sorted(effective_sizes.items(), key=lambda x: x[1], reverse=True)
            
            for cid, _ in sorted_effective_clusters:
                  if remaining_budget <= 0:
                        break
                  # 确保不会超过该簇实际拥有的数量
                  if cluster_budgets[cid] < cluster_sizes[cid]: 
                        cluster_budgets[cid] += 1
                        remaining_budget -= 1

      # 阶段 5.2: 【修正】如果仍有剩余预算，从噪声点中补充采样 (按 PageRank 排序)
      if remaining_budget > 0 and -1 in clusters:
            print(f"   阶段 5.2: 仍剩余 {remaining_budget} 个名额，从 {cluster_sizes[-1]} 个噪声点中补充采样 (PageRank Top-K)...")
            
            # 将剩余预算分配给噪声簇，但不超过噪声簇的实际大小
            noise_budget = min(remaining_budget, cluster_sizes[-1])
            cluster_budgets[-1] = noise_budget
            remaining_budget -= noise_budget # 更新剩余预算（可能仍然不为 0，如果噪声点也不够）
            print(f"   -> 噪声簇 (-1) 分配 {noise_budget} 个名额。")
            
            # 最终检查：如果到这里仍未达到目标，则无法满足
            if remaining_budget > 0:
                print(f"   警告：目标总长 {sample_length} 仍有 {remaining_budget} 个名额未满足 (数据点不足)。")

      # 6. 执行采样
      for cid, budget in cluster_budgets.items():
            if budget > 0:
                  # 取出该簇中 PR 最高的前 budget 个 (包括噪声簇 -1)
                  selection = clusters[cid][:budget]
                  sampled_text_units.extend(selection)

      # ======================================================================

      print(f"已采样 {len(sampled_text_units)} 个文本块。")
                  
      return sampled_text_units

