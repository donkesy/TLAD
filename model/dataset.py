import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import community as community_louvain
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import os
from dgl.dataloading import GraphDataLoader
import numpy as np
from collections import deque, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph
from itertools import islice, chain
import time
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import gc
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
from model.CGT_model import CGT_Model
seed = 66
torch.manual_seed(seed)


class GraphLoader:
    def __init__(self, nodes_file, links_file):
        """
        从您的文件格式加载数据并构建一个 NetworkX DiGraph。
        
        参数:
            nodes_file (str): 节点文件路径。
            links_file (str): 边文件路径。
        """
        self.nodes_df = pd.read_csv(nodes_file, 
                                    sep="\t", 
                                    header=None,   
                                    names=["node_id", "node_types", "node_attr"]
                                    )

        # 判断 links_file 是否为空
        if os.path.getsize(links_file) == 0:
            self.links_df = pd.DataFrame(columns=["u", "v", "edge_type"])
            self.graph = nx.DiGraph()  # 空图
        else:
            self.links_df = pd.read_csv(links_file
                                        , sep="\t",
                                        header=None,
                                        names=["u", "v", "edge_type", "is_edge"])   
        self.graph = self._build_graph()

    def _build_graph(self):
        G = nx.DiGraph()
        
        # 添加节点，并将所有属性存储在节点上
        for _, row in self.nodes_df.iterrows():
            # 将 node_attr (字符串) 转换为 numpy 数组
            # 假设属性是以空格分隔的浮点数
            # print(row['node_id'], row['node_types'], row['node_attr'])
            attr_vector = np.fromstring(row['node_attr'], sep=',')
            G.add_node(
                row['node_id'], 
                node_type=row['node_types'], 
                feature=attr_vector
            )
            
        # 添加边，并将边类型作为属性
        for _, row in self.links_df.iterrows():
            G.add_edge(
                row['u'], 
                row['v'], 
                edge_type=row['edge_type']
            )
            
        return G

    def get_graph(self):
        return self.graph

class PathSamplerAndAugmentor:
    """
    负责从图中采样路径并根据您的策略进行增强。
    围绕一个中心节点采样多条不同的路径。
    """
    def __init__(self, graph, k=7, m_pos=1, m_neg=6, 
                pad_node_type_id=5, pad_edge_type_id=10, num_paths_per_node=5):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.k = k
        self.num_paths_per_node = num_paths_per_node # 【新增】每个中心节点采样多少条路径
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.num_node_types = 6

        self.PAD_NODE_TYPE_ID = pad_node_type_id
        self.PAD_EDGE_TYPE_ID = pad_edge_type_id + self.num_node_types
        self.PAD_NODE_ID = -1 # 用于填充的特殊节点ID
        self.TARGET_NODE_TYPES = {"Asset-Transfer", "Invocation-Node", "Information-Node"}

    def _random_walk_down(self, start_node, length):
        """
        【新】从一个节点开始，进行“向下”（沿出边）的随机游走。
        """
        if start_node is None: return []
        
        path_steps = []
        current_node = start_node
        
        for i in range(length):
            # 添加当前节点
            step = {"node": current_node}
            
            # 寻找下一个节点和出边
            successors = list(self.graph.successors(current_node))
            if not successors:
                step["edge_to_next"] = None
                path_steps.append(step)
                break # 到达路径末端

            next_node = int(random.choice(successors))
            edge_data = self.graph.edges[current_node, next_node]
            step["edge_to_next"] = int(edge_data['edge_type'])# + self.num_node_types
            path_steps.append(step)
            
            current_node = next_node

        # 如果循环是因为达到长度限制而结束，确保最后一个节点也被添加
        if len(path_steps) < length + 1:
             path_steps.append({"node": current_node, "edge_to_next": None})
             
        # 确保路径长度不超过 k+1 个节点 (k条边)
        return path_steps[:length+1]

    def _random_walk_up(self, start_node, length):
        """
        【新】从一个节点开始，进行“向上”（沿入边）的回溯随机游走。
        """
        if start_node is None: return []

        reversed_path_steps = []
        current_node = start_node

        for _ in range(length):
            predecessors = list(self.graph.predecessors(current_node))
            if not predecessors:
                break # 到达路径的源头

            prev_node = int(random.choice(predecessors))
            edge_data = self.graph.edges[prev_node, current_node]

            # 我们是回溯地发现路径，所以记录的是 prev_node -> current_node 这一步
            reversed_path_steps.append({
                "node": prev_node,
                "edge_to_next": int(edge_data['edge_type'])# + self.num_node_types
            })
            current_node = prev_node
        
        # 因为我们是回溯的，所以需要将结果反转，得到正确的正向路径
        return reversed_path_steps[::-1]

    def sample_diverse_paths(self):
        start_node = int(random.choice(self.nodes))
        diverse_paths = []
        for _ in range(self.num_paths_per_node):
            up_walk_steps = self._random_walk_up(start_node, self.k)
            down_walk_steps = self._random_walk_down(start_node, self.k)
            anchor_path_steps = up_walk_steps + down_walk_steps
            diverse_paths.append(anchor_path_steps)
        
        return diverse_paths
  
    def _format_path_for_model(self, path_steps):
        """
        辅助函数，将原子单元列表转换为模型需要的字典格式。
        """
        if not path_steps: 
            return None
        path_dict = {"nodes": [], "node_types": [], "node_features": [], "edge_types": []}
        
        for i, step in enumerate(path_steps):
            node_id = step["node"]
            node_data = self.graph.nodes[node_id]
            
            path_dict["nodes"].append(node_id)
            path_dict["node_types"].append(node_data['node_type'])
            # path_dict["node_features"].append(node_data['feature'])
            
            if step.get('is_masked', False):
                # 如果这个step被标记为masked，我们添加一个零向量
                feature_dim = node_data['feature'].shape[0]
                path_dict["node_features"].append(np.zeros(feature_dim))
            else:
                # 否则，添加原始的特征向量
                path_dict["node_features"].append(node_data['feature'])

            # 边的数量应该比节点少一个
            if i < len(path_steps) - 1:
                edge_type = step["edge_to_next"]
                if edge_type is None:
                    path_dict["edge_types"].append(-1) # -1 代表路径自然终结或逻辑断裂
                else:
                    path_dict["edge_types"].append(int(edge_type))

        # 保证边的数量严格比节点少一
        while len(path_dict["edge_types"]) >= len(path_dict["nodes"]):
            path_dict["edge_types"].pop()
        while len(path_dict["edge_types"]) < len(path_dict["nodes"]) - 1:
             path_dict["edge_types"].append(-1)


        return path_dict
    
    def sample_diverse_paths_from_node(self, start_node):
        """
        围绕一个给定的 start_node 采样多条路径。
        """
        if start_node is None:
            return []
            
        diverse_paths = []
        for _ in range(self.num_paths_per_node):
            up_walk_steps = self._random_walk_up(start_node, self.k)
            down_walk_steps = self._random_walk_down(start_node, self.k)
            # 将向上和向下的路径在起始节点处拼接起来
            anchor_path_steps = up_walk_steps + [{"node": start_node, "edge_to_next": None}] + down_walk_steps[1:]
            
            # 简单的去重逻辑，确保我们不只是简单地重复相同的路径
            if any(step['node'] for step in anchor_path_steps) and anchor_path_steps not in diverse_paths:
                diverse_paths.append(anchor_path_steps)
        
        return diverse_paths

    def generate_samples_for_node(self, start_node):
        """
        【新的主方法】为给定的起始节点生成路径、正样本和负样本。
        """
        # 1. 围绕指定节点采样多条基础路径（锚点）
        diverse_paths = self.sample_diverse_paths_from_node(start_node)
        if not diverse_paths: 
            return []

        # 2. 对采样到的每一条路径进行正负增强 (与您之前的逻辑相同)
        all_samples = []
        for anchor_path_steps in diverse_paths:
            if not anchor_path_steps: continue

            anchor_path_dict = self._format_path_for_model(anchor_path_steps)
            if not anchor_path_dict:
                continue
                
            current_sample = {
                "anchor": anchor_path_dict,
                "positives": [],
                "negatives": []
            }

            for _ in range(self.m_pos):
                pos_steps = self._augment_positive(anchor_path_steps)
                pos_dict = self._format_path_for_model(pos_steps)
                if pos_dict:
                    current_sample["positives"].append(pos_dict)

            for _ in range(self.m_neg):
                neg_steps = self._augment_negative(anchor_path_steps)
                neg_dict = self._format_path_for_model(neg_steps)
                if neg_dict:
                    current_sample["negatives"].append(neg_dict)

            if current_sample["positives"] and current_sample["negatives"]:
                all_samples.append(current_sample)

        return all_samples
        
    def _augment_positive(self, original_path_steps):
        strategy = random.choice(['mask'])
        path_steps = [s.copy() for s in original_path_steps] # 深拷贝

        if strategy == 'mask':
            for step in path_steps:
                step['is_masked'] = False
                if random.random() < 0.15:
                    step['is_masked'] = True
            return path_steps 

    def _augment_negative(self, original_path_steps): #应该再添加一个去除掉前面几个节点,表示缺失某些先验条件
        strategy = random.choice(['delete', 'shuffle', 'crop'])
        # 创建一个副本以避免修改原始列表
        path_steps = [s.copy() for s in original_path_steps]

        if strategy == 'delete':
            # “删除节点及其出边”的操作，就是直接从原子单元列表中删除一个元素。
            # 例如，删除 A in [ {A, A->B}, {B, B->C}, {C, None} ]
            # 结果是 [ {B, B->C}, {C, None} ]，这自然地移除了A和A->B这条边。
            if len(path_steps) < 4: return path_steps
            delete_count = random.randint(1, max(1, len(path_steps) // 2))
            delete_count = min(delete_count, len(path_steps) - 2) 
            new_path_steps = path_steps[delete_count:]
            return new_path_steps

        if strategy == 'crop':
            num_steps = len(path_steps)
            min_len = max(4, int(0.3 * num_steps))
            if min_len >= num_steps: return path_steps
            return path_steps[min_len:]
        
        if strategy == 'shuffle':
            num_steps = len(path_steps)
            if num_steps < 4:  # 至少需要2个节点和2条边才有意义
                return path_steps

            # 1. 随机选择子区间
            start = random.randint(0, num_steps - 2)       # 起点
            steps = random.randint(2, num_steps - start)   # 区间长度至少2
            end = max(start + steps, num_steps)  # 终点

            # 2. 提取子区间
            sub_steps = path_steps[start:end]

            # 3. 分别提取节点和边
            nodes = [step['node'] for step in sub_steps]
            edges = [step['edge_to_next'] for step in sub_steps[:-1]]  # 最后一个没有边

            # 4. 分别打乱节点和边
            random.shuffle(nodes)
            random.shuffle(edges)

            # 5. 按 node-edge-node 规则重建子区间
            shuffled_sub_steps = []
            for i in range(len(nodes)):
                shuffled_sub_steps.append({
                    "node": nodes[i],
                    "edge_to_next": edges[i] if i < len(edges) else None
                })

            # 6. 拼回原序列
            shuffled_path_steps = path_steps[:start] + shuffled_sub_steps + path_steps[end:]

            return shuffled_path_steps
        
    def _pad_path(self, path, target_length, feature_dim):
        """
        辅助函数，将单个路径填充到目标长度。
        """
        current_length = len(path['nodes'])
        
        if current_length >= target_length:
            # 如果路径已经比目标长（例如，乱序路径长度不变），则直接返回
            # 或者可以进行截断，但保持原样更简单
            return path
        
        num_pads = target_length - current_length
        pad_feature = np.zeros(feature_dim)
        
        path['nodes'].extend([self.PAD_NODE_ID] * num_pads)
        # 【核心改動】使用安全的填充ID
        path['node_types'].extend([self.PAD_NODE_TYPE_ID] * num_pads)
        path['node_features'].extend([pad_feature] * num_pads)
        
        path['edge_types'].extend([self.PAD_EDGE_TYPE_ID] * num_pads)  #感觉这里的PAD_EDGE_TYPE_ID应该是[1, 0, 0 ,0 ,0],不应该是pad_edge_type_id + self.num_node_types 
        
        return path

class ContrastivePathDataset(Dataset):
    def __init__(self, list_of_graph_files, num_node_types, num_edge_types,
                 k=7, m_pos=1, m_neg=6, num_paths_per_node=5):
        
        self.k, self.m_pos, self.m_neg = k, m_pos, m_neg
        self.num_paths_per_node = num_paths_per_node
        self.node_type_pad_idx = num_node_types
        self.edge_type_pad_idx = num_edge_types

        # 加载所有有效的图
        self.valid_graphs = self._load_and_filter_graphs(list_of_graph_files)
        
        # 【核心修改】创建“主索引”，映射到每个图中的每个目标节点
        self.target_node_map = self._create_target_node_map()

        if not self.target_node_map:
            raise ValueError("在所有图中都未能找到任何目标节点!")

    def _load_and_filter_graphs(self, graph_files):
        valid_graphs = []
        print("Pre-loading and filtering graphs into memory...")
        for file_pair in tqdm(graph_files, desc="Loading graphs"):
            try:
                graph = GraphLoader(file_pair['nodes'], file_pair['links']).get_graph()
                if graph and graph.number_of_edges() > 0:
                    valid_graphs.append(graph)
            except Exception as e:
                print(f"Warning: Could not load graph from {file_pair['nodes']}. Error: {e}")
        return valid_graphs

    def _create_target_node_map(self):
        """
        遍历所有已加载的图，找出所有目标节点，并为它们创建索引。
        返回一个列表，每个元素是 (图在valid_graphs中的索引, 目标节点ID)。
        """
        print("Creating index of target nodes...")
        target_node_map = []
        TARGET_NODE_TYPES = {"Asset-Transfer", "Invocation-Node"}
        OTHER_NODE_TYPES = {"Information-Node", "Compute-Node"}
        
        for graph_idx, graph in enumerate(self.valid_graphs):
            for node_id, data in graph.nodes(data=True):
                node_type_str = data.get('node_type', '')
                # 检查节点类型字符串是否包含任何一个目标类型
                if any(target_type in node_type_str for target_type in TARGET_NODE_TYPES):
                    target_node_map.append((graph_idx, node_id))
                if len(target_node_map) >= 15:  # Apply limit
                    break
        while len(target_node_map) < 15:
            for node_id, data in enumerate(self.valid_graphs):
                node_type_str = data.get('node_type', '')
                # 检查节点类型字符串是否包含任何一个目标类型
                if any(target_type in node_type_str for target_type in OTHER_NODE_TYPES):
                    target_node_map.append((graph_idx, node_id))
                if len(target_node_map) >= 15:  # Apply limit
                    break
        
        print(f"Found {len(target_node_map)} target nodes across {len(self.valid_graphs)} graphs.")
        return target_node_map

    def __len__(self):
        """
        【核心修改】数据集的长度现在是目标节点的总数。
        """
        return len(self.target_node_map)

    def __getitem__(self, idx):
        """
        【核心修改】根据索引直接定位到一个目标节点并为其生成样本。
        """
        # 1. 从主索引中获取图的索引和目标节点的ID
        graph_idx, target_node_id = self.target_node_map[idx]
        graph = self.valid_graphs[graph_idx]

        # 2. 为这个特定的图和节点实例化采样器
        sampler = PathSamplerAndAugmentor(
            graph, self.k, self.m_pos, self.m_neg,
            num_paths_per_node=self.num_paths_per_node
        )

        # 3. 循环直到成功生成一个样本（以防万一某个节点无法生成路径）
        samples_list = []
        max_tries = 5 # 设置一个尝试上限
        tries = 0
        while not samples_list and tries < max_tries:
            # 调用新的主方法，传入目标节点ID
            samples_list = sampler.generate_samples_for_node(target_node_id)
            tries += 1
        
        # 如果多次尝试后仍然失败，可以返回 None，DataLoader 会处理
        if not samples_list:
            # 或者为了保证批次大小，可以随机取另一个样本
            # print(f"Warning: Failed to generate samples for node {target_node_id}. Resampling.")
            return None

        return samples_list