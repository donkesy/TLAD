import os
import random
import networkx as nx
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import deque, defaultdict
from itertools import chain

# 从您的原始代码中复制GraphLoader和PathSamplerAndAugmentor
# =========================================================================
# 以下是GraphLoader和PathSamplerAndAugmentor类的完整定义
# 请确保这些类与您当前代码中的定义完全一致

class GraphLoader:
    def __init__(self, nodes_file, links_file):
        self.nodes_df = pd.read_csv(nodes_file,
                                    sep="\t",
                                    header=None,
                                    names=["node_id", "node_types", "node_attr"])

        if os.path.getsize(links_file) == 0:
            self.links_df = pd.DataFrame(columns=["u", "v", "edge_type"])
            self.graph = nx.DiGraph()
        else:
            self.links_df = pd.read_csv(links_file,
                                        sep="\t",
                                        header=None,
                                        names=["u", "v", "edge_type", "is_edge"])
        self.graph = self._build_graph()

    def _build_graph(self):
        G = nx.DiGraph()
        for _, row in self.nodes_df.iterrows():
            attr_vector = np.fromstring(row['node_attr'], sep=',')
            G.add_node(
                row['node_id'],
                node_type=row['node_types'],
                feature=attr_vector
            )
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
    def __init__(self, graph, k=7, m_pos=1, m_neg=6,
                pad_node_type_id=5, pad_edge_type_id=10, num_paths_per_node=5):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.k = k
        self.num_paths_per_node = num_paths_per_node
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.num_node_types = 6 # 假设6种类型

        self.PAD_NODE_TYPE_ID = pad_node_type_id
        # self.PAD_EDGE_TYPE_ID = pad_edge_type_id + self.num_node_types # 这个在collate_fn中处理，这里不需要
        self.PAD_EDGE_TYPE_ID = pad_edge_type_id # 使用原始的边类型填充ID即可，因为在collate_fn中会统一转换为6位向量
        self.PAD_NODE_ID = -1
        self.TARGET_NODE_TYPES = {"Asset-Transfer", "Invocation-Node", "Information-Node"}

    def _random_walk_down(self, start_node, length):
        if start_node is None: return []

        path_steps = []
        current_node = start_node

        for i in range(length):
            step = {"node": current_node}

            successors = list(self.graph.successors(current_node))
            if not successors:
                step["edge_to_next"] = None
                path_steps.append(step)
                break

            next_node = int(random.choice(successors))
            edge_data = self.graph.edges[current_node, next_node]
            step["edge_to_next"] = int(edge_data['edge_type'])
            path_steps.append(step)

            current_node = next_node

        if len(path_steps) < length + 1:
             path_steps.append({"node": current_node, "edge_to_next": None})

        return path_steps[:length+1]

    def _random_walk_up(self, start_node, length):
        if start_node is None: return []

        reversed_path_steps = []
        current_node = start_node

        for _ in range(length):
            predecessors = list(self.graph.predecessors(current_node))
            if not predecessors:
                break

            prev_node = int(random.choice(predecessors))
            edge_data = self.graph.edges[prev_node, current_node]

            reversed_path_steps.append({
                "node": prev_node,
                "edge_to_next": int(edge_data['edge_type'])
            })
            current_node = prev_node

        return reversed_path_steps[::-1]

    def sample_diverse_paths_from_node(self, start_node):
        if start_node is None:
            return []

        diverse_paths = []
        for _ in range(self.num_paths_per_node):
            up_walk_steps = self._random_walk_up(start_node, self.k)
            down_walk_steps = self._random_walk_down(start_node, self.k)
            
            # 避免重复添加 start_node
            if up_walk_steps and down_walk_steps:
                 anchor_path_steps = up_walk_steps + [{"node": start_node, "edge_to_next": None}] + down_walk_steps[1:]
            elif up_walk_steps:
                 anchor_path_steps = up_walk_steps + [{"node": start_node, "edge_to_next": None}]
            elif down_walk_steps:
                 anchor_path_steps = [{"node": start_node, "edge_to_next": None}] + down_walk_steps
            else:
                 anchor_path_steps = [{"node": start_node, "edge_to_next": None}] # 只有anchor节点本身

            # 过滤掉空路径或者只有锚点本身的路径（如果需要更长的路径）
            if len(anchor_path_steps) <= 1: # 仅含起始节点
                continue

            # 简单的去重逻辑（根据节点序列判断），避免在少量路径采样时生成完全相同的路径
            path_nodes_sequence = tuple(step['node'] for step in anchor_path_steps)
            if path_nodes_sequence not in [tuple(step['node'] for step in p) for p in diverse_paths]:
                diverse_paths.append(anchor_path_steps)
        
        return diverse_paths

    def _format_path_for_model(self, path_steps):
        if not path_steps:
            return None
        path_dict = {"nodes": [], "node_types": [], "node_features": [], "edge_types": []}

        for i, step in enumerate(path_steps):
            node_id = step["node"]
            node_data = self.graph.nodes[node_id]

            path_dict["nodes"].append(node_id)
            path_dict["node_types"].append(node_data['node_type'])

            if step.get('is_masked', False):
                feature_dim = node_data['feature'].shape[0]
                path_dict["node_features"].append(np.zeros(feature_dim, dtype=np.float32)) # 确保dtype
            else:
                path_dict["node_features"].append(node_data['feature'].astype(np.float32)) # 确保dtype

            if i < len(path_steps) - 1:
                edge_type = step["edge_to_next"]
                if edge_type is None:
                    path_dict["edge_types"].append(-1)
                else:
                    path_dict["edge_types"].append(int(edge_type))

        while len(path_dict["edge_types"]) >= len(path_dict["nodes"]):
            path_dict["edge_types"].pop()
        while len(path_dict["edge_types"]) < len(path_dict["nodes"]) - 1:
             path_dict["edge_types"].append(-1)

        return path_dict

    def generate_samples_for_node(self, start_node):
        diverse_paths = self.sample_diverse_paths_from_node(start_node)
        if not diverse_paths:
            return []

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

            # 确保锚点路径长度至少为2，否则无法进行有意义的增强
            if len(anchor_path_steps) < 2:
                continue 

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

            # 只有当正负样本都存在时才添加这个样本组
            if current_sample["positives"] and current_sample["negatives"]:
                all_samples.append(current_sample)

        return all_samples

    def _augment_positive(self, original_path_steps):
        strategy = random.choice(['mask']) # 简化策略，只用mask
        path_steps = [s.copy() for s in original_path_steps]

        if strategy == 'mask':
            # 至少保留一个非masked节点
            num_nodes_to_mask = random.randint(1, max(1, len(path_steps) - 1)) # 至少mask1个，最多mask len-1个
            mask_indices = random.sample(range(len(path_steps)), num_nodes_to_mask)
            for i in mask_indices:
                path_steps[i]['is_masked'] = True
            return path_steps
        
        # 暂时不启用crop，因为会导致长度变化，在预处理时统一长度会更复杂
        # if strategy == 'crop':
        #     num_steps = len(path_steps)
        #     min_len = max(3, int(0.3 * num_steps))
        #     if min_len >= num_steps: return path_steps
        #     start = 0
        #     length = random.randint(min_len, num_steps)
        #     return path_steps[start:start+length]

    def _augment_negative(self, original_path_steps):
        strategy = random.choice(['delete', 'shuffle', 'crop']) # 'crop' 这里用于删去开头
        path_steps = [s.copy() for s in original_path_steps]

        if strategy == 'delete':
            if len(path_steps) < 3: return path_steps # 至少保留2个节点
            delete_count = random.randint(1, max(1, len(path_steps) // 2))
            delete_count = min(delete_count, len(path_steps) - 2)
            new_path_steps = path_steps[delete_count:]
            if not new_path_steps: return path_steps # 避免空路径
            return new_path_steps

        if strategy == 'crop': # 裁剪开头
            num_steps = len(path_steps)
            if num_steps < 3: return path_steps # 至少保留2个节点
            crop_start_idx = random.randint(1, num_steps - 2) # 至少删1个，最多删到只剩2个
            return path_steps[crop_start_idx:]

        if strategy == 'shuffle':
            num_steps = len(path_steps)
            if num_steps < 3: return path_steps # 至少3个节点，2条边才有意义

            # 随机选择子区间进行打乱
            start_idx = random.randint(0, num_steps - 2) # 子区间起点
            end_idx = random.randint(start_idx + 1, num_steps - 1) # 子区间终点，确保至少包含2个节点
            
            sub_path_steps = path_steps[start_idx : end_idx + 1] # 提取子序列

            nodes_in_sub = [step['node'] for step in sub_path_steps]
            edges_in_sub = [step['edge_to_next'] for step in sub_path_steps[:-1]]

            random.shuffle(nodes_in_sub)
            random.shuffle(edges_in_sub)

            shuffled_sub_steps = []
            for i in range(len(nodes_in_sub)):
                shuffled_sub_steps.append({
                    "node": nodes_in_sub[i],
                    "edge_to_next": edges_in_sub[i] if i < len(edges_in_sub) else None,
                    "is_masked": sub_path_steps[i].get('is_masked', False) # 保持masked状态
                })
            
            shuffled_path_steps = path_steps[:start_idx] + shuffled_sub_steps + path_steps[end_idx + 1:]
            return shuffled_path_steps
# =========================================================================

# 辅助函数：将路径填充到最大长度
def _pad_single_path(path_dict, max_nodes, feature_dim,
                     PAD_NODE_ID, PAD_NODE_TYPE_ID, PAD_EDGE_TYPE_ID):
    
    current_nodes = len(path_dict['nodes'])
    num_node_pads = max_nodes - current_nodes
    num_edge_pads = max_nodes - 1 - len(path_dict['edge_types'])

    if num_node_pads < 0 or num_edge_pads < 0:
        # 如果路径超过了max_nodes，则进行截断
        path_dict['nodes'] = path_dict['nodes'][:max_nodes]
        path_dict['node_types'] = path_dict['node_types'][:max_nodes]
        path_dict['node_features'] = path_dict['node_features'][:max_nodes]
        path_dict['edge_types'] = path_dict['edge_types'][:max_nodes - 1]
        return path_dict

    pad_feature = np.zeros(feature_dim, dtype=np.float32)

    path_dict['nodes'].extend([PAD_NODE_ID] * num_node_pads)
    path_dict['node_types'].extend([PAD_NODE_TYPE_ID] * num_node_pads)
    path_dict['node_features'].extend([pad_feature] * num_node_pads)

    path_dict['edge_types'].extend([PAD_EDGE_TYPE_ID] * num_edge_pads)

    return path_dict

def preprocess_and_save_samples(
    list_of_graph_files,
    output_dir,
    num_node_types, # 例如 5 (Asset-Transfer到Common-Node)
    num_edge_types, # 例如 10
    k=7, m_pos=1, m_neg=6, num_paths_per_node=5,
    max_path_nodes=64 # 新增参数：限制每个路径的最大节点数
):
    os.makedirs(output_dir, exist_ok=True)

    PAD_NODE_TYPE_ID_CONST = 5 # 对应您的node_type_map中Common-Node后的填充位
    PAD_EDGE_TYPE_ID_CONST = 10 # 假设原始边类型ID是0-9，10作为填充
    PAD_NODE_ID_CONST = -1

    # 1. 加载所有图
    print("Pre-loading and filtering graphs into memory...")
    valid_graphs = []
    # 临时存放第一个图的特征维度，用于填充
    first_graph_feature_dim = None

    for file_pair in tqdm(list_of_graph_files, desc="Loading graphs"):
        try:
            graph = GraphLoader(file_pair['nodes'], file_pair['links']).get_graph()
            if graph and graph.number_of_edges() > 0 and graph.number_of_nodes() > 0:
                valid_graphs.append(graph)
                # 获取特征维度
                if first_graph_feature_dim is None:
                    for node_id, data in graph.nodes(data=True):
                        if 'feature' in data and data['feature'] is not None:
                            first_graph_feature_dim = data['feature'].shape[0]
                            break
        except Exception as e:
            print(f"Warning: Could not load graph from {file_pair['nodes']}. Error: {e}")
            continue

    if not valid_graphs:
        print("No valid graphs loaded. Exiting preprocessor.")
        return

    if first_graph_feature_dim is None:
        print("Could not determine node feature dimension. Exiting preprocessor.")
        return

    # 2. 创建目标节点索引
    print("Creating index of target nodes...")
    target_node_map = []
    TARGET_NODE_TYPES = {"Asset-Transfer", "Invocation-Node"}
    # OTHER_NODE_TYPES = {"Information-Node", "Compute-Node"} # 暂时只采样目标节点，以减少样本量

    for graph_idx, graph in enumerate(valid_graphs):
        for node_id, data in graph.nodes(data=True):
            node_type_str = data.get('node_type', '')
            if any(target_type in node_type_str for target_type in TARGET_NODE_TYPES):
                target_node_map.append((graph_idx, node_id))

    print(f"Found {len(target_node_map)} target nodes across {len(valid_graphs)} graphs.")
    if not target_node_map:
        print("No target nodes found. Exiting preprocessor.")
        return

    # 3. 为每个目标节点生成样本并保存
    print("Generating and saving samples...")
    for idx, (graph_idx, target_node_id) in enumerate(tqdm(target_node_map, desc="Processing target nodes")):
        graph = valid_graphs[graph_idx]
        sampler = PathSamplerAndAugmentor(
            graph, k, m_pos, m_neg,
            pad_node_type_id=PAD_NODE_TYPE_ID_CONST, # 保持与collate_fn中的映射一致
            pad_edge_type_id=PAD_EDGE_TYPE_ID_CONST, # 保持与collate_fn中的映射一致
            num_paths_per_node=num_paths_per_node
        )

        samples_list = []
        max_tries = 5 # 设置一个尝试上限
        tries = 0
        while not samples_list and tries < max_tries:
            try:
                samples_list = sampler.generate_samples_for_node(target_node_id)
            except Exception as e:
                # print(f"Error generating samples for node {target_node_id} in graph {graph_idx}: {e}")
                samples_list = [] # 确保失败时列表为空
            tries += 1

        if not samples_list:
            # print(f"Warning: Failed to generate valid samples for node {target_node_id} after {max_tries} tries. Skipping.")
            continue
        
        # 对生成的样本进行统一填充
        padded_samples_list = []
        for sample_group in samples_list:
            padded_sample_group = {
                "anchor": _pad_single_path(sample_group["anchor"], max_path_nodes, first_graph_feature_dim, PAD_NODE_ID_CONST, PAD_NODE_TYPE_ID_CONST, PAD_EDGE_TYPE_ID_CONST),
                "positives": [],
                "negatives": []
            }
            for path_dict in sample_group["positives"]:
                padded_sample_group["positives"].append(
                    _pad_single_path(path_dict, max_path_nodes, first_graph_feature_dim, PAD_NODE_ID_CONST, PAD_NODE_TYPE_ID_CONST, PAD_EDGE_TYPE_ID_CONST)
                )
            for path_dict in sample_group["negatives"]:
                padded_sample_group["negatives"].append(
                    _pad_single_path(path_dict, max_path_nodes, first_graph_feature_dim, PAD_NODE_ID_CONST, PAD_NODE_TYPE_ID_CONST, PAD_EDGE_TYPE_ID_CONST)
                )
            padded_samples_list.append(padded_sample_group)

        # 保存为.pt文件
        output_path = os.path.join(output_dir, f"sample_{idx}.pt")
        torch.save(padded_samples_list, output_path)

    print(f"Preprocessing complete. Samples saved to {output_dir}")

# 示例调用
if __name__ == "__main__":
    base_root = "E:\\Py_projects\\CrossVulDec\\cleandata\\cleandata_768\\pretrain_small"
    graph_files = []
    for subdir in os.listdir(base_root):
        subdir_path = os.path.join(base_root, subdir)
        for dstdir in os.listdir(subdir_path):
            dstdir_path = os.path.join(subdir_path, dstdir)
            if os.path.isdir(dstdir_path):
                nodes_file = os.path.join(dstdir_path, "graph_node_ST.txt")
                links_file = os.path.join(dstdir_path, "graph_link_ST.txt")
                if os.path.exists(nodes_file) and os.path.exists(links_file):
                    graph_files.append({"nodes": nodes_file, "links": links_file})

    output_directory = "preprocessed_samples" # 输出目录
    
    preprocess_and_save_samples(
        list_of_graph_files=graph_files,
        output_dir=output_directory,
        num_node_types=5, # 根据您的实际节点类型数量调整
        num_edge_types=10, # 根据您的实际边类型数量调整
        k=7, m_pos=1, m_neg=6, num_paths_per_node=5,
        max_path_nodes=8 # 设置最大路径长度，所有样本将填充到此长度
    )