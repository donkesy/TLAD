from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import os
import pickle
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from model.CGT_model import CGT_Model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

class InferenceDataset(Dataset):
    """
    加载推断数据，并直接使用文件中保存的 contract_name。
    """
    def __init__(self, preprocessed_data_dir):
        self.files = [
            os.path.join(preprocessed_data_dir, f)
            for f in os.listdir(preprocessed_data_dir)
            if f.endswith('.pt')
        ]
        if not self.files:
            raise ValueError(f"在目录 {preprocessed_data_dir} 中没有找到 .pt 文件")
        logger.info(f"成功初始化推断数据集，从 {preprocessed_data_dir} 加载了 {len(self.files)} 个合约文件。")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            data = torch.load(file_path)
            # data 的格式是 {'paths': [...], 'label': ..., 'contract_name': '...'}
            paths = data['paths']
            # 【核心改动】直接从文件中读取 contract_name
            contract_name = data['contract_name']
            
            return paths, contract_name
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {e}")
            return None, None

def inference_collate_fn(batch_of_paths_and_names):
    """
    收集路径和它们对应的 contract_name。
    """
    all_paths = []
    names_per_path = []

    for paths, name in batch_of_paths_and_names:
        if paths and name:
            all_paths.extend(paths)
            names_per_path.extend([name] * len(paths))
            
    if not all_paths:
        return None, None
        
    padded_batch = pad_paths_to_tensors(all_paths)
    
    return padded_batch, names_per_path

class ContrastivePathDataset(Dataset):
    """
    【修改版】此Dataset直接从预处理好的数据目录中加载样本。
    """
    def __init__(self, preprocessed_data_dir):
        """
        初始化数据集。
        参数:
            preprocessed_data_dir (str): 存放 .pt 预处理文件的目录路径。
        """
        self.preprocessed_files = [
            os.path.join(preprocessed_data_dir, f)
            for f in os.listdir(preprocessed_data_dir)
            if f.endswith('.pt')
        ]
        if not self.preprocessed_files:
            raise ValueError(f"在目录 {preprocessed_data_dir} 中没有找到预处理好的 .pt 文件")
        
        logger.info(f"成功初始化数据集，从 {preprocessed_data_dir} 加载了 {len(self.preprocessed_files)} 个预处理样本文件。")

    def __len__(self):
        return len(self.preprocessed_files)

    def __getitem__(self, idx):
        file_path = self.preprocessed_files[idx]
        try:
            # 直接从磁盘加载一个已经处理好的样本列表
            samples_list = torch.load(file_path)
            return samples_list
        except Exception as e:
            logger.error(f"加载预处理文件 {file_path} 时出错: {e}")
            # 返回 None, DataLoader 会自动跳过这个坏样本
            return None

# 您的节点类型到6位向量的映射
node_type_map = {
    "Asset-Transfer":   torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32),
    "Invocation-Node":  torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32),
    "Compute-Node":     torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32),
    "Information-Node": torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32),
    "Common-Node":      torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32),
}

def pad_paths_to_tensors(paths):
    """
    【修改版】此函数将预填充好的路径数据转换为模型所需的张量格式。
    主要工作是将节点/边类型ID转换为统一的6位向量。
    """
    if not paths:
        return None

    NODE_PAD_VECTOR = torch.zeros(6, dtype=torch.float32)
    EDGE_PAD_VECTOR = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32)
    BROKEN_EDGE_VECTOR = EDGE_PAD_VECTOR

    batch_node_features = []
    batch_unified_types = []

    for p in paths:
        # 1. 节点特征: 直接将预处理的numpy数组列表转换为张量
        nf = [torch.from_numpy(f) for f in p['node_features']]
        batch_node_features.append(torch.stack(nf))

        # 2. 统一类型序列: 将节点类型和边类型交织并编码
        unified_type_seq = []
        max_nodes_in_path = len(p['nodes'])
        max_seq_len = 2 * max_nodes_in_path - 1 if max_nodes_in_path > 0 else 0

        for i in range(max_seq_len):
            if i % 2 == 0:  # 偶数位是节点
                node_idx = i // 2
                node_type = p['node_types'][node_idx]
                
                # 检查是否为填充节点类型
                if isinstance(node_type, (int, float)) and node_type in [-1, 5]: # 兼容不同的填充ID
                    unified_type_seq.append(NODE_PAD_VECTOR)
                else:
                    binary_vec = torch.zeros(6, dtype=torch.float32)
                    for t in node_type.split(","):
                        t = t.strip()
                        if t in node_type_map:
                            binary_vec += node_type_map[t]
                    unified_type_seq.append(binary_vec)
            else:  # 奇数位是边
                edge_idx = i // 2
                edge_type_id = p['edge_types'][edge_idx]
                
                if edge_type_id == -1:  # 逻辑断裂
                    unified_type_seq.append(BROKEN_EDGE_VECTOR)
                elif edge_type_id == 10: # 边填充ID
                    unified_type_seq.append(EDGE_PAD_VECTOR)
                else:
                    edge_vec = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32)
                    binary_str = bin(int(edge_type_id))[2:].zfill(4)
                    for j, bit in enumerate(binary_str):
                        if bit == '1':
                            edge_vec[j + 2] = 1.0
                    unified_type_seq.append(edge_vec)
        
        batch_unified_types.append(torch.stack(unified_type_seq))

    final_node_features = torch.stack(batch_node_features)
    final_unified_types = torch.stack(batch_unified_types)

    # 拆分以对齐模型输入
    final_node_types = final_unified_types[:, 0::2, :]
    final_edge_types = final_unified_types[:, 1::2, :]

    return {
        "node_features": final_node_features,
        "node_types": final_node_types,
        "edge_types": final_edge_types
    }

def contrastive_collate_fn(batch_of_sample_lists):
    """
    【修改版】收集函数，现在处理的是已经预处理和预填充的样本。
    """
    # 过滤掉加载失败的样本 (返回值为None)
    batch = [b for b in batch_of_sample_lists if b is not None and len(b) > 0]
    if not batch:
        return None

    # 将批次内的样本列表展平
    all_samples = list(chain.from_iterable(batch))

    if not all_samples:
        return None

    all_paths = []
    num_pos_per_anchor = []
    num_neg_per_anchor = []

    for sample in all_samples:
        all_paths.append(sample['anchor'])
        all_paths.extend(sample['positives'])
        all_paths.extend(sample['negatives'])
        
        num_pos_per_anchor.append(len(sample['positives']))
        num_neg_per_anchor.append(len(sample['negatives']))

    padded_batch = pad_paths_to_tensors(all_paths) 
    
    return (padded_batch, torch.tensor(num_pos_per_anchor), torch.tensor(num_neg_per_anchor))

class PrototypicalContrastiveLoss(nn.Module):
    """
    【简化优化版】原型对比损失 - 专为异常检测设计
    
    该版本简化了损失函数，使其目标更明确，避免了不同损失项之间的目标重叠，
    并修复了分离损失的计算逻辑。

    核心思想:
    1. 紧凑性 (Compactness): 使用均方误差直接惩罚正常样本(anchor, positive)与
       原型(prototype)之间的距离，鼓励它们聚集在原型周围。
    2. 分离性 (Separation): 使用带有边距(margin)的Hinge Loss，强制*每一个*
       异常样本(negative)与原型保持一个最小距离，确保边界清晰。
    
    所有计算都在L2归一化的单位超球面上进行，以保证数值稳定。
    """
    def __init__(self, margin=2.0, compact_weight=0.5, prototype_momentum=0.99):
        """
        初始化损失函数。
        参数:
            margin (float): 分离损失的边距。在单位超球面上，两个向量的最大距离平方是4，
                            因此 2.0 是一个比较强的边距。
            compact_weight (float): 紧凑损失的权重，用于平衡两项损失。
            prototype_momentum (float): 原型更新时的指数移动平均(EMA)动量。
        """
        super().__init__()
        self.margin = margin
        self.compact_weight = compact_weight
        self.prototype_momentum = prototype_momentum
        self.prototype = None  # 初始化为None,首次调用时根据设备创建
        
    def initialize_prototype(self, embed_dim, device):
        """第一次调用时初始化原型"""
        if self.prototype is None:
            # 初始化为一个零向量，将在第一次更新时被替换
            self.prototype = torch.zeros(1, embed_dim, device=device)
    
    def update_prototype(self, normal_embeddings_norm):
        """
        【内部函数】使用EMA更新原型。
        关键：始终保持原型在单位超球面上。
        """
        with torch.no_grad():
            # 计算批次内正常样本的中心
            new_prototype = normal_embeddings_norm.mean(dim=0, keepdim=True)
            
            # 如果原型是第一次初始化（零向量），直接赋值
            if self.prototype is None or self.prototype.norm() < 1e-6:
                self.prototype = F.normalize(new_prototype, p=2, dim=1)
            else:
                # 确保设备一致
                if self.prototype.device != new_prototype.device:
                    self.prototype = self.prototype.to(new_prototype.device)
                    
                # EMA 更新
                updated_proto = (self.prototype_momentum * self.prototype + 
                                (1 - self.prototype_momentum) * new_prototype)
                
                # 【关键】更新后必须再次归一化，以确保原型始终在单位球面上
                self.prototype = F.normalize(updated_proto, p=2, dim=1)
    
    def forward(self, anchor, positive, negatives):
        """
        计算损失。
        anchor: (B, D) - 原始样本
        positive: (B, D) - 正样本(特征掩码)
        negatives: (B, N, D) - 负样本(错误顺序)
        """
        B, D = anchor.shape
        
        # 步骤 0: 确保原型已初始化并在正确的设备上
        if self.prototype is None:
            self.initialize_prototype(D, anchor.device)
        if self.prototype.device != anchor.device:
            self.prototype = self.prototype.to(anchor.device)
        
        # 步骤 1: 【关键】对所有嵌入向量进行L2归一化
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negatives_norm = F.normalize(negatives, p=2, dim=2) # 在最后一个维度(D)上归一化
        
        # 步骤 2: 计算紧凑损失 (Compact Loss)
        # 目标: 拉近所有正常样本 (anchor, positive) 到原型
        dist_anchor_proto = torch.sum((anchor_norm - self.prototype) ** 2, dim=1)
        dist_pos_proto = torch.sum((positive_norm - self.prototype) ** 2, dim=1)
        compact_loss = (dist_anchor_proto.mean() + dist_pos_proto.mean()) / 2

        # 步骤 3: 计算分离损失 (Separation Loss)
        # 目标: 将所有负样本推离原型，保持一个最小边距 margin
        # dist_neg_proto 的形状: (B, N)
        dist_neg_proto = torch.sum((negatives_norm - self.prototype.unsqueeze(1)) ** 2, dim=2)
        
        # 【正确实现】对每一个负样本的距离应用 Hinge Loss，然后再求平均
        # 这确保了模型会惩罚每一个离得太近的负样本
        separation_loss = F.relu(self.margin - dist_neg_proto).mean()

        # 步骤 4: 组合损失
        total_loss = self.compact_weight * compact_loss + separation_loss
        
        # 步骤 5: 更新原型 (如果损失有效)
        if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
            normal_samples_norm = torch.cat([anchor_norm, positive_norm], dim=0)
            self.update_prototype(normal_samples_norm)
        
        return total_loss, {
            'total_loss': total_loss.item() if not torch.isnan(total_loss) else float('nan'),
            'compact': compact_loss.item() if not torch.isnan(compact_loss) else float('nan'),
            'separation': separation_loss.item() if not torch.isnan(separation_loss) else float('nan'),
            'proto_norm': self.prototype.norm().item(), # 应恒为 1.0
        }

class BoundaryVAE(nn.Module):
    """
    边界VAE - 专门学习正常和异常边界的变分自编码器
    """
    def __init__(self, input_dim, latent_dim=128, hidden_dim=512):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def slerp(self, p0, p1, t, eps=1e-8):
        """
        球形插值函数。
        p0, p1: (N, D) 形状的张量
        t: (N, 1) 形状的插值权重
        """
        # (这里放入上面提供的 slerp 函数代码)
        p0_norm = F.normalize(p0, p=2, dim=-1)
        p1_norm = F.normalize(p1, p=2, dim=-1)
        dot = torch.sum(p0_norm * p1_norm, dim=-1, keepdim=True)
        omega = torch.acos(dot.clamp(-1.0 + eps, 1.0 - eps))
        sin_omega = torch.sin(omega)
        mask = sin_omega.abs() < eps
        lerp_result = (1.0 - t) * p0 + t * p1
        s0 = torch.sin((1.0 - t) * omega) / sin_omega
        s1 = torch.sin(t * omega) / sin_omega
        slerp_result = s0 * p0 + s1 * p1
        result = torch.where(mask, lerp_result, slerp_result)
        return result
    
    def generate_boundary_samples(self, normal_emb, anomaly_emb, num_samples=32, 
                                  boundary_range=(0.35, 0.85)):
        """
        【Slerp 修改版】在正常和异常样本之间生成边界样本。
        """
        with torch.no_grad():
            mu_normal, _ = self.encode(normal_emb)
            mu_anomaly, _ = self.encode(anomaly_emb)
            
            # 随机选择配对 (逻辑不变)
            B = min(len(mu_normal), len(mu_anomaly))
            if B == 0: return None, None # 如果其中一个为空，则无法生成
            num_samples = min(num_samples, B)

            idx_normal = torch.randperm(len(mu_normal))[:num_samples]
            idx_anomaly = torch.randperm(len(mu_anomaly))[:num_samples]
            
            selected_normal = mu_normal[idx_normal]
            selected_anomaly = mu_anomaly[idx_anomaly]
            
            device = normal_emb.device
            weights = (torch.rand(num_samples, 1, device=device) * 
                      (boundary_range[1] - boundary_range[0]) + boundary_range[0])
            
            # --- 【核心修改】 ---
            # 【旧代码】线性插值 (Lerp)
            # z_boundary = selected_normal * (1 - weights) + selected_anomaly * weights
            
            # 【新代码】球形插值 (Slerp)
            z_boundary = self.slerp(selected_normal, selected_anomaly, weights)
            # --- 结束修改 ---
            
            boundary_samples = self.decode(z_boundary)
            
            return boundary_samples, weights
        
    # def generate_boundary_samples(self, normal_emb, anomaly_emb, num_samples=32, 
    #                               boundary_range=(0.3, 0.7)):
    #     """
    #     在正常和异常样本之间生成边界样本
    #     boundary_range: 插值权重范围,越接近0.5越难
    #     """
    #     with torch.no_grad():
    #         # 编码到潜在空间
    #         mu_normal, _ = self.encode(normal_emb)
    #         mu_anomaly, _ = self.encode(anomaly_emb)
            
    #         # 随机选择配对
    #         B = min(len(mu_normal), len(mu_anomaly))
    #         idx_normal = torch.randperm(len(mu_normal))[:num_samples]
    #         idx_anomaly = torch.randperm(len(mu_anomaly))[:num_samples]
            
    #         selected_normal = mu_normal[idx_normal]
    #         selected_anomaly = mu_anomaly[idx_anomaly]
            
    #         # 在潜在空间插值
    #         device = normal_emb.device
    #         weights = (torch.rand(num_samples, 1, device=device) * 
    #                   (boundary_range[1] - boundary_range[0]) + boundary_range[0])
            
    #         z_boundary = selected_normal * (1 - weights) + selected_anomaly * weights
            
    #         # 解码回嵌入空间
    #         boundary_samples = self.decode(z_boundary)
            
    #         return boundary_samples, weights


# ============================================================================
# 阶段 2: 训练流程
# ============================================================================

class AnomalyDetectionTrainer:
    """完整的训练流程管理器"""
    
    def __init__(self, model, embed_dim, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # 损失函数
        self.contrastive_loss = PrototypicalContrastiveLoss(
            margin=config['margin'],
            compact_weight=config['compact_weight'],
            prototype_momentum=config['prototype_momentum']
        ).to(device)
        
        self.contrastive_loss.initialize_prototype(embed_dim, device)
        
        # VAE
        self.vae = BoundaryVAE(
            input_dim=embed_dim,
            latent_dim=config['latent_dim'],
            hidden_dim=config['vae_hidden_dim']
        ).to(device)
        
        # 优化器
        self.optimizer_model = optim.AdamW(
            model.parameters(),
            lr=config['lr_model'],
            weight_decay=config['weight_decay']
        )
        
        self.optimizer_vae = optim.AdamW(
            self.vae.parameters(),
            lr=config['lr_vae'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler_model = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_model, T_0=10, T_mult=2
        )
        
        self.scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_vae, mode='min', factor=0.5, patience=5
        )
        
        # 统计信息
        self.embedding_stats = {'mean': None, 'std': None}
        self.training_history = {
            'phase1': [], 'phase2': [], 'phase3': []
        }
    
    def compute_embedding_statistics(self, dataloader):
        """计算嵌入的统计信息用于标准化"""
        logger.info("计算嵌入统计信息...")
        all_embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="统计嵌入"):
                if batch_data is None:
                    continue
                
                padded_batch, _, _ = batch_data
                node_features = padded_batch['node_features'].to(self.device)
                node_types = padded_batch['node_types'].to(self.device)
                edge_types = padded_batch['edge_types'].to(self.device)
                
                embeddings = self.model(node_features, node_types, edge_types)
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.embedding_stats['mean'] = all_embeddings.mean(dim=0).to(self.device)
        self.embedding_stats['std'] = (all_embeddings.std(dim=0) + 1e-6).to(self.device)
        
        logger.info(f"嵌入统计: mean norm={self.embedding_stats['mean'].norm():.4f}, "
                   f"std mean={self.embedding_stats['std'].mean():.4f}")
    
    def standardize_embeddings(self, embeddings):
        """标准化嵌入"""
        return (embeddings - self.embedding_stats['mean']) / self.embedding_stats['std']
    
    def unstandardize_embeddings(self, embeddings):
        """反标准化"""
        return embeddings * self.embedding_stats['std'] + self.embedding_stats['mean']
    
    # ========================================================================
    # 阶段 1: 基础对比学习 (Epochs 1-50)
    # ========================================================================
    def train_phase1_contrastive(self, dataloader, num_epochs):
        """
        阶段1: 使用原型对比损失训练编码器
        目标: 学习正常样本的紧密表征
        """
        logger.info("="*60)
        logger.info("阶段 1: 基础对比学习 (学习正常样本表征)")
        logger.info("="*60)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_metrics = {
                'total_loss': 0.0,
                'compact': 0.0,
                'separation': 0.0,
                'batches': 0
            }
            
            pbar = tqdm(dataloader, desc=f"Phase 1 Epoch {epoch+1}/{num_epochs}")
            
            for batch_data in pbar:
                if batch_data is None:
                    continue
                
                padded_batch, num_pos_per_anchor, num_neg_per_anchor = batch_data
                
                # 验证数据一致性
                if not (all(n == num_pos_per_anchor[0] for n in num_pos_per_anchor) and
                       all(n == num_neg_per_anchor[0] for n in num_neg_per_anchor)):
                    continue
                
                num_pos = num_pos_per_anchor[0].item()
                num_neg = num_neg_per_anchor[0].item()
                
                # 前向传播
                node_features = padded_batch['node_features'].to(self.device)
                node_types = padded_batch['node_types'].to(self.device)
                edge_types = padded_batch['edge_types'].to(self.device)
                
                embeddings = self.model(node_features, node_types, edge_types)
                
                # 分离 anchor, positive, negative
                B = len(num_pos_per_anchor)
                total_paths = 1 + num_pos + num_neg
                emb_reshaped = embeddings.view(B, total_paths, -1)
                
                anchor = emb_reshaped[:, 0, :]  # (B, D)
                positive = emb_reshaped[:, 1, :]  # (B, D) - 只取第一个正样本
                negatives = emb_reshaped[:, 1+num_pos:, :]  # (B, num_neg, D)
                
                # 计算损失
                self.optimizer_model.zero_grad()
                loss, metrics = self.contrastive_loss(anchor, positive, negatives)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer_model.step()
                    
                    # 更新统计
                    epoch_metrics['total_loss'] += loss.item()
                    epoch_metrics['compact'] += metrics['compact']
                    epoch_metrics['separation'] += metrics['separation']
                    epoch_metrics['batches'] += 1
                    
                    pbar.set_postfix({
                        'loss': f"{metrics['total_loss']:.4f}",
                        'comp': f"{metrics['compact']:.3f}", # compact loss
                        'sep': f"{metrics['separation']:.3f}", # separation loss
                        'proto': f"{metrics['proto_norm']:.2f}"
                    })
            
            # Epoch 总结
            if epoch_metrics['batches'] > 0:
                avg_loss = epoch_metrics['total_loss'] / epoch_metrics['batches']
                avg_metrics = {k: v / epoch_metrics['batches'] 
                              for k, v in epoch_metrics.items() if k != 'batches'}
                
                logger.info(f"Phase 1 Epoch {epoch+1}/{num_epochs} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Compact: {avg_metrics['compact']:.4f} | "
                          f"Separation: {avg_metrics['separation']:.4f}")
                
                self.training_history['phase1'].append(avg_metrics)
                
                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint('phase1_best.pth', epoch, avg_loss)
                    logger.info(f"  ✓ 保存最佳模型 (loss: {best_loss:.4f})")
            
            self.scheduler_model.step()
            self._cleanup_memory()
        
        logger.info(f"阶段 1 完成,最佳损失: {best_loss:.4f}\n")
        return best_loss
    
    # ========================================================================
    # 阶段 2: 训练 VAE (Epochs 51-100)
    # ========================================================================
    def train_phase2_vae(self, dataloader, num_epochs):
        """
        阶段2: 训练 VAE 学习嵌入分布
        目标: 能够生成介于正常和异常之间的边界样本
        """
        logger.info("="*60)
        logger.info("阶段 2: 训练边界VAE")
        logger.info("="*60)
        
        # 冻结编码器
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # 计算嵌入统计信息
        self.compute_embedding_statistics(dataloader)
        
        best_vae_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.vae.train()
            epoch_metrics = {
                'total': 0.0,
                'recon': 0.0,
                'kl': 0.0,
                'batches': 0
            }
            
            pbar = tqdm(dataloader, desc=f"Phase 2 Epoch {epoch+1}/{num_epochs}")
            
            for batch_data in pbar:
                if batch_data is None:
                    continue
                
                padded_batch, _, _ = batch_data
                
                # 获取嵌入
                with torch.no_grad():
                    node_features = padded_batch['node_features'].to(self.device)
                    node_types = padded_batch['node_types'].to(self.device)
                    edge_types = padded_batch['edge_types'].to(self.device)
                    
                    embeddings = self.model(node_features, node_types, edge_types)
                    standardized_emb = self.standardize_embeddings(embeddings)
                
                # VAE 前向传播
                self.optimizer_vae.zero_grad()
                recon, mu, logvar = self.vae(standardized_emb)
                
                # VAE 损失
                recon_loss = F.mse_loss(recon, standardized_emb, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vae_loss = recon_loss + self.config['vae_beta'] * kl_loss
                
                if not torch.isnan(vae_loss):
                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    self.optimizer_vae.step()
                    
                    epoch_metrics['total'] += vae_loss.item()
                    epoch_metrics['recon'] += recon_loss.item()
                    epoch_metrics['kl'] += kl_loss.item()
                    epoch_metrics['batches'] += 1
                    
                    pbar.set_postfix({
                        'vae': f"{vae_loss.item():.4f}",
                        'recon': f"{recon_loss.item():.4f}",
                        'kl': f"{kl_loss.item():.4f}"
                    })
            
            # Epoch 总结
            if epoch_metrics['batches'] > 0:
                avg_loss = epoch_metrics['total'] / epoch_metrics['batches']
                avg_metrics = {k: v / epoch_metrics['batches'] 
                              for k, v in epoch_metrics.items() if k != 'batches'}
                
                logger.info(f"Phase 2 Epoch {epoch+1}/{num_epochs} | "
                          f"VAE Loss: {avg_loss:.4f} | "
                          f"Recon: {avg_metrics['recon']:.4f} | "
                          f"KL: {avg_metrics['kl']:.4f}")
                
                self.training_history['phase2'].append(avg_metrics)
                
                # 保存最佳模型
                if avg_loss < best_vae_loss:
                    best_vae_loss = avg_loss
                    self.save_checkpoint('phase2_vae_best.pth', epoch, avg_loss)
                    logger.info(f"  ✓ 保存最佳VAE (loss: {best_vae_loss:.4f})")
                
                self.scheduler_vae.step(avg_loss)
            
            self._cleanup_memory()
        
        logger.info(f"阶段 2 完成,最佳VAE损失: {best_vae_loss:.4f}\n")
        return best_vae_loss
    
    # ========================================================================
    # 阶段 3: 使用边界样本微调 (Epochs 101-200)
    # ========================================================================
    def train_phase3_finetune(self, dataloader, num_epochs):
        """
        阶段3: 使用VAE生成的边界样本增强训练
        目标: 提升模型对难样本的判别能力
        """
        logger.info("="*60)
        logger.info("阶段 3: 边界样本增强训练")
        logger.info("="*60)
        
        # 解冻编码器,冻结VAE
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.model.train()
        self.vae.eval()
        
        finetune_lr = self.config['lr_model'] / 10
        optimizer_finetune = optim.AdamW(
            self.model.parameters(),
            lr=finetune_lr,
            weight_decay=self.config['weight_decay']
        )
        
        scheduler_finetune = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_finetune, 
        T_0=15,  # 可以为微调阶段设置更长的重启周期
        T_mult=2
    )
        # 降低学习率
        # for param_group in self.optimizer_model.param_groups:
        #     param_group['lr'] = self.config['lr_model'] / 10
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                'total_loss': 0.0,
                'base_loss': 0.0,  # 原来的 'contrastive'
                'boundary_loss': 0.0, # 原来的 'boundary'
                'batches': 0
            }
            
            pbar = tqdm(dataloader, desc=f"Phase 3 Epoch {epoch+1}/{num_epochs}")
            
            for batch_data in pbar:
                if batch_data is None:
                    continue
                
                padded_batch, num_pos_per_anchor, num_neg_per_anchor = batch_data
                
                if not (all(n == num_pos_per_anchor[0] for n in num_pos_per_anchor) and
                       all(n == num_neg_per_anchor[0] for n in num_neg_per_anchor)):
                    continue
                
                num_pos = num_pos_per_anchor[0].item()
                num_neg = num_neg_per_anchor[0].item()
                
                # 前向传播
                node_features = padded_batch['node_features'].to(self.device)
                node_types = padded_batch['node_types'].to(self.device)
                edge_types = padded_batch['edge_types'].to(self.device)
                
                embeddings = self.model(node_features, node_types, edge_types)
                
                # 分离样本
                B = len(num_pos_per_anchor)
                total_paths = 1 + num_pos + num_neg
                emb_reshaped = embeddings.view(B, total_paths, -1)
                
                anchor = emb_reshaped[:, 0, :]
                positive = emb_reshaped[:, 1, :]
                negatives = emb_reshaped[:, 1+num_pos:, :]
                
                # 1. 原始对比损失
                self.optimizer_model.zero_grad()
                base_loss, metrics = self.contrastive_loss(anchor, positive, negatives)
                
                # 2. 生成边界样本并计算额外损失
                boundary_loss = torch.tensor(0.0, device=self.device)
                
                if epoch > 5:  # 前几个epoch不使用边界样本
                    with torch.no_grad():
                        # 标准化
                        std_normal = self.standardize_embeddings(
                            torch.cat([anchor, positive], dim=0)
                        )
                        std_anomaly = self.standardize_embeddings(
                            negatives.reshape(-1, embeddings.size(-1))
                        )
                        
                        # 生成边界样本
                        boundary_emb, weights = self.vae.generate_boundary_samples(
                            std_normal, std_anomaly,
                            num_samples=min(B, 32),
                            boundary_range=(0.45, 0.85)  # 难度适中的边界
                        )
                        
                        # 反标准化
                        boundary_emb = self.unstandardize_embeddings(boundary_emb)
                    
                    # 边界样本应该比负样本更接近原型,但仍然比正样本远
                    proto_norm = F.normalize(self.contrastive_loss.prototype, p=2, dim=1)
                    boundary_emb_norm = F.normalize(boundary_emb, p=2, dim=1)
                    anchor_norm = F.normalize(anchor, p=2, dim=1)
                    negatives_norm = F.normalize(negatives, p=2, dim=2)
                    
                    boundary_dist = torch.sum((boundary_emb_norm - proto_norm) ** 2, dim=1)
                    anchor_dist = torch.sum((anchor_norm - proto_norm) ** 2, dim=1)
                    
                    # 边界样本的理想距离: 介于正样本和负样本之间
                    # 根据权重动态调整目标距离
                    neg_dist = torch.sum(
                        (negatives_norm - proto_norm.unsqueeze(1)) ** 2, dim=2
                    ).mean(dim=1)
                    
                    target_dist = anchor_dist.mean() + weights.squeeze() * (
                       neg_dist.mean() - anchor_dist.mean()
                    )
                    
                    boundary_loss = F.mse_loss(
                        boundary_dist[:len(target_dist)], 
                        target_dist.detach()
                    )
                
                # 总损失
                total_loss = base_loss + 0.3 * boundary_loss
                
                if not torch.isnan(total_loss):
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer_finetune.step()
                    
                    epoch_metrics['total_loss'] += total_loss.item()
                    epoch_metrics['base_loss'] += base_loss.item()
                    epoch_metrics['boundary_loss'] += boundary_loss.item()
                    epoch_metrics['batches'] += 1
                    
                    pbar.set_postfix({
                        'total': f"{total_loss.item():.4f}",
                        'base': f"{base_loss.item():.4f}",
                        'boundary': f"{boundary_loss.item():.4f}"
                    })
            
            # Epoch 总结
            if epoch_metrics['batches'] > 0:
                avg_loss = epoch_metrics['total_loss'] / epoch_metrics['batches']
                avg_metrics = {k: v / epoch_metrics['batches'] 
                              for k, v in epoch_metrics.items() if k != 'batches'}
                
                logger.info(f"Phase 3 Epoch {epoch+1}/{num_epochs} | "
                          f"Total Loss: {avg_loss:.4f} | "
                          f"Base Loss: {avg_metrics['base_loss']:.4f} | "
                          f"Boundary Loss: {avg_metrics['boundary_loss']:.4f}")
                
                self.training_history['phase3'].append(avg_metrics)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint('phase3_final_best.pth', epoch, avg_loss)
                    logger.info(f"  ✓ 保存最佳模型 (loss: {best_loss:.4f})")
            
            scheduler_finetune.step()
            self._cleanup_memory()
        
        logger.info(f"阶段 3 完成,最佳损失: {best_loss:.4f}\n")
        return best_loss
    
    # ========================================================================
    # 工具函数
    # ========================================================================
    def save_checkpoint(self, filename, epoch, loss):
        """保存检查点"""
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'vae_state_dict': self.vae.state_dict(),
            'contrastive_loss_state': self.contrastive_loss.state_dict(),
            'prototype': self.contrastive_loss.prototype,
            'optimizer_model_state': self.optimizer_model.state_dict(),
            'optimizer_vae_state': self.optimizer_vae.state_dict(),
            'embedding_stats': self.embedding_stats,
            'loss': loss,
            'config': self.config
        }, save_path)
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        load_path = os.path.join(self.config['save_dir'], filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.contrastive_loss.load_state_dict(checkpoint['contrastive_loss_state'])
        self.embedding_stats = checkpoint['embedding_stats']
        
        logger.info(f"从 {load_path} 加载模型 (epoch {checkpoint['epoch']}, "
                   f"loss {checkpoint['loss']:.4f})")
        
        return checkpoint
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# 阶段 3: 异常检测器
# ============================================================================

class PathAnomalyDetector:
    """
    基于密度和距离的混合异常检测器
    """
    def __init__(self, model, contrastive_loss, vae, embedding_stats, device, config, prototype=None):
        self.model = model
        self.contrastive_loss = contrastive_loss
        self.vae = vae
        self.embedding_stats = embedding_stats
        self.device = device
        self.config = config
        
        # 从训练集计算正常样本的统计信息
        self.normal_stats = {
            'prototype': prototype,  # 原型向量
            'distances': None,  # 正常样本到原型的距离分布
            'percentiles': None,  # 距离的百分位数
            'reconstruction_errors': None,  # VAE重构误差分布
        }
    
    def extract_embeddings_from_dir(self, directory):
        """
        使用 InferenceDataset 从目录提取嵌入，按 contract_name 组织
        """
        logger.info(f"从目录 '{directory}' 提取嵌入...")
        
        try:
            dataset = InferenceDataset(preprocessed_data_dir=directory)
        except ValueError as e:
            logger.error(e)
            return {}

        dataloader = DataLoader(dataset, batch_size=8, collate_fn=inference_collate_fn, num_workers=4)
        
        embeddings_by_contract = defaultdict(list)
        
        self.model.eval()
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=f"提取特征"):
                if batch_data is None or batch_data[0] is None: 
                    continue
                
                padded_batch, contract_names = batch_data
                
                # 将数据移动到设备
                node_features = padded_batch['node_features'].to(self.device)
                node_types = padded_batch['node_types'].to(self.device)
                edge_types = padded_batch['edge_types'].to(self.device)

                embeddings = self.model(node_features, node_types, edge_types)

                # 将每个嵌入分配给其对应的合约名
                for i in range(len(contract_names)):
                    contract_name = contract_names[i]
                    embedding = embeddings[i].cpu().numpy()
                    embeddings_by_contract[contract_name].append(embedding)
        
        final_embeddings = {name: np.array(embs) for name, embs in embeddings_by_contract.items()}
        logger.info(f"特征提取完成，共 {len(final_embeddings)} 个合约。")
        
        # label_mapping = {}
        # with open('./data/vul_name.txt', 'r', encoding='utf-8') as f:
        #     for line in f:
        #         # 1. 去除行尾的换行符和首尾空白
        #         line = line.strip()
                
        #         # 跳过空行
        #         if not line:
        #             continue
                
        #         # 2. 按照 Tab ('\t') 进行分割
        #         # 如果你的txt里确实是tab，用 split('\t')
        #         # 如果你不确定是tab还是多个空格，可以用 split()，它会自动识别所有空白间隔
        #         parts = line.split(' ') 
        #         # print(parts)
                
        #         if len(parts) >= 2:
        #             name = parts[0].strip() # 获取地址
        #             label = parts[1].strip()   # 获取标签 (例如 SR)
        #             label_mapping[name] = label
        # print("正在聚合特征向量...")
        # # print(label_mapping)
        # contract_names = []
        # aggregated_vectors = []
        # labels = []
        # for name, embs in final_embeddings.items():
        #     if len(embs) == 0:
        #         continue
                
        #     # --- 核心操作：聚合 ---
        #     # 策略：取平均值 (Mean Pooling)。这是最常用的方法，代表合约的“平均行为特征”。
        #     # 如果你的异常通常只出现在某几步，也可以尝试 np.max (Max Pooling)
        #     contract_vec = np.mean(embs, axis=0) 
        #     contract_vec = np.mean(contract_vec, axis=0)
        #     print(contract_vec.shape)
            
        #     aggregated_vectors.append(contract_vec)
        #     contract_names.append(name)
        #     if 'normal' in name:
        #         labels.append('Normal')
        #     else:
        #         # print(name)
        #         labels.append(label_mapping.get(name, 'Unknown'))
        # X = np.array(aggregated_vectors)
        # y = np.array(labels)

        # print(f"聚合完成。样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

        # # 3. 降维 (t-SNE)
        # # t-SNE 比 PCA 更适合观察高维数据的局部聚类结构
        # print("正在进行 t-SNE 降维...")
        
        # # Perplexity 参数通常在 5 到 50 之间，样本少时需要调小
        # n_samples = X.shape[0]
        # perp = min(30, n_samples - 1) if n_samples > 1 else 1
        
        # tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
        # X_embedded = tsne.fit_transform(X)

        # # 4. 可视化 (Visualization)
        # plt.figure(figsize=(10, 8))
        
        # # 使用 Pandas 和 Seaborn 方便绘图
        # df_plot = pd.DataFrame({
        #     'x': X_embedded[:, 0],
        #     'y': X_embedded[:, 1],
        #     'Label': y
        # })
        
        # sns.scatterplot(data=df_plot, x='x', y='y', hue='Label', style='Label', s=100, palette='viridis')
        
        # plt.title('Visualization of Contract Embeddings by Vulnerability Type', fontsize=15)
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.show()
            
        return final_embeddings
    
    def calibrate(self, normal_data_dir):
        """
        使用正常训练数据校准检测器
        计算正常样本的距离和重构误差分布
        
        Args:
            normal_data_dir: 正常样本的预处理数据目录
        """
        logger.info("校准异常检测器...")
        
        # 提取所有正常样本的嵌入
        embeddings_dict = self.extract_embeddings_from_dir(normal_data_dir)
        if not embeddings_dict:
            raise RuntimeError(f"未能从 {normal_data_dir} 提取任何正常样本的嵌入，无法校准。")
            
        # 合并所有嵌入
        all_embeddings = np.concatenate(list(embeddings_dict.values()), axis=0)
        all_embeddings = torch.tensor(all_embeddings, dtype=torch.float32).to(self.device)
        
        
        # logger.info(f"prototype: {self.normal_stats['prototype']}")
        if self.normal_stats['prototype'] is None or torch.all(self.normal_stats['prototype'] == 0):
            logger.warning("⚠️  Prototype 未初始化或为零向量，从正常样本重新计算...")
            self.normal_stats['prototype'] = torch.mean(all_embeddings, dim=0)
            logger.info(f"Prototype: {self.normal_stats['prototype']}")
            logger.info(f"✓ Prototype 重新计算完成，shape: {self.normal_stats['prototype'].shape}")
        
        all_distances = []
        all_recon_errors = []
        
        self.vae.eval()
        with torch.no_grad():
            # 分批处理以避免内存问题
            batch_size = 256
            for i in range(0, len(all_embeddings), batch_size):
                batch_emb = all_embeddings[i:i+batch_size]
                
                # 1. 计算到原型的距离
                proto = self.normal_stats['prototype']
                distances = torch.sqrt(torch.sum((batch_emb - proto) ** 2, dim=1))
                all_distances.append(distances.cpu())
                
                # 2. 计算VAE重构误差
                std_emb = self.standardize_embeddings(batch_emb)
                recon, _, _ = self.vae(std_emb)
                recon_errors = torch.sum((recon - std_emb) ** 2, dim=1)
                all_recon_errors.append(recon_errors.cpu())
        
        # 统计
        all_distances = torch.cat(all_distances)
        all_recon_errors = torch.cat(all_recon_errors)
        
        self.normal_stats['distances'] = all_distances
        self.normal_stats['reconstruction_errors'] = all_recon_errors
        
        # 计算百分位数
        self.normal_stats['percentiles'] = {
            'distance': {
                50: torch.quantile(all_distances, 0.5).item(),
                90: torch.quantile(all_distances, 0.9).item(),
                95: torch.quantile(all_distances, 0.95).item(),
                99: torch.quantile(all_distances, 0.99).item(),
            },
            'recon_error': {
                50: torch.quantile(all_recon_errors, 0.5).item(),
                90: torch.quantile(all_recon_errors, 0.9).item(),
                95: torch.quantile(all_recon_errors, 0.95).item(),
                99: torch.quantile(all_recon_errors, 0.99).item(),
            }
        }
        
        logger.info("校准完成:")
        logger.info(f"  样本数量: {len(all_embeddings)}")
        logger.info(f"  距离分布: median={self.normal_stats['percentiles']['distance'][50]:.4f}, "
                   f"95%={self.normal_stats['percentiles']['distance'][95]:.4f}")
        logger.info(f"  重构误差: median={self.normal_stats['percentiles']['recon_error'][50]:.4f}, "
                   f"95%={self.normal_stats['percentiles']['recon_error'][95]:.4f}")
    
    def standardize_embeddings(self, embeddings):
        """标准化嵌入"""
        return (embeddings - self.embedding_stats['mean']) / self.embedding_stats['std']
     
    def compute_anomaly_scores(self, embeddings):
        """
        计算嵌入的异常分数（基于距离、重构误差和潜在空间密度的混合方法）
        
        这是核心的异常检测方法，完全不同于 Isolation Forest：
        1. Distance-based Score: 到正常原型的欧氏距离
        2. Reconstruction-based Score: VAE重构误差
        3. Latent Density Score: 潜在空间的密度估计
        
        Args:
            embeddings: torch.Tensor (N, D) 或 numpy.ndarray
        
        Returns:
            dict: 包含各种分数的字典
        """
        # 转换为 tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        self.vae.eval()
        with torch.no_grad():
            # 1. 计算到原型的距离 (Distance-based Score)
            # 正常样本应该靠近原型，异常样本会远离
            proto = self.normal_stats['prototype']
            distances = torch.sqrt(torch.sum((embeddings - proto) ** 2, dim=1))
            
            # 标准化距离分数 (相对于正常样本的95%分位数)
            distance_threshold = self.normal_stats['percentiles']['distance'][95]
            distance_scores = (distances / distance_threshold).clamp(0, 5)  # cap at 5x
            
            # 2. 计算VAE重构误差 (Reconstruction-based Score)
            # VAE在正常样本上训练，异常样本的重构误差会更大
            std_emb = self.standardize_embeddings(embeddings)
            recon, mu, logvar = self.vae(std_emb)
            recon_errors = torch.sum((recon - std_emb) ** 2, dim=1)
            
            # 标准化重构误差
            recon_threshold = self.normal_stats['percentiles']['recon_error'][95]
            recon_scores = (recon_errors / recon_threshold).clamp(0, 5)
            
            # 3. 计算潜在空间的异常分数 (Latent Density Score)
            # 正常样本应该接近标准正态分布 N(0,1)
            # KL散度越大，说明偏离标准正态分布越多
            latent_density = -0.5 * torch.sum(mu ** 2 + logvar.exp() - logvar - 1, dim=1)
            # logger.info(f"Latent density: mean={latent_density.mean().item():.4f}")
            latent_scores = torch.sigmoid(latent_density / 10)  # normalize to [0,1]
            
            # 4. 组合异常分数（加权平均）
            # 这是最终的异常分数，综合考虑了三个维度
            anomaly_scores = (
                self.config['weight_distance'] * distance_scores +
                self.config['weight_recon'] * recon_scores +
                self.config['weight_latent'] * latent_scores
            )
            
            return {
                'anomaly_scores': anomaly_scores.cpu().numpy(),
                'distance_scores': distance_scores.cpu().numpy(),
                'recon_scores': recon_scores.cpu().numpy(),
                'latent_scores': latent_scores.cpu().numpy(),
                'distances': distances.cpu().numpy(),
                'recon_errors': recon_errors.cpu().numpy(),
            }
    
    def detect_directory(self, test_data_dir):
        """
        对目录中的所有合约进行异常检测
        
        异常分数含义：
        - 分数越高越异常
        
        Args:
            test_data_dir: 测试样本的预处理数据目录
        
        Returns:
            dict: {contract_name: detection_results}
        """
        logger.info(f"开始异常检测: {test_data_dir}")
        
        # 提取嵌入
        embeddings_dict = self.extract_embeddings_from_dir(test_data_dir)
        # exit(0)
        
        # 对每个合约进行检测
        results_by_contract = {}
        for contract_name, embeddings in tqdm(embeddings_dict.items(), desc="检测合约"):
            if len(embeddings) == 0:
                continue
            
            # 计算异常分数
            scores = self.compute_anomaly_scores(embeddings)
            
            # 添加合约级别的统计
            results_by_contract[contract_name] = {
                **scores,
                'num_paths': len(embeddings),
                'mean_anomaly_score': np.mean(scores['anomaly_scores']),
                'max_anomaly_score': np.max(scores['anomaly_scores']),
                'min_anomaly_score': np.min(scores['anomaly_scores']),
                'max_distance': np.max(scores['distance_scores']), 
                'min_distance': np.min(scores['distance_scores']),
                'anomalous_ratio': np.sum(scores['anomaly_scores'] > 1.0) / len(embeddings),
            }
        
        return results_by_contract
    
    def evaluate(self, test_data_dir, label_fn=None, anomaly_ratio_threshold=0.01):
        """
        评估检测器性能
        
        Args:
            test_data_dir: 测试样本目录
            label_fn: 函数，接收 contract_name 返回标签 (0=正常, 1=异常)
                     默认: 如果名字包含 'normal' 则为正常，否则为异常
            anomaly_ratio_threshold: 异常路径比例阈值，超过此值判定为异常合约
        """
        logger.info("评估异常检测器...")
        
        # 默认标签函数
        if label_fn is None:
            label_fn = lambda name: 0 if 'normal' in name.lower() else 1
        
        # 检测
        results = self.detect_directory(test_data_dir)
        
        # 评估
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        all_scores = []
        all_labels = []
        
        logger.info("="*80)
        logger.info("合约级检测结果:")
        logger.info("="*80)
        
        for contract_name, result in results.items():
            true_label = label_fn(contract_name)
            # anomalous_ratio = result['anomalous_ratio']
            # is_anomalous = result['max_anomaly_score'] < 0.7
            is_anomalous = (1 - result['mean_anomaly_score']) > 0.413
            # is_anomalous = anomalous_ratio > anomaly_ratio_threshold
            
            all_scores.append(1 - result['mean_anomaly_score'])
            all_labels.append(true_label)
            
            # 统计
            if true_label == 1:  # 真实异常
                if is_anomalous:
                    true_positives += 1
                    status = "正确检测为漏洞 (TP)"
                else:
                    false_negatives += 1
                    status = "漏报为正常 (FN)"
            else:  # 真实正常
                if is_anomalous:
                    false_positives += 1
                    status = "误报为漏洞 (FP)"
                else:
                    true_negatives += 1
                    status = "正确识别为正常 (TN)"
            
            logger.info(f"合约: {contract_name:<40} | "
                       f"平均分数: {1 - result['mean_anomaly_score']:.4f} | "
                    #    f"最大分数: {result['max_anomaly_score']:.4f} | "
                    #    f"最小分数: {result['min_anomaly_score']:.4f} | "
                    #    f"最大距离: {result['max_distance']:.4f} | "
                    #    f"最小距离: {result['min_distance']:.4f} | "
                       f"状态: {status}")
        
        # 计算指标
        logger.info("="*80)
        logger.info("最终评估结果:")
        logger.info(f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算 AUC
        if len(set(all_labels)) > 1:  # 需要至少两个类别
            auc = roc_auc_score(all_labels, all_scores)
        else:
            auc = 0.0
            logger.warning("只有一个类别，无法计算 AUC")
        
        logger.info(f"召回率 (Recall): {recall:.2%}")
        logger.info(f"精确率 (Precision): {precision:.2%}")
        logger.info(f"误报率 (FPR): {fpr:.2%}")
        logger.info(f"F1-Score: {f1_score:.4f}")
        logger.info(f"AUC-ROC: {auc:.4f}")
        
        return {
            'auc': auc,
            'f1': f1_score,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tp': true_positives,
            'fp': false_positives,
            'tn': true_negatives,
            'fn': false_negatives,
            'results_by_contract': results
        }

# ============================================================================
# 阶段 4: 完整训练流程
# ============================================================================

def main():
    """完整的训练和评估流程"""
    
    # ========== 配置参数 ==========
    config = {
        # 模型参数
        'embed_dim': 512,
        'num_layers': 4,
        'nhead': 8,
        'dim_feedforward': 1024,
        'node_feature_dim': 384,
        'type_vector_dim': 6,
        'max_path_nodes': 8,
        
        # 训练参数
        'batch_size': 128,
        'num_workers': 8,
        
        # 三阶段训练周期
        'epochs_phase1': 50,   # 基础对比学习
        'epochs_phase2': 70,   # VAE训练
        'epochs_phase3': 100,  # 边界样本微调
        
        # 学习率
        'lr_model': 1e-4,
        'lr_vae': 1e-3,
        'weight_decay': 1e-5,
        
        # 损失函数参数
        'temperature': 0.07,           # 对比学习温度
        'margin':2.5,
        'compact_weight': 0.6,
        'prototype_momentum': 0.99,   # 原型更新动量
        
        # VAE参数
        'latent_dim': 128,
        'vae_hidden_dim': 512,
        'vae_beta': 0.1,             # KL散度权重
        
        # 异常检测权重
        'weight_distance': 0.4,      # 距离分数权重
        'weight_recon': 0.4,         # 重构误差权重
        'weight_latent': 0.2,        # 潜在空间密度权重
        
        # 路径
        'preprocessed_data_dir': 'preprocessed_samples',
        'save_dir': './pretrained_model',
        'test_samples_dir': 'preprocessed_samples_vul_full',
        'normal_samples_dir': 'preprocessed_samples_normal',
        'checkpoint_path': './pretrained_model/final_model.pth',#FineTuned_final_model  phase3_final_best phase1_best
    }
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # ========== 设备设置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # ========== 加载数据 ==========
    logger.info("加载数据集...")
    
    train_dataset = ContrastivePathDataset(config['preprocessed_data_dir'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        collate_fn=contrastive_collate_fn,
        num_workers=config['num_workers'],
        shuffle=True,
        pin_memory=True
    )
    
    logger.info("\n" + "="*60)
    logger.info("加载预训练模型")
    logger.info("="*60 + "\n")
    
    # 加载完整的checkpoint
    logger.info(f"加载 checkpoint: {config['checkpoint_path']}")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    
    # 从checkpoint中获取配置（如果保存了的话）
    saved_config = checkpoint.get('config', {})
    logger.info(f"训练轮次: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"训练损失: {checkpoint.get('loss', 'unknown')}")
    
    model_max_seq_len = 2 * config['max_path_nodes'] - 1
    
    logger.info("初始化 CGT 模型...")
    model = CGT_Model(
        node_feature_dim=config['node_feature_dim'],
        embed_dim=config['embed_dim'],
        max_seq_len=model_max_seq_len,
        type_vector_dim=config['type_vector_dim'],
        num_layers=config['num_layers'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        num_classes=None
    ).to(device)
    
    # ========== 创建训练器 ==========
    trainer = AnomalyDetectionTrainer(model, config['embed_dim'], device, config)
    
    # ========== 三阶段训练 ==========
    logger.info("\n" + "="*60)
    logger.info("开始三阶段训练流程")
    logger.info("="*60 + "\n")
    
    # 阶段1: 基础对比学习
    phase1_loss = trainer.train_phase1_contrastive(
        train_loader, 
        config['epochs_phase1']
    )
    
    # 阶段2: VAE训练
    phase2_loss = trainer.train_phase2_vae(
        train_loader,
        config['epochs_phase2']
    )
    
    # 阶段3: 边界样本微调
    phase3_loss = trainer.train_phase3_finetune(
        train_loader,
        config['epochs_phase3']
    )
    
    logger.info("\n" + "="*60)
    logger.info("训练完成!")
    logger.info("="*60)
    logger.info(f"阶段1最佳损失: {phase1_loss:.4f}")
    logger.info(f"阶段2最佳损失: {phase2_loss:.4f}")
    logger.info(f"阶段3最佳损失: {phase3_loss:.4f}")
    # ========== 保存最终模型 ==========
    logger.info("\n保存最终模型...")
    trainer.save_checkpoint('final_model.pth', 
                           config['epochs_phase1'] + config['epochs_phase2'] + config['epochs_phase3'],
                           phase3_loss)
    
    pass
    #异常检测器部分
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    logger.info("✓ CGT 模型加载完成")
    
    # 创建并加载 VAE
    logger.info("初始化 VAE 模型...")
    vae = BoundaryVAE(
        input_dim=config['embed_dim'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['vae_hidden_dim']
    ).to(device)
    
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae.eval()
    logger.info("✓ VAE 模型加载完成")
    
    logger.info("重建对比损失...")
    contrastive_loss = PrototypicalContrastiveLoss(
        margin=saved_config.get('margin', 2.0),
        compact_weight=saved_config.get('compact_weight', 0.5),
        prototype_momentum=saved_config.get('prototype_momentum', 0.99)
    ).to(device)

    # 加载对比损失的状态（包含 prototype）
    contrastive_loss.load_state_dict(checkpoint['contrastive_loss_state'])
    logger.info("✓ 对比损失加载完成")
    
    prototype = checkpoint['prototype'] # checkpoint['prototype']
    logger.info(f"Prototype 范数: {prototype}")
    
    # if contrastive_loss.prototype is None or contrastive_loss.prototype.norm() < 1e-6:
    #     logger.error("加载后的 Prototype 仍然是零！请检查 save_checkpoint 逻辑！")
    #     # 作为备用方案，可以从单独保存的字段加载
    #     if 'prototype' in checkpoint:
    #         contrastive_loss.prototype = checkpoint['prototype'].to(device)
    #         logger.info("已从独立的 'prototype' 字段恢复。")
    #     else:
    #         logger.info(f"✓ Prototype 加载成功，范数: {contrastive_loss.prototype.norm().item():.4f}")
                
    # 加载嵌入统计信息
    embedding_stats = checkpoint['embedding_stats']
    logger.info("✓ 嵌入统计信息加载完成")
    
    
    # ========== 创建异常检测器 ==========
    logger.info("\n" + "="*60)
    logger.info("创建异常检测器")
    logger.info("="*60 + "\n")
    detector = PathAnomalyDetector(
        model=model,
        contrastive_loss=contrastive_loss,
        vae=vae,
        embedding_stats=embedding_stats,
        device=device,
        config=config,
        prototype=prototype
    )
    # '''
    # ========== 校准检测器 ==========
    logger.info("\n" + "="*60)
    logger.info("使用正常样本校准检测器")
    logger.info("="*60 + "\n")
    detector.calibrate(config['normal_samples_dir'])
    
    # ========== 异常检测评估 ==========
    logger.info("\n" + "="*60)
    logger.info("开始异常检测评估")
    logger.info("="*60 + "\n")
    
    # 定义标签函数：根据合约名称判断是否异常
    # 如果名称包含 'normal' 则为正常(0)，否则为异常(1)
    def label_fn(contract_name):
        return 0 if 'normal' in contract_name.lower() else 1
    
    eval_results = detector.evaluate(
        test_data_dir=config['test_samples_dir'],
        label_fn=label_fn,
        anomaly_ratio_threshold=0.05
    )
    
    # ========== 保存检测结果 ==========
    logger.info("\n" + "="*60)
    logger.info("保存检测结果")
    logger.info("="*60 + "\n")
    
    results_save_path = './detection_results.pkl'
    with open(results_save_path, 'wb') as f:
        pickle.dump(eval_results, f)
    logger.info(f"检测结果已保存至: {results_save_path}")
    
    # ========== 总结 ==========
    logger.info("\n" + "="*60)
    logger.info("检测完成!")
    logger.info("="*60)
    logger.info(f"AUC-ROC: {eval_results['auc']:.4f}")
    logger.info(f"F1-Score: {eval_results['f1']:.4f}")
    logger.info(f"Precision: {eval_results['precision']:.4f}")
    logger.info(f"Recall: {eval_results['recall']:.4f}")
    logger.info(f"FPR: {eval_results['fpr']:.4f}")
    logger.info("="*60 + "\n")

    trainer = AnomalyDetectionTrainer(model, config['embed_dim'], device, config)
    return trainer, detector

if __name__ == "__main__":
    # 运行完整训练流程
    trainer, detector = main()
    