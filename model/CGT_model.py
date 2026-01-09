import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. 输入嵌入层
# ==============================================================================

class SpecializedInputEmbedding(nn.Module):
    """
    专门为控制流图 (Control Flow Graph, CFG) 序列设计的输入嵌入层。
    它将多种特征（Token本身、类型、位置、深度、分支）融合成一个综合的向量表示。
    ## 为了统一输入序列的长度，可能需要用[PAD]标志位进行填充。那么这时对应的元素类型等应该标记为什么 ##
    """
    def __init__(self, input_dim: int, embed_dim: int, max_seq_len: int, type_vector_dim: int):
        """
        初始化函数

        参数:
            vocab_size (int): 词汇表大小 (节点/边的种类数量)。
            embed_dim (int): 嵌入向量的维度。
            max_seq_len (int): 支持的最大序列长度。
            max_depth (int): 支持的最大CFG深度。
        """
        super().__init__()
        self.embed_dim = embed_dim
        # self.node_type_pad_idx = num_node_types  # 假设节点类型有 num_node_types 种，使用最后一个索引作为 PAD
        # self.edge_type_pad_idx = num_edge_types# + num_node_types  # 假设边类型有 num_edge_types 种，使用最后一个索引作为 PAD
        
        # Token 嵌入 - 为每个节点和边提供独立的向量表示
        self.token_proj_layer = nn.Linear(input_dim, embed_dim)
        self.edge_type_proj_layer = nn.Linear(type_vector_dim, embed_dim)

        self.type_proj_layer = nn.Linear(type_vector_dim, embed_dim)
        # self.node_type_embedding = nn.Embedding(num_node_types, embed_dim)  
        
        # 类型嵌入 - 区分不同类型的元素
        # 假设 0: 普通节点, 1: 边, 2: 控制节点 (如if/while), 3: 特殊Token (如[CLS], [SEP])
        # self.edge_type_embedding = nn.Embedding(num_edge_types + 1, embed_dim, padding_idx=self.edge_type_pad_idx)
        
        # 路径感知位置编码 - 结合了序列中的顺序和图中的深度
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)  # 序列中的绝对位置
        
    def forward(self, node_features, node_types, edge_types):
        batch_size, num_nodes, _ = node_features.shape
        num_edges = edge_types.shape[1]
        assert num_edges == num_nodes - 1
        seq_len = num_nodes + num_edges  # N1,E1,N2,E2,...,Nn 的总长度
        
        node_embeds = self.token_proj_layer(node_features)
        edge_embeds = self.edge_type_proj_layer(edge_types)
        
        token_embeds = torch.zeros(batch_size, seq_len, self.embed_dim, device=node_features.device)
        token_embeds[:, 0::2, :] = node_embeds
        token_embeds[:, 1::2, :] = edge_embeds
        
        type_vectors = torch.zeros(batch_size, seq_len, edge_types.shape[-1], device=node_features.device)
        type_vectors[:, 0::2, :] = node_types
        type_vectors[:, 1::2, :] = edge_types
        type_embeds = self.type_proj_layer(type_vectors)
        # 3. 获取位置嵌入 (如果未提供，则自动创建从 0 到 seq_len-1 的位置)
        positions = torch.arange(seq_len, device=node_features.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        

        # 4. 将所有嵌入相加，形成最终的综合嵌入
        embeddings = token_embeds + type_embeds + pos_embeds
        
        return embeddings, token_embeds

# ==============================================================================
# 2. CGT 编码器核心模块
# ==============================================================================
class StructuralAwareContextModule(nn.Module):
    """
    结构感知上下文模块。边的关系偏置现在是单向的，以正确反映图的有向性。
    """
    def __init__(self, d_model, nhead, max_seq_len, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_seq_len = max_seq_len

        # 您之前的代码已经简化得很好，不再需要 edge_type_ids
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relative_pos_embedding = nn.Embedding(2 * max_seq_len - 1, self.nhead)
        self.edge_proj = nn.Linear(d_model, self.nhead, bias=False)

    def forward(self, x, token_types, token_embeds, mask=None):
        # 注意：这里的 token_types 参数已经不再需要了，因为我们直接通过位置判断边
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        content_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        pos = torch.arange(seq_len, device=x.device)
        relative_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        relative_pos = relative_pos + self.max_seq_len - 1
        pos_bias_raw = self.relative_pos_embedding(relative_pos)
        pos_bias = pos_bias_raw.unsqueeze(0).permute(0, 3, 1, 2)
        
        # --- 边的关系偏置计算 ---
        edge_bias_matrix = torch.zeros(batch_size, self.nhead, seq_len, seq_len, device=x.device)

        # 1. 找到所有边的位置 (奇数索引)
        # 注意：这里 seq_len 可能是奇数或偶数
        edge_positions = torch.arange(1, seq_len, 2, device=x.device)
        
        if edge_positions.numel() > 0:
            # 2. 获取所有边的嵌入向量
            # token_embeds shape: [batch_size, seq_len, d_model]
            edge_embeds = token_embeds[:, edge_positions, :] # [batch_size, num_edges, d_model]
            
            # 3. 将边的嵌入投影为偏置值
            # edge_bias_values shape: [batch_size, num_edges, nhead]
            edge_bias_values = self.edge_proj(edge_embeds)

            # 4. 【核心修正】将偏置值精确地、单向地填充到矩阵中
            # 对于位于 edge_pos 的边，它定义了 (edge_pos - 1) -> (edge_pos + 1) 的关系
            src_node_positions = edge_positions - 1
            dst_node_positions = edge_positions + 1
            
            # 使用 advanced indexing 进行高效的、并行的填充
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
            
            # edge_bias_matrix[batch_indices, :, src_node_positions, dst_node_positions] = ...
            # 这种直接赋值在 PyTorch 中有点棘手，我们用一个循环来保证清晰和正确
            for i in range(edge_positions.numel()):
                src_idx = src_node_positions[i]
                dst_idx = dst_node_positions[i]
                # bias_for_this_edge shape: [batch_size, nhead]
                bias_for_this_edge = edge_bias_values[:, i, :]
                edge_bias_matrix[:, :, src_idx, dst_idx] = bias_for_this_edge

        # --- 核心修正结束 ---

        total_scores = content_scores + pos_bias + edge_bias_matrix
        
        if mask is not None:
            # mask 的形状通常是 [batch_size, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
            total_scores = total_scores.masked_fill(mask == 0, float('-inf'))
            
        attn_probs = F.softmax(total_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(output)
    
# class StructuralAwareContextModule(nn.Module): ##记得修改成改进后的代码
#     """
#     结构感知上下文模块 (Structural-Aware Context Module)
#     支持一个包含多种边类型的列表。
#     """
#     def __init__(self, d_model, nhead, max_seq_len, edge_type_ids: list, dropout=0.1):
#         super().__init__()
#         assert d_model % nhead == 0
#         self.d_model = d_model
#         self.nhead = nhead
#         self.head_dim = d_model // nhead
#         self.max_seq_len = max_seq_len

#         # # 【改动】将边的ID列表存储下来
#         # self.edge_type_ids = edge_type_ids

#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.relative_pos_embedding = nn.Embedding(2 * max_seq_len - 1, self.nhead)
#         self.edge_proj = nn.Linear(d_model, self.nhead, bias=False)

#     def forward(self, x, token_types, token_embeds, mask=None):
#         batch_size, seq_len, _ = x.shape

#         q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
#         k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
#         v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
#         content_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

#         pos = torch.arange(seq_len, device=x.device)
#         relative_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
#         relative_pos = relative_pos + self.max_seq_len - 1
#         pos_bias = self.relative_pos_embedding(relative_pos).unsqueeze(0).permute(0, 3, 1, 2)
        
#         # --- 【核心改动逻辑】 ---
#         # 1. 创建一个布尔掩码，标记出所有边的位置
#         is_edge_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
#         is_edge_mask[1::2] = True
        
#         # 2. 使用这个掩码找到所有边的索引和嵌入
#         edge_indices = is_edge_mask.unsqueeze(0).expand(batch_size, -1).nonzero(as_tuple=False) # 使用 as_tuple=False 获取 (N, 2) 的索引
        
#         edge_bias_matrix = torch.zeros(batch_size, self.nhead, seq_len, seq_len, device=x.device)

#         if edge_indices.numel() > 0: # 只有当序列中存在边时才执行
#             # edge_indices 是一个 [N, 2] 的张量, N是batch中所有边的总数, [:, 0]是batch索引, [:, 1]是seq索引
#             edge_batch_idx = edge_indices[:, 0]
#             edge_seq_idx = edge_indices[:, 1]
            
#             edge_embeds = token_embeds[edge_batch_idx, edge_seq_idx] # 获取所有边的嵌入向量
#             edge_bias_values = self.edge_proj(edge_embeds) # (num_total_edges, nhead)

#             # 遍历所有找到的边，并填充偏置矩阵
#             for i in range(edge_indices.shape[0]):
#                 b = edge_batch_idx[i]
#                 edge_pos = edge_seq_idx[i]
                
#                 if edge_pos > 0 and edge_pos < seq_len - 1:
#                     src_node_pos, dst_node_pos = edge_pos - 1, edge_pos + 1
#                     bias_val_for_this_edge = edge_bias_values[i] # (nhead,)
#                     edge_bias_matrix[b, :, src_node_pos, dst_node_pos] = bias_val_for_this_edge
#                     edge_bias_matrix[b, :, dst_node_pos, src_node_pos] = bias_val_for_this_edge
#         # --- 核心改动结束 ---
#         # print(f"content_scores: {content_scores.shape}, pos_bias: {pos_bias.shape}, edge_bias_matrix: {edge_bias_matrix.shape}")
#         total_scores = content_scores + pos_bias + edge_bias_matrix
        
#         if mask is not None:
#             total_scores = total_scores.masked_fill(mask == 0, float('-inf'))
            
#         attn_probs = F.softmax(total_scores, dim=-1)
#         attn_probs = self.dropout(attn_probs)
        
#         output = torch.matmul(attn_probs, v)
#         output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
#         return self.out_proj(output)
class CausalGateUnit(nn.Module):
    """
    双探针因果门控单元 (Dual-Probe Causal Gate Unit, DPCGU)
    
    功能: 
    并行地执行两种因果验证逻辑：
    1. 前提探针 (Prerequisite Probe): 寻找必须存在的历史条件 (越高越好)。
    2. 风险探针 (Hazard Probe): 警惕不应出现的历史条件 (越低越好)。
    """
    def __init__(self, d_model):
        super(CausalGateUnit, self).__init__()
        self.d_model = d_model
        
        # --- 1. 双探针生成层 ---
        # 生成前提探针 P_pre
        self.pre_probe_layer = nn.Linear(d_model, d_model)
        # 生成风险探针 P_haz
        self.haz_probe_layer = nn.Linear(d_model, d_model)
        
        # --- 2. 双门控生成层 ---
        # 生成两个标量门: [g_pre, g_haz]
        # g_pre: 当前节点对前置条件的依赖程度
        # g_haz: 当前节点对风险条件的敏感程度
        self.gate_layer = nn.Linear(d_model, 2)
        
        # --- 3. 信号编码器 ---
        # 输入维度变为 4: [g_pre, score_pre, g_haz, score_haz]
        # 用于将标量信号映射回高维特征空间
        self.signal_encoder = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # --- 4. 最终融合层 ---
        # 将原始特征 x 与 逻辑特征 logic_feat 融合
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x, q, k):
        """
        x: 当前层的输入特征 [Batch, Seq, D]
        q: Query 向量 (通常来自 x) [Batch, Seq, D]
        k: Key 向量 (通常来自 x，用于历史扫描) [Batch, Seq, D]
        """
        batch_size, seq_len, _ = q.size()
        
        # ------------------------------------------------------------------
        # Step 1: 生成探针 (Probes) 和 门控 (Gates)
        # ------------------------------------------------------------------
        p_pre = self.pre_probe_layer(q)  # [B, L, D]
        p_haz = self.haz_probe_layer(q)  # [B, L, D]
        
        # 生成门控信号 [B, L, 2] -> split -> g_pre, g_haz
        gates = torch.sigmoid(self.gate_layer(q)) 
        g_pre = gates[:, :, 0:1]  # [B, L, 1]
        g_haz = gates[:, :, 1:2]  # [B, L, 1]
        
        # ------------------------------------------------------------------
        # Step 2: 历史扫描 (Historical Scanning) - 并行计算
        # ------------------------------------------------------------------
        
        k_transpose = k.transpose(-2, -1) # [B, D, L]
        
        # 计算前提相似度矩阵
        sim_pre = torch.matmul(p_pre, k_transpose) / math.sqrt(self.d_model) # [B, L, L]
        # 计算风险相似度矩阵
        sim_haz = torch.matmul(p_haz, k_transpose) / math.sqrt(self.d_model) # [B, L, L]
        
        # ------------------------------------------------------------------
        # Step 3: 因果掩码 (Causal Masking)
        # ------------------------------------------------------------------
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device),
            diagonal=-1 
        ) # [L, L]
        
        # 将未来和当前位置设为负无穷
        mask_expanded = causal_mask.unsqueeze(0) == 0 # [1, L, L]
        sim_pre = sim_pre.masked_fill(mask_expanded, float('-inf'))
        sim_haz = sim_haz.masked_fill(mask_expanded, float('-inf'))
        
        # ------------------------------------------------------------------
        # Step 4: 获取最佳匹配分数 (Max Pooling)
        # ------------------------------------------------------------------
        score_pre, _ = torch.max(sim_pre, dim=-1, keepdim=True) # [B, L, 1]
        score_haz, _ = torch.max(sim_haz, dim=-1, keepdim=True) # [B, L, 1]
        
        zero_tensor = torch.zeros_like(score_pre)
        score_pre = torch.where(torch.isinf(score_pre), zero_tensor, score_pre)
        score_haz = torch.where(torch.isinf(score_haz), zero_tensor, score_haz)
        
        # ------------------------------------------------------------------
        # Step 5: 信号编码与逻辑特征生成
        # ------------------------------------------------------------------
        raw_signals = torch.cat([g_pre, score_pre, g_haz, score_haz], dim=-1)
        
        # 映射回高维空间，得到 h_logic
        h_logic = self.signal_encoder(raw_signals) # [B, L, D]
        
        return h_logic
    
# class CausalGateUnit(nn.Module):
#     """
#     因果门控单元 (Causal Gate Unit, CGU)
#     功能: 显式地建模和验证“A之前必须存在B/C”这类硬性时序逻辑。
#     """
#     def __init__(self, d_model):
#         super(CausalGateUnit, self).__init__()
#         self.d_model = d_model
#         self.probe_layer = nn.Linear(d_model, d_model)
#         self.gate_layer = nn.Linear(d_model, 1)
#         self.causal_score_transform = nn.Linear(1, d_model)

#     def forward(self, q, k):
#         batch_size, seq_len, _ = q.size()
        
#         # 计算前置条件探测向量
#         precondition_probes = self.probe_layer(q)  # [B, L, D]
#         requirement_gates = torch.sigmoid(self.gate_layer(q))  # [B, L, 1]
        
#         # 【关键优化】一次性计算所有相似度
#         # similarity: [B, L, L] - 每个位置与所有历史位置的相似度
#         similarity = torch.matmul(
#             precondition_probes, 
#             k.transpose(-2, -1)
#         ) / math.sqrt(self.d_model)
        
#         # 创建因果掩码：只保留当前位置之前的历史
#         causal_mask = torch.tril(
#             torch.ones(seq_len, seq_len, device=q.device),
#             diagonal=-1  # 不包括对角线（当前位置）
#         )  # [L, L]
        
#         # 应用掩码（将未来位置设为极小值）
#         similarity = similarity.masked_fill(
#             causal_mask.unsqueeze(0) == 0, 
#             float('-inf')
#         )
        
#         # 对每个位置取历史中的最大相似度
#         # 第一个位置没有历史，会得到 -inf，需要特殊处理
#         causal_scores, _ = torch.max(similarity, dim=-1, keepdim=True)  # [B, L, 1]
        
#         # 处理第一个位置（没有历史记录）
#         causal_scores[:, 0, :] = 0.0
        
#         # 将 -inf 替换为 0（避免 tanh 计算问题）
#         causal_scores = causal_scores.masked_fill(
#             torch.isinf(causal_scores), 
#             0.0
#         )
        
#         # 最终因果信号
#         causal_signal = requirement_gates * torch.tanh(
#             self.causal_score_transform(causal_scores)
#         )
        
#         return causal_signal

class FusionModule(nn.Module):
    """
    融合模块 (Fusion Module)
    功能: 将通用的上下文信息和专门的因果逻辑信息进行有效结合。
    """
    def __init__(self, d_model):
        super(FusionModule, self).__init__()
        self.fusion_gate_layer = nn.Linear(d_model * 2, d_model)

    def forward(self, output_context, output_causal):
        combined_output = torch.cat([output_context, output_causal], dim=-1)
        gate_fusion = torch.sigmoid(self.fusion_gate_layer(combined_output))
        output_fused = gate_fusion * output_context + (1 - gate_fusion) * output_causal
        return output_fused

class CGTEncoderLayer(nn.Module):
    """
    CGT 编码器层 (The CGT Encoder Layer)
    这是模型的核心，它整合了三个并行子模块，并加入了标准的层归一化和前馈网络。
    """
    def __init__(self, d_model, nhead, max_seq_len, dim_feedforward=2048, dropout=0.1):
        super(CGTEncoderLayer, self).__init__()
        self.structural_context = StructuralAwareContextModule(d_model, nhead, max_seq_len, dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.causal_gate_unit = CausalGateUnit(d_model)
        self.fusion_module = FusionModule(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, token_types, token_embeds, src_mask=None):
        output_context = self.structural_context(x, token_types, token_embeds, mask=src_mask)
        q = self.q_proj(x)
        k = self.k_proj(x)
        output_causal = self.causal_gate_unit(q, k)
        output_fused = self.fusion_module(output_context, output_causal)
        x = self.norm1(x + self.dropout1(output_fused))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x

# ==============================================================================
# 3. 完整的 CGT-CFG 模型
# ==============================================================================

class CGT_Model(nn.Module):
    """
    一个完整的，基于 CGT 编码器和专用 CFG 嵌入的模型。
    """
    def __init__(self, node_feature_dim, embed_dim, max_seq_len, type_vector_dim, num_layers, nhead,
                 dim_feedforward, num_classes=None, dropout=0.1):
        """
        初始化函数

        参数:
            vocab_size (int): 词汇表大小。
            embed_dim (int): 嵌入和模型的维度 (d_model)。
            max_seq_len (int): 最大序列长度。
            max_depth (int): 最大CFG深度。
            num_layers (int): CGT 编码器层的数量。
            nhead (int): 多头注意力的头数。
            dim_feedforward (int): 前馈网络的隐藏层维度。
            num_classes (int): 最终分类任务的类别数。
            dropout (float): Dropout 比例。
        """
        super().__init__()
        
        # 1. 输入嵌入层
        self.embedding = SpecializedInputEmbedding(
            input_dim=node_feature_dim,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            type_vector_dim=type_vector_dim,
        )
        
        # 2. 堆叠的 CGT 编码器层
        self.encoder_layers = nn.ModuleList([
            CGTEncoderLayer(
                d_model=embed_dim, nhead=nhead, max_seq_len=max_seq_len, 
                dim_feedforward=dim_feedforward, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 3. 输出层 (例如，用于一个分类任务)
        # 这里我们简单地用第一个 token ([CLS] token) 的输出来做分类
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, num_classes)
            )

        else:
            self.classifier = None
        
    def forward(self, node_features, node_types, edge_types, src_mask=None):
        """
        完整模型的前向传播

        参数:
            tokens, token_types, depths, branches: 传递给嵌入层的参数。
            src_mask (Tensor, optional): 注意力掩码。

        返回:
            Tensor: 分类任务的 logits，形状 (batch_size, num_classes)。
        """
        # 步骤 1: 通过嵌入层，将输入的 ID 转换为特征向量
        x, token_embeds = self.embedding(node_features, node_types, edge_types)
        
        # 步骤 2: 依次通过每一个 CGT 编码器层
        for layer in self.encoder_layers:
            x = layer(x, token_types=None, token_embeds=token_embeds, src_mask=src_mask)
        
        cls_token_output = x[:, 0, :]
        # 步骤 3: 使用序列的第一个位置（通常是 [CLS] 标志位）的输出来进行分类
        if self.classifier:
            logits = self.classifier(cls_token_output)
            return logits
        
        return cls_token_output
    """在进行无标签的对比学习时，对比的特征应该是这里的logits还是cls_token_output？"""
    """一定要使用cls_token_output, 它是经过多层 CGT 编码器深度处理后，代表整个输入序列的最终聚合向量。"""

# ==============================================================================
# 4. 示例用法
# ==============================================================================

# if __name__ == '__main__':
#     # --- 模型超参数定义 ---
#     VOCAB_SIZE = 1000      # 假设词汇表有1000个 token
#     EMBED_DIM = 256        # 嵌入维度，也是模型的 d_model
#     MAX_SEQ_LEN = 128      # 最大序列长度
#     MAX_DEPTH = 16         # CFG 最大深度
#     NUM_LAYERS = 4         # 4层 CGT Encoder
#     NHEAD = 8              # 8个注意力头
#     DIM_FEEDFORWARD = 1024 # FFN 隐藏层维度
#     NUM_CLASSES = 10       # 最终分类任务的类别数
#     BATCH_SIZE = 4         # 批量大小
#     NODE_TYPES_IDS = [0, 1, 2, 3, 4, 5] # 5 是 PAD
#     EDGE_TYPE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 假设边的类型ID是0到9
#     NODE_FEATURE_DIM = 100 # 假设输入的节点特征维度是100
#     TYPE_VECTOR_DIM = 5    # 假设节点与边类型向量的维度是5 (多标签)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     '''
#     edgetype2id = {'JUMPI-True': 0, 'JUMPI-False': 1, 'JUMP': 2, 'Sequence': 3, 'CALLPRIVATE': 4, 'RETURNPRIVATE': 5,
#                    'CALL': 6, 'STATICCALL': 7, 'DELEGATECALL': 8, 'RETURN': 9}

#     nodetype2id = {
#         "Asset-Transfer": 0,
#         "Invocation-Node": 1,
#         "Compute-Node": 2,
#         "Information-Node": 3,
#         "Common-Node": 4,
#     }
#     '''

#     NUM_NODE_TYPES = 5
#     NUM_EDGE_TYPES = 10 # 嵌入表大小要能容纳最大ID

#     # --- 实例化完整模型 ---
#     model = CGT_Model(
#         node_feature_dim=NODE_FEATURE_DIM, embed_dim=EMBED_DIM, max_seq_len=MAX_SEQ_LEN,
#         type_vector_dim=TYPE_VECTOR_DIM, num_layers=NUM_LAYERS, nhead=NHEAD,
#         dim_feedforward=DIM_FEEDFORWARD, num_classes=NUM_CLASSES
#     )
#     model = model.to(device)
    
#     print("="*40)
#     print("CGT-CFG 完整模型结构")
#     print("="*40)
#     print(model)
    
#     # --- 创建模拟的输入数据 (根据新规则重写) ---
#     num_nodes, num_edges = 60, 59
#     mock_node_features = torch.randn(BATCH_SIZE, num_nodes, NODE_FEATURE_DIM).to(device)
    
#     # --- 根据新规则生成类型向量 ---
    
#     # 1. 生成节点类型向量
#     # 规则：第4位必须是0
#     mock_node_type_vectors = torch.randint(0, 2, (BATCH_SIZE, num_nodes, TYPE_VECTOR_DIM)).float().to(device)
#     mock_node_type_vectors[:, :, 4] = 0 # 强制最高位为 0
    
#     # 处理节点PAD情况：如果一个向量是全0，它就是PAD向量，我们接受它。
#     # 如果不是全0，则它是一个合法的节点类型。这个逻辑不需要额外处理，因为全0已经是PAD的表示。
#     print("Generated raw node types. A few examples:\n", mock_node_type_vectors[0, :3])

#     # 2. 生成边类型向量
#     # 规则：第4位必须是1
#     mock_edge_type_vectors = torch.randint(0, 2, (BATCH_SIZE, num_edges, TYPE_VECTOR_DIM)).float().to(device)
#     mock_edge_type_vectors[:, :, 4] = 1 # 强制最高位为 1

#     # 处理边PAD情况：根据您的定义，边的PAD不是全0，而是 [1,0,0,0,0]
#     # 我们之前的逻辑是防止生成PAD，但在真实场景中可能需要PAD。
#     # 这里我们假设模拟数据不包含PAD边，所以如果恰好生成了[x,x,x,x,1]之外的向量，
#     # 我们确保它不是全0（虽然概率极低）
#     print("Generated raw edge types. A few examples:\n", mock_edge_type_vectors[0, :3])

#     # --- 模型前向传播 ---
#     output_logits = model(
#         node_features=mock_node_features, node_types=mock_node_type_vectors, edge_types=mock_edge_type_vectors
#     )
    
#     # --- 验证输出 ---
#     print("\n" + "="*40)
#     print("模型运行验证")
#     print("="*40)
#     print(f"输入 Node Features 形状: {mock_node_features.shape}")
#     print(f"输入 Edge Types 形状:   {mock_edge_type_vectors.shape}")
#     # print(f"输入 Branches 形状:  {mock_branches.shape}")
#     print(f"\n最终输出 Logits 形状: {output_logits.shape}")
    
#     # 验证输出形状是否符合预期 (batch_size, num_classes)
#     assert output_logits.shape == (BATCH_SIZE, NUM_CLASSES)
#     print("\n模型成功运行，输入输出形状匹配！")