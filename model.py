import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import GATConv, GCNConv
import math
from Series_Mix import MultiScaleTrendMixing, TemporalLSTM, TemporalCausalConv
from Series_Decom import Temporal_Decomposition_DWT_GPU
from DirectedGCNConv import DirectedGCNConv

class DWTEnhancedSTGCN(nn.Module):
    """增强的STGCN:使用DWT分解,高频走因果卷积加普通图卷积(不做对抗),低频走LSTM加因果图卷积(做对抗），最后通过注意力机制融合并再次对抗"""
    def __init__(self, input_dim, lstm_hidden, temporal_dim, gcn_hidden, output_dim, args=None):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.temporal_dim = temporal_dim
        self.gcn_hidden = gcn_hidden
        self.output_dim = output_dim
        self.args = args

        # 总的残差连接 - 从原始输入信号到输出特征
        self.global_residual_proj = nn.Linear(12, output_dim)  # 用于原始输入的全局残差投影
        self.global_residual_ln = nn.LayerNorm(output_dim)  # 对全局残差做归一化
        self.high_residual_proj = nn.Linear(temporal_dim, output_dim)  # 高频分支残差投影
        self.low_residual_proj = nn.Linear(temporal_dim, output_dim)   # 低频分支残差投影

        # 分支出口的归一化层
        self.high_freq_norm = nn.LayerNorm(output_dim)
        self.low_freq_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, edge_index_or_adj, causal_edge_index_or_adj, domain_label=None):
        """
        Args:
            x: 输入特征 [B, T, N] 或 [B, T, N, 2]（最后一维为 [low, high]）
            edge_index_or_adj: 基础图边索引或邻接矩阵
            causal_edge_index_or_adj: 因果图边索引或邻接矩阵
            domain_label: 领域标签，用于对抗训练（可选）
        Returns:
            fused: 融合后的特征 [B, output_dim, N]
            high_freq: 高频特征 [B, output_dim, N]
            low_freq: 低频特征 [B, output_dim, N]
        """
        # 支持 [B,T,N] 或 [B,T,N,2]
        '''
        low_src = x[..., 0]
        high_src = x[..., 1]
        '''

        low_src = x
        high_src = x
        x_for_residual = low_src + high_src  # 用于全局残差
        B, T, N = low_src.shape
        # 处理基础图结构
        if edge_index_or_adj.dim() == 2 and edge_index_or_adj.shape[0] == edge_index_or_adj.shape[1]:
            edge_index = self._adj_matrix_to_edge_index(edge_index_or_adj)
        else:
            edge_index = edge_index_or_adj

        edge_index = edge_index.to(x.device).long()

        # 处理因果图结构
        causal_edge_index = None
        causal_edge_weight = None
        if causal_edge_index_or_adj is not None:
            if causal_edge_index_or_adj.dim() == 2 and causal_edge_index_or_adj.shape[0] == causal_edge_index_or_adj.shape[1]:
                # 如果是邻接矩阵，提取权重信息
                causal_edge_index, causal_edge_weight = self._adj_matrix_to_edge_index(causal_edge_index_or_adj, return_weights=True)
            else:
                causal_edge_index = causal_edge_index_or_adj
            causal_edge_index = causal_edge_index.to(x.device).long()
            if causal_edge_weight is not None:
                causal_edge_weight = causal_edge_weight.to(x.device)

        # 构造batch的edge_index
        E = edge_index.size(1)
        batch_offsets = (torch.arange(B, device=x.device) * N).repeat_interleave(E)
        batched_edge_index = edge_index.repeat(1, B) + batch_offsets.unsqueeze(0)

        # 构造batch的causal_edge_index和权重
        batched_causal_edge_index = None
        batched_causal_edge_weight = None
        if causal_edge_index is not None:
            E_causal = causal_edge_index.size(1)
            causal_batch_offsets = (torch.arange(B, device=x.device) * N).repeat_interleave(E_causal)
            batched_causal_edge_index = causal_edge_index.repeat(1, B) + causal_batch_offsets.unsqueeze(0)

            # 批次化权重
            if causal_edge_weight is not None:
                batched_causal_edge_weight = causal_edge_weight.repeat(B)  # [B*E_causal]

        # 2. 高频分量处理
        # 2.1 高频时间特征 
        high_freq_temporal = self.high_freq_temporal(high_src)  # [B, temporal_dim, N]
        # 2.2 高频空间特征 - 使用外部普通GCN
        high_freq_input = high_freq_temporal.permute(0, 2, 1).reshape(B * N, -1)  # [B*N, temporal_dim]
        high_freq = self.high_freq_spatial(high_freq_input, batched_edge_index, B, N, None)  # 不使用因果图
        high_freq = high_freq.reshape(B, N, -1).permute(0, 2, 1)  # [B, output_dim, N]
        # 添加高频分支出口残差连接 - 对分支输入进行投影
        high_freq_residual = self.high_residual_proj(high_freq_temporal.permute(0, 2, 1)).permute(0, 2, 1)  # [B, output_dim, N]
        high_freq = high_freq + 0.2 * high_freq_residual

        high_freq = high_freq.permute(0, 2, 1)  # [B, N, C]
        high_freq = self.high_freq_norm(high_freq)  # LayerNorm(output_dim)
        high_freq = high_freq.permute(0, 2, 1)  # 再转回来 [B, C, N]

        high_freq = F.leaky_relu(high_freq, negative_slope=0.1)

        # 3. 低频分量处理（做对抗）
        # 3.1 低频时间特征 - 使用外部TemporalLSTM
        low_freq_temporal = self.low_freq_temporal(low_src)  # [B, temporal_dim, N]
        # 3.2 低频空间特征 - 使用外部因果GCN
        low_freq_input = low_freq_temporal.permute(0, 2, 1).reshape(B * N, -1)  # [B*N, temporal_dim]
        #low_freq = self.low_freq_spatial(low_freq_input, batched_edge_index, B, N, None)  # 不使用因果图

        low_freq = self.low_freq_spatial(low_freq_input, batched_edge_index, B, N, batched_causal_edge_index,
                                        edge_weight=None, causal_edge_weight=batched_causal_edge_weight)  # 使用因果图

        low_freq = low_freq.reshape(B, N, -1).permute(0, 2, 1)  # [B, output_dim, N]
        # 添加低频分支出口残差连接 - 对分支输入进行投影
        low_freq_residual = self.low_residual_proj(low_freq_temporal.permute(0, 2, 1)).permute(0, 2, 1)  # [B, output_dim, N]
        low_freq = low_freq + 0.2 * low_freq_residual
        # 分支出口归一化和激活函数 - 调整LayerNorm以对通道维度进行归一化
        low_freq = low_freq.permute(0, 2, 1)  # [B, N, C]
        low_freq = self.low_freq_norm(low_freq)  # LayerNorm(output_dim)
        low_freq = low_freq.permute(0, 2, 1)  # 再转回来 [B, C, N]
        low_freq = F.gelu(low_freq)

        # 4. 特征融合 - 使用外部注意力机制
        
        fused = self.attention_fusion(high_freq, low_freq)  # [B, output_dim, N]
        fused = fused + 0.3 * high_freq + 0.3 * low_freq  # 调整融合残差权重为0.3
        #fused = fused 

        # 5. 残差连接（使用融合前原始输入或叠加输入）
        residual = self.global_residual_proj(x_for_residual.permute(0, 2, 1)).permute(0, 2, 1)  # [B, output_dim, N]
        # 对全局残差做 LayerNorm（在通道维C上归一），先转为 [B, N, C]
        residual = residual.permute(0, 2, 1)
        residual = self.global_residual_ln(residual)
        residual = residual.permute(0, 2, 1)  # 转回 [B, C, N]
        fused = fused + 0.1 * residual
        
        return fused, high_freq, low_freq
    
    def _adj_matrix_to_edge_index(self, adj_matrix, return_weights=False):
        """
        将邻接矩阵转换为PyTorch Geometric的edge_index格式

        Args:
            adj_matrix: 邻接矩阵 [N, N]
            return_weights: 是否同时返回权重信息

        Returns:
            如果return_weights=False: edge_index [2, E]
            如果return_weights=True: (edge_index [2, E], edge_weight [E])
        """
        edge_indices = torch.nonzero(adj_matrix, as_tuple=True)
        edge_index = torch.stack([edge_indices[0], edge_indices[1]], dim=0)

        if return_weights:
            # 提取对应的权重
            edge_weight = adj_matrix[edge_indices[0], edge_indices[1]]
            return edge_index, edge_weight
        else:
            return edge_index

class GradReverse(torch.autograd.Function):
    """
    梯度反转(用于对抗)
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        # [DEBUG] 前向传播被调用
        if not hasattr(GradReverse, '_forward_calls'):
            GradReverse._forward_calls = 0
        GradReverse._forward_calls += 1
        if GradReverse._forward_calls % 1000 == 1:
            print(f"  [DEBUG GradReverse Forward] 被调用 {GradReverse._forward_calls} 次, constant={constant:.3f}")
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        original_grad_norm = grad_output.norm().item()
        grad_output = grad_output.neg() * ctx.constant
        reversed_grad_norm = grad_output.norm().item()
        
        # [DEBUG] 梯度翻转调试信息 - 每100次打印一次
        if not hasattr(GradReverse, '_debug_counter'):
            GradReverse._debug_counter = 0
        GradReverse._debug_counter += 1
        
        '''
        if GradReverse._debug_counter % 100 == 0:
            print(f"  [DEBUG GradReverse Backward] 第{GradReverse._debug_counter}次调用, constant={ctx.constant:.3f}")
            print(f"  [DEBUG] 原始梯度范数={original_grad_norm:.4f}, 翻转后范数={reversed_grad_norm:.4f}")
            print(f"  [DEBUG] 梯度是否被翻转: {abs(reversed_grad_norm - original_grad_norm * ctx.constant) < 1e-6}")
        '''
        return grad_output, None

    @staticmethod
    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Grad(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.constant
        return grad_output, None

    @staticmethod
    def grad(x, constant):
        return Grad.apply(x, constant)

class Domain_classifier_DG(nn.Module):
    """域分类器（支持三维输入）- 简化2层版本
    功能：处理 [batch_size, encode_dim, numnodes] 输入
    原理：维度合并 + 特征压缩 + 对抗训练
    """
    def __init__(self, num_class, encode_dim):
        """初始化域分类器
        Args:
            num_class (int): 域的数量(如2个域对应num_class=2)
            encode_dim (int): 输入特征维度
        """
        super().__init__()
        self.num_class = num_class
        self.encode_dim = encode_dim


        self.fc1 = nn.Linear(self.encode_dim, 16)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(16, num_class)        # 隐藏层 → 输出层

        # 权重初始化优化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier权重初始化，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # LayerNorm 通常不需要显式初始化权重偏置（默认为1/0），保留默认

    def forward(self, input, constant, Reverse):
        """前向传播（三维输入处理）- 简化2层版本
        Args:
            input (Tensor):   [batch_size, encode_dim, numnodes]
            constant (float): 梯度反转系数
            Reverse (bool):   梯度方向标志
        Returns:
            logits (Tensor): 域分类概率 [batch_size * numnodes, num_class]
        """
        # === 维度合并 ===
        # 将 batch_size 和 numnodes 维度合并为单一维度
        # 原始维度: [batch_size, encode_dim, numnodes]
        input = input.permute(0, 2, 1)  # 变为 [batch_size, numnodes, encode_dim]
        batch_size, num_nodes, feat_dim = input.shape
        input = input.reshape(-1, feat_dim)  # [batch_size * numnodes, encode_dim]

        # === 梯度反转层（GRL）===
        # 确保constant是torch.Tensor类型
        if isinstance(constant, (np.floating, float)):
            constant = torch.tensor(constant, dtype=torch.float32, device=input.device)

        if Reverse:
            input = GradReverse.grad_reverse(input, constant)
        else:
            input = Grad.grad(input, constant)

        # === 特征变换（2层简化版本）===
        logits = torch.tanh(self.fc1(input))
        logits = self.fc2(logits)
        logits = F.log_softmax(logits, 1)

        return logits

class GCNModule(nn.Module):
    """GCN模块: 根据use_causal参数决定两层都使用因果图卷积还是两层都使用普通图卷积"""
    def __init__(self, input_dim, gcn_hidden, output_dim, use_causal=True):
        super().__init__()
        if use_causal:
            # 使用因果图：两层都使用DirectedGCNConv
            self.gcn1 = DirectedGCNConv(input_dim, gcn_hidden)
            self.gcn2 = DirectedGCNConv(gcn_hidden, output_dim)
        else:
            # 不使用因果图：两层都使用普通GCNConv
            self.gcn1 = GCNConv(input_dim, gcn_hidden)
            self.gcn2 = GCNConv(gcn_hidden, output_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.residual_proj = nn.Linear(input_dim, gcn_hidden)  # 局部残差投影到第一层输出维度
        self.use_causal = use_causal

    def forward(self, x, edge_index, B, N, causal_edge_index, edge_weight=None, causal_edge_weight=None):
        x_input = x
        
        if self.use_causal and causal_edge_index is not None:
            # 使用因果图：两层都使用因果图
            # 第一层
            x_gcn1 = self.gcn1(x, causal_edge_index, edge_weight=causal_edge_weight)
            x_gcn1 = x_gcn1.reshape(B, N, -1)  # 依赖外部batch offset处理
            x_gcn1 = F.gelu(x_gcn1)  # 统一使用GELU
            x_gcn1 = self.dropout(x_gcn1)
            
            # 局部残差连接 - 从输入到第一层输出
            residual = self.residual_proj(x_input)
            residual = residual.reshape(B, N, -1)
            x_gcn1 = x_gcn1 + 0.3 * residual  # 局部残差，强度为0.4（推荐范围0.3-0.5）
            
            # 第二层
            x_gcn1_reshaped = x_gcn1.reshape(B * N, -1)  # 依赖外部batch offset处理
            spatial = self.gcn2(x_gcn1_reshaped, causal_edge_index, edge_weight=causal_edge_weight)
            spatial = F.gelu(spatial)  # 统一使用GELU
        else:
            # 不使用因果图：两层都使用基础图
            # 第一层
            x_gcn1 = self.gcn1(x, edge_index, edge_weight=edge_weight)
            x_gcn1 = x_gcn1.reshape(B, N, -1)  # 依赖外部batch offset处理
            x_gcn1 = F.gelu(x_gcn1)  # 统一使用GELU
            x_gcn1 = self.dropout(x_gcn1)
            
            # 局部残差连接 - 从输入到第一层输出
            residual = self.residual_proj(x_input)
            residual = residual.reshape(B, N, -1)
            x_gcn1 = x_gcn1 + 0.3 * residual  # 局部残差，强度为0.4（推荐范围0.3-0.5）
            
            # 第二层
            x_gcn1_reshaped = x_gcn1.reshape(B * N, -1)  # 依赖外部batch offset处理
            spatial = self.gcn2(x_gcn1_reshaped, edge_index, edge_weight=edge_weight)
            spatial = F.gelu(spatial)  # 统一使用GELU
        
        return spatial

class BranchAttentionFusion(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.W_q = nn.Linear(output_dim, output_dim)
        self.W_k = nn.Linear(output_dim, output_dim)
        self.W_v = nn.Linear(output_dim, output_dim)

    def forward(self, temporal, spatial):
        # temporal, spatial: [B, output_dim, N]
        # 先转为 [B, N, output_dim]
        temporal = temporal.permute(0, 2, 1)
        spatial = spatial.permute(0, 2, 1)
        Q = self.W_q(temporal)  # [B, N, output_dim]
        K = self.W_k(spatial)   # [B, N, output_dim]
        V = self.W_v(spatial)   # [B, N, output_dim]

        # 计算注意力分数
        # Q @ K^T: [B, N, output_dim] x [B, output_dim, N] -> [B, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, N]

        # 加权求和
        fused = torch.matmul(attn_weights, V)  # [B, N, output_dim]
        fused = fused.permute(0, 2, 1)        # [B, output_dim, N]
        return fused

class SpeedPredictor(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp  # 共享传入的 MLP

    def forward(self, x):
        # x: [B, C, N] -> [B, N, C] -> MLP -> [B, N, 12] -> [B, 12, N]
        out = self.mlp(x.permute(0, 2, 1))
        return out.permute(0, 2, 1)

def pyg_adj_to_edge_index(adj_matrix):
    """
    将邻接矩阵转换为PyTorch Geometric的edge_index格式
    Args:
        adj_matrix: [N, N] 邻接矩阵
    Returns:
        edge_index: [2, E] 边索引
    """
    if torch.is_tensor(adj_matrix):
        if adj_matrix.is_cuda:
            adj_matrix = adj_matrix.cpu()
        adj_matrix = adj_matrix.numpy()

    sparse_adj = sp.coo_matrix(adj_matrix)
    edge_index, _ = from_scipy_sparse_matrix(sparse_adj)

    # 确保返回的edge_index是正确的格式
    if edge_index.dim() == 1:
        edge_index = edge_index.view(2, -1)
    elif edge_index.dim() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index


