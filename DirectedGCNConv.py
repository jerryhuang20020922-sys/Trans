#!/usr/bin/env python3
"""
有向图图卷积网络层
调用接口与 PyTorch Geometric 的 GCNConv 保持一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DirectedGCNConv(nn.Module):
    """
    有向图图卷积网络层

    与 PyTorch Geometric GCNConv 调用接口完全一致：
    - 初始化：DirectedGCNConv(in_channels, out_channels)
    - 前向：out = conv(x, edge_index)

    特点：
    - 只考虑入边的聚合（有向图特性）
    - 自动计算入度归一化
    - 支持残差连接
    - 与标准GCNConv完全相同的接口
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        """
        初始化有向图GCN

        Args:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            improved (bool): 是否使用改进的归一化（类似GCNConv v2）
            bias (bool): 是否使用偏置项
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        # 权重矩阵
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # 偏置项
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化（与GCNConv保持一致）"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播（与GCNConv接口完全一致）

        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]，表示有向边 (源节点 -> 目标节点)
            edge_weight: 边权重 [num_edges]，表示每条边的权重，可选

        Returns:
            out: 输出节点特征 [num_nodes, out_channels]
        """
        # 1. 线性变换
        x = torch.matmul(x, self.weight)

        # 2. 消息传递（有向图版本）
        out = self.propagate(x, edge_index, edge_weight=edge_weight)

        # 3. 添加偏置
        if self.bias is not None:
            out = out + self.bias

        return out

    def propagate(self, x, edge_index, edge_weight=None):
        """
        消息传递函数（核心的有向图卷积逻辑）

        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges]，可选

        Returns:
            out: 聚合后的节点特征 [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # 获取边的源节点和目标节点
        source_nodes = edge_index[0]  # 出边起始节点
        target_nodes = edge_index[1]  # 入边目标节点

        # 1. 聚合入邻居特征
        # 使用scatter_add进行高效聚合
        aggregated = torch.zeros_like(x)

        # 对于每条边，将源节点的特征加到目标节点
        if edge_index.size(1) > 0:
            # 如果提供了边权重，则根据权重缩放源节点特征
            if edge_weight is not None:
                weighted_features = x[source_nodes] * edge_weight.unsqueeze(1)
            else:
                weighted_features = x[source_nodes]
            # 使用PyTorch的高效实现
            aggregated = aggregated.scatter_add_(
                0,  # 在节点维度上聚合
                target_nodes.unsqueeze(1).expand(-1, x.size(1)),  # 目标节点索引扩展到特征维度
                weighted_features  # 加权后的源节点特征
            )

        # 2. 计算入度或权重和
        in_degrees = torch.zeros(num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            if edge_weight is not None:
                # 如果有边权重，则计算加权入度（权重和）
                in_degrees = in_degrees.scatter_add_(
                    0,  # 在节点维度上
                    target_nodes,  # 目标节点
                    edge_weight  # 每条边的权重
                )
            else:
                # 否则计算普通入度
                in_degrees = in_degrees.scatter_add_(
                    0,  # 在节点维度上
                    target_nodes,  # 目标节点
                    torch.ones_like(target_nodes, dtype=torch.float)  # 每条边贡献1
                )

        # 3. 归一化 - 使用类似标准GCN的对称归一化方式
        # 避免除零：将入度为0的节点设为1
        in_degrees = torch.clamp(in_degrees, min=1.0)
        deg_inv_sqrt = in_degrees.pow(-0.5)
        aggregated = aggregated * deg_inv_sqrt.unsqueeze(1)
        out = x * deg_inv_sqrt.unsqueeze(1) + aggregated  # 对自身特征和邻居特征都应用 D^{-1/2} 归一化

        return out

    def __repr__(self):
        """字符串表示（与GCNConv保持一致的格式）"""
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'
