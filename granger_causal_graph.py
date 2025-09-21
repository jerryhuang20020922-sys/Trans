import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.stats import chi2
import time

class GrangerCausalGraph(nn.Module):
    """
    基于Granger因果检验的交通因果图
    只在有道路连接的节点间计算因果关系
    """
    def __init__(self, num_nodes, max_lag=3, significance_level=0.05, enable_cache=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.enable_cache = enable_cache
        
        # 缓存机制
        self.cached_causal_graph = None
        self.last_adj_matrix = None
        self.last_input_hash = None
        
        # 预计算的滞后矩阵
        self._init_lag_matrices()
    
    def _init_lag_matrices(self):
        """初始化滞后矩阵"""
        self.lag_matrices = {}
        for lag in range(1, self.max_lag + 1):
            # 创建滞后矩阵 [N, N, lag]
            lag_matrix = torch.zeros(self.num_nodes, self.num_nodes, lag)
            self.lag_matrices[lag] = lag_matrix
    
    def _compute_input_hash(self, x, adj_matrix):
        """计算输入数据的哈希值"""
        if not self.enable_cache:
            return None
        
        # 基于输入形状和邻接矩阵的稀疏性
        x_hash = hash((x.shape[0], x.shape[1], x.shape[2]))
        adj_hash = hash(adj_matrix.sum().item()) if adj_matrix is not None else 0
        return hash((x_hash, adj_hash))
    
    def forward(self, x, adj_matrix=None):
        """
        构建基于Granger因果检验的因果图
        Args:
            x: [B, T, N] 或 [T, N] 交通流量数据
            adj_matrix: [N, N] 邻接矩阵（道路连接关系）
        Returns:
            causal_graph: [N, N] 因果图
        """
        start_time = time.time()
        # 处理 [T, N] 形状的输入
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 转换为 [1, T, N]
        B, T, N = x.shape
        device = x.device
        
        # 检查缓存
        if self.enable_cache:
            input_hash = self._compute_input_hash(x, adj_matrix)
            if (self.cached_causal_graph is not None and 
                input_hash == self.last_input_hash and
                self.cached_causal_graph.device == device):
                return self.cached_causal_graph
        
        # 确保邻接矩阵在正确的设备上
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, device=device, dtype=torch.float32)
        else:
            adj_matrix = adj_matrix.to(device)
        
        # 构建因果图
        causal_graph = self._build_granger_causal_graph(x, adj_matrix, device)
        
        # 更新缓存
        if self.enable_cache:
            self.cached_causal_graph = causal_graph
            self.last_adj_matrix = adj_matrix
            self.last_input_hash = input_hash
        
        total_time = time.time() - start_time
        if total_time > 0.05:
            print(f"Granger因果图构建耗时: {total_time:.3f}s (N={N})")
        
        return causal_graph
    
    def _build_granger_causal_graph(self, x, adj_matrix, device):
        """
        构建基于Granger因果检验的因果图
        """
        N = self.num_nodes
        causal_graph = torch.zeros(N, N, device=device, dtype=torch.float32)
        
        # 获取有连接的节点对
        connected_pairs = torch.nonzero(adj_matrix, as_tuple=True)
        
        print(f"计算{len(connected_pairs[0])}个连接节点对的Granger因果关系...")
        
        # 分批处理以节省内存
        batch_size = 100
        num_pairs = len(connected_pairs[0])
        
        for i in range(0, num_pairs, batch_size):
            end_i = min(i + batch_size, num_pairs)
            batch_pairs = (connected_pairs[0][i:end_i], connected_pairs[1][i:end_i])
            
            # 处理这一批节点对
            batch_causal = self._compute_batch_granger_causality(x, batch_pairs, device)
            
            # 更新因果图
            for idx, (node_i, node_j) in enumerate(zip(batch_pairs[0], batch_pairs[1])):
                causal_graph[node_i, node_j] = batch_causal[idx]
        
        # 确保因果图是有向图结构：对于每一对节点，根据阈值决定是否保留双向因果关系
        threshold = 0.3
        for i in range(N):
            for j in range(i + 1, N):
                if causal_graph[i, j] > 0 and causal_graph[j, i] > 0:
                    if causal_graph[i, j] >= threshold and causal_graph[j, i] >= threshold:
                        # 如果两个方向的因果强度都超过阈值，保留双向边
                        continue
                    else:
                        # 否则只保留因果强度较高的方向
                        if causal_graph[i, j] > causal_graph[j, i]:
                            causal_graph[j, i] = 0
                        else:
                            causal_graph[i, j] = 0
        
        # 归一化
        row_sums = causal_graph.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-8)
        causal_graph = causal_graph / row_sums
        
        # 添加数值稳定性检查
        # 确保因果图不包含NaN或无穷大值
        causal_graph = torch.where(
            torch.isfinite(causal_graph), 
            causal_graph, 
            torch.zeros_like(causal_graph)
        )
        
        # 确保因果图在合理范围内
        causal_graph = torch.clamp(causal_graph, 0, 1)
        
        return causal_graph
    
    def _compute_batch_granger_causality(self, x, node_pairs, device):
        """
        批量计算Granger因果关系
        Args:
            x: [B, T, N] 输入数据
            node_pairs: (indices_i, indices_j) 节点对索引
            device: 计算设备
        Returns:
            causal_strengths: [batch_size] 因果强度
        """
        batch_size = len(node_pairs[0])
        causal_strengths = torch.zeros(batch_size, device=device)
        
        # 转换为numpy进行计算（scipy.stats需要numpy）
        x_np = x.cpu().numpy()
        
        for idx in range(batch_size):
            node_i = node_pairs[0][idx].item()
            node_j = node_pairs[1][idx].item()
            
            # 检查索引是否有效
            if node_i >= x_np.shape[2] or node_j >= x_np.shape[2]:
                causal_strengths[idx] = 0.1
                continue
            
            # 获取两个节点的时间序列
            series_i = x_np[:, :, node_i]  # [B, T]
            series_j = x_np[:, :, node_j]  # [B, T]
            
            # 计算Granger因果关系
            causality = self._granger_causality_test(series_i, series_j)
            
            # 确保因果强度是有效数值
            if not np.isfinite(causality) or causality < 0:
                causality = 0.1
            elif causality > 1:
                causality = 1.0
            
            causal_strengths[idx] = causality
        
        return causal_strengths
    
    def _granger_causality_test(self, series_i, series_j):
        """
        执行Granger因果检验
        Args:
            series_i: [B, T] 节点i的时间序列
            series_j: [B, T] 节点j的时间序列
        Returns:
            causality_strength: float 因果强度 [0, 1]
        """
        try:
            # 聚合多个batch的数据进行更稳定的检验
            if series_i.shape[0] > 0:
                # 选择前几个batch进行聚合（最多3个batch）
                num_batches_to_use = min(3, series_i.shape[0])
                y = np.mean(series_i[:num_batches_to_use, :], axis=0)  # 平均多个batch
                x = np.mean(series_j[:num_batches_to_use, :], axis=0)  # 平均多个batch
            else:
                return 0.5
            
            # 确保数据长度足够
            if len(y) < self.max_lag * 3:
                return 0.5
            
            # 执行Granger因果检验
            f_stat, p_value = self._granger_test(y, x, self.max_lag)
            
            # 基于p值计算因果强度
            if p_value < self.significance_level:
                # 显著：基于F统计量计算强度
                causality_strength = min(1.0, f_stat / 100.0)  # 归一化F统计量
            else:
                # 不显著：返回较小的值
                causality_strength = 0.1
            
            return causality_strength
            
        except Exception as e:
            # 如果计算失败，返回默认值
            return 0.5
    
    def _granger_test(self, y, x, max_lag):
        """
        执行Granger因果检验
        Args:
            y: 因变量时间序列
            x: 自变量时间序列
            max_lag: 最大滞后阶数
        Returns:
            f_stat: F统计量
            p_value: p值
        """
        n = len(y)
        
        # 构建滞后变量矩阵
        Y = y[max_lag:]
        X_lagged = np.zeros((n - max_lag, max_lag))
        Y_lagged = np.zeros((n - max_lag, max_lag))
        
        for lag in range(1, max_lag + 1):
            X_lagged[:, lag-1] = x[max_lag-lag:n-lag]
            Y_lagged[:, lag-1] = y[max_lag-lag:n-lag]
        
        # 受限模型（只包含Y的滞后项）
        X_restricted = np.column_stack([np.ones(n - max_lag), Y_lagged])
        
        # 非受限模型（包含Y和X的滞后项）
        X_unrestricted = np.column_stack([np.ones(n - max_lag), Y_lagged, X_lagged])
        
        # 拟合模型
        try:
            # 受限模型
            beta_restricted = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            residuals_restricted = Y - X_restricted @ beta_restricted
            rss_restricted = np.sum(residuals_restricted ** 2)
            
            # 非受限模型
            beta_unrestricted = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            residuals_unrestricted = Y - X_unrestricted @ beta_unrestricted
            rss_unrestricted = np.sum(residuals_unrestricted ** 2)
            
            # 计算F统计量
            df1 = max_lag  # 新增参数数量
            df2 = n - max_lag - 2 * max_lag - 1  # 自由度
            
            # 添加数值稳定性检查
            if (df2 > 0 and 
                rss_restricted > rss_unrestricted and 
                rss_unrestricted > 1e-10 and  # 确保非零
                rss_restricted - rss_unrestricted > 1e-10):  # 确保差异非零
                
                f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
                # 确保F统计量是有限数
                if np.isfinite(f_stat) and f_stat > 0:
                    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                    # 确保p值是有限数
                    if not np.isfinite(p_value):
                        p_value = 1.0
                else:
                    f_stat = 0.0
                    p_value = 1.0
            else:
                f_stat = 0.0
                p_value = 1.0
            
            return f_stat, p_value
            
        except:
            return 0.0, 1.0