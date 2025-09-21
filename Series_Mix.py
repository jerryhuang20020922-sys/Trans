import torch
import torch.nn as nn
import torch.nn.functional as F
from Series_Decom import Temporal_Decomposition_DWT_GPU

class MultiScaleTrendMixing(nn.Module):
    """基于DWT分解分量的多尺度趋势混合模块"""
    def __init__(self, 
                 history_seq_len,
                 future_seq_len,
                 num_channels,
                 ds_layers,
                 ds_window,
                 hidden_dim=None):  # 新增hidden_dim参数
        super(MultiScaleTrendMixing, self).__init__()

        self.history_seq_len = history_seq_len
        self.future_seq_len = future_seq_len
        self.num_channels = num_channels
        self.ds_layers = ds_layers        
        self.ds_window = ds_window
        self.hidden_dim = hidden_dim
        
        # DWT分解器
        self.dwt_decomposer = Temporal_Decomposition_DWT_GPU(
            wave='db4',  # 小波类型：db4
            J=ds_layers   # 分解层数
        )
        
        # === 直接创建两个简单的线性层 ===
        # 计算DWT分解后的序列长度（通常是原长度的一半）
        # 低频分量处理层 - 使用 TemporalLSTM
        self.low_processor = TemporalLSTM(input_dim=1, lstm_hidden=hidden_dim, output_dim=history_seq_len)
    
        # 高频分量处理层 - 暂时保留线性层，未来可根据需要替换为 TemporalLSTM
        self.high_processor = TemporalCausalConv(input_dim=1, hidden_dim=hidden_dim, output_dim=history_seq_len, kernel_size=3)
        
        # 如果指定了hidden_dim，创建投影层
        # 注意：fused张量的维度是history_seq_len，不是dwt_output_len
        self.output_proj = nn.Linear(history_seq_len, hidden_dim)
        
        # 预定义残差连接的投影层
        self.residual_proj = nn.Linear(history_seq_len, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [B, N, T] 或列表 [B, N, T]
        Returns:
            output: 处理后的特征 [B, hidden_dim, N] 或 [B, T, N]
        """
        B, _, T = x.size()  # 注意：输入是 [B, N, T]，不显式存储N
        # 使用DWT分解 - 需要转换为 [B, T, N] 格式
        trend_for_dwt = x.permute(0, 2, 1)  # [B, N, T] -> [B, T, N]
        cA, cD = self.dwt_decomposer(trend_for_dwt)  # [B, T, N]
        
        cA_reshaped = cA.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]
        cA_processed = self.low_processor(cA_reshaped)  # [B, N, history_seq_len]
        cA_processed = cA_processed.permute(0, 2, 1)  # [B, history_seq_len, N]

        #  处理高频分量 - 重塑为 [B, N, T] 格式以适应 TemporalCausalConv
        cD_reshaped = cD.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]
        cD_processed = self.high_processor(cD_reshaped)  # [B, N, history_seq_len]
        cD_processed = cD_processed.permute(0, 2, 1)  # [B, history_seq_len, N]
        # 融合两个分量 - 简单加权平均
        alpha = 0.7  # 低频分量权重
        beta = 0.3   # 高频分量权重
        # 加权融合
        fused = alpha * cA_processed + beta * cD_processed  # [B, history_seq_len, N]
        
        # 输出投影 - 需要重塑为 [B*N, T] 格式
        fused_reshaped = fused.permute(0, 2, 1).reshape(-1, self.history_seq_len)  # [B*N, history_seq_len]
        output = self.output_proj(fused_reshaped)  # [B*N, hidden_dim]
        output = output.view(B, -1, self.hidden_dim).permute(0, 2, 1)  # [B, hidden_dim, N]
        
        # === 添加总的残差连接 ===
        # 使用预定义的投影层 - 需要重塑为 [B*N, history_seq_len] 格式
        original_reshaped = x.reshape(-1, self.history_seq_len)  # [B*N, history_seq_len]
        original_projected = self.residual_proj(original_reshaped)  # [B*N, hidden_dim]
        original_projected = original_projected.view(B, -1, self.hidden_dim).permute(0, 2, 1)  # [B, hidden_dim, N]
        
        # 残差连接：output + residual_weight * original_projected
        residual_weight = 0.1  # 残差权重，可以调整
        output = output + residual_weight * original_projected
        #print("series output", output.shape)
        return output

class TemporalLSTM(nn.Module):
    """双层时序LSTM模块,带残差连接"""
    def __init__(self, input_dim, lstm_hidden, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim

        # 双层LSTM
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=2,
                          batch_first=True, dropout=0.1)

        # 中间层投影（第一层到第二层）
        self.inter_proj = nn.Linear(lstm_hidden, lstm_hidden)

        # 最终投影层
        self.temporal_proj = nn.Linear(lstm_hidden, output_dim)

        # 残差连接投影
        self.residual_proj = nn.Sequential(
            nn.Linear(12, lstm_hidden),
            nn.Linear(lstm_hidden, output_dim),
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(lstm_hidden)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, N] 输入时间序列
        Returns:
            output: [B, output_dim, N] 时序特征
        """
        B, T, N = x.shape
        total_nodes = B * N

        # 重排为 [B*N, T, 1] 以便LSTM处理每个节点的时间序列
        x_temporal = x.permute(0, 2, 1).reshape(total_nodes, T, self.input_dim)

        # 双层LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x_temporal)  # lstm_out: [B*N, T, lstm_hidden]

        # 聚合所有时间步的输出，使用平均池化
        temporal_features = lstm_out.mean(dim=1)  # [B*N, lstm_hidden]

        # 第一层归一化
        temporal_features = self.layer_norm1(temporal_features)

        # 中间层处理（可选的非线性变换）
        temporal_features = self.inter_proj(temporal_features)

        # 投影到输出维度
        temporal_features = self.temporal_proj(temporal_features)  # [B*N, output_dim]

        # 残差连接 - 直接将12步时间序列投影到输出维度
        x_for_residual = x.reshape(total_nodes, T)  # [B*N, 12]
        residual = self.residual_proj(x_for_residual)  # [B*N, output_dim]

        # 加上残差连接（权重0.3）
        temporal_features = temporal_features + 0.1 * residual

        # 最终层归一化
        temporal_features = self.layer_norm2(temporal_features)

        # 重塑回 [B, N, output_dim] 然后转置为 [B, output_dim, N]
        output = temporal_features.reshape(B, N, self.output_dim).permute(0, 2, 1)
        return output

class TemporalCausalConv(nn.Module):
    """因果卷积模块, 带残差连接"""
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # 因果卷积: padding=kernel_size-1，避免未来信息泄露
        self.causal_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size-1
        )

        # 投影层
        self.temporal_proj = nn.Linear(hidden_dim, output_dim)

        # 残差连接投影
        self.residual_proj = nn.Sequential(
            nn.Linear(12, output_dim),  # 输入T=12
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, N] 输入时间序列(来自DWT分解)
        Returns:
            output: [B, output_dim, N]
        """
        B, T, N = x.shape
        total_nodes = B * N
        # 重排为 [B*N, input_dim, T] 以便Conv1d处理
        x_temporal = x.permute(0, 2, 1).reshape(total_nodes, T, self.input_dim)
        x_temporal = x_temporal.transpose(1, 2)  # [B*N, input_dim, T]

        # 因果卷积
        conv_out = self.causal_conv(x_temporal)  # [B*N, hidden_dim, T]
        conv_out = conv_out[:, :, :T]  # 确保因果性

        # 取最后一个时间步特征
        temporal_features = conv_out[:, :, -1]  # [B*N, hidden_dim]

        # 投影到输出维度
        temporal_features = self.temporal_proj(temporal_features)  # [B*N, output_dim]

        # 残差连接 - 直接将12步时间序列投影到输出维度
        x_for_residual = x.permute(0,2,1).reshape(total_nodes, T)  # [B*N, 12]
        residual = self.residual_proj(x_for_residual)  # [B*N, output_dim]

        # 加上残差连接（权重0.3）
        temporal_features = temporal_features + 0.3 * residual

        # 层归一化
        temporal_features = self.layer_norm(temporal_features)

        # 重塑回 [B, N, output_dim] 然后转置为 [B, output_dim, N]
        output = temporal_features.reshape(B, N, self.output_dim).permute(0, 2, 1)

        return output