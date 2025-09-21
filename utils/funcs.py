from tkinter import S
import torch
import numpy as np
from .data import MyDataLoader

class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_all_adj(device):

    adj_pems04 = get_adjacency_matrix(distance_df_filename="./data/PEMS04/PEMS04.csv", num_of_vertices=307)
    adj_pems07 = get_adjacency_matrix(distance_df_filename="./data/PEMS07/PEMS07.csv", num_of_vertices=883)
    adj_pems08 = get_adjacency_matrix(distance_df_filename="./data/PEMS08/PEMS08.csv", num_of_vertices=170)

    return torch.tensor(adj_pems04).to(device), torch.tensor(adj_pems07).to(device), torch.tensor(adj_pems08).to(device)


def load_data(args, scaler=None, visualize=False, distribution=False):
    DATA_PATHS = {
        "3": {"feat": "./data/PEMS03/PEMS03.npz", "adj": "./data/PEMS03/PEMS03.csv"},
        "4": {"feat": "./data/PEMS04/PEMS04.npz", "adj": "./data/PEMS04/PEMS04.csv"},
        "7": {"feat": "./data/PEMS07/PEMS07.npz", "adj": "./data/PEMS07/PEMS07.csv"},
        "8": {"feat": "./data/PEMS08/PEMS08.npz", "adj": "./data/PEMS08/PEMS08.csv"},
        "METRA-LA": {"feat": "./data/METRA-LA/METRA-LA.npz", "adj": "./data/METRA-LA/METRA-LA.csv"},
        "PEMS-BAY": {"feat": "./data/PEMS-BAY/PEMS-BAY.npz", "adj": "./data/PEMS-BAY/PEMS-BAY.csv"},
    }
    time = False

    if args.dataset == '3':
        feat_dir = DATA_PATHS['3']['feat']
        adj_dir = DATA_PATHS['3']['adj']
        num_of_vertices = 358

    elif args.dataset == '4':
        feat_dir = DATA_PATHS['4']['feat']
        adj_dir = DATA_PATHS['4']['adj']
        num_of_vertices = 307

    elif args.dataset == '7':
        feat_dir = DATA_PATHS['7']['feat']
        adj_dir = DATA_PATHS['7']['adj']
        num_of_vertices = 883

    elif args.dataset == '8':
        feat_dir = DATA_PATHS['8']['feat']
        adj_dir = DATA_PATHS['8']['adj']
        num_of_vertices = 170

    elif args.dataset == 'METRA-LA':
        feat_dir = DATA_PATHS['METRA-LA']['feat']
        adj_dir = DATA_PATHS['METRA-LA']['adj']
        num_of_vertices = 207

    elif args.dataset == 'PEMS-BAY':
        feat_dir = DATA_PATHS['PEMS-BAY']['feat']
        adj_dir = DATA_PATHS['PEMS-BAY']['adj']
        num_of_vertices = 325
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets: {list(DATA_PATHS.keys())}")

    train_X, train_Y, val_X, val_Y, test_X, test_Y, max_speed, scaler = load_graphdata_channel1(args, feat_dir, time, scaler, visualize=visualize)
    train_dataloader = MyDataLoader(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
                                    batch_size=args.batch_size)
    val_dataloader = MyDataLoader(torch.FloatTensor(val_X), torch.FloatTensor(val_Y), batch_size=args.batch_size)
    test_dataloader = MyDataLoader(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), batch_size=args.batch_size)
    adj = get_adjacency_matrix(distance_df_filename=adj_dir, num_of_vertices=num_of_vertices)

    '''
    # 在此调用基于整个训练集窗口构建的Granger因果图，并输出统计信息（仍返回静态邻接）
    from granger_causal_graph import GrangerCausalGraph
    x_tensor = torch.as_tensor(train_X, dtype=torch.float32)  # [samples, seq_len, nodes] 
    # 展平成 [T, N] 形状
    if len(x_tensor.shape) == 3:
        x_tensor = x_tensor.reshape(-1, x_tensor.shape[-1])
    granger_builder = GrangerCausalGraph(num_nodes=num_of_vertices, max_lag=3, significance_level=0.05, enable_cache=False)
    with torch.no_grad():
        causal_adj_tensor = granger_builder(x_tensor, adj_matrix=adj)  # [N, N]
    density = (causal_adj_tensor > 1e-6).float().mean().item()
    # 统计因果图的边数
    causal_edge_count = torch.sum(causal_adj_tensor > 1e-6).item()
    '''


    return train_dataloader, val_dataloader, test_dataloader,torch.tensor(adj), torch.tensor(adj), scaler

def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A

def load_distribution(feat_dir):
    file_data = np.load(feat_dir)
    data = file_data['data']
    where_are_nans = np.isnan(data)
    data[where_are_nans] = 0
    where_are_nans = (data != data)
    data[where_are_nans] = 0
    data = data[:, :, 0]  # flow only
    data = np.array(data)

    return data

def load_graphdata_channel1(args, feat_dir, time, scaler=None, visualize=False):
    """
        dir: ./data/PEMS04/PEMS04.npz, shape: (16992, 307, 3) 59 days, 2018, 1.1 - 2.28 , [flow, occupy, speed]  24%
        dir: ./data/PEMS07/PEMS07.npz, shape: (28224, 883, 1) 98 days, 2017, 5.1 - 8.31 , [flow]                 14%
        dir: ./data/PEMS08/PEMS08.npz, shape: (17856, 170, 3) 62 days, 2016, 7.1 - 8.31 , [flow, occupy, speed]  23%
    """
    file_data = np.load(feat_dir)
    data = file_data['data']  # shape: (T, N) or (T, N, features)
    where_are_nans = np.isnan(data)
    data[where_are_nans] = 0
    where_are_nans = (data != data)
    data[where_are_nans] = 0
    
    # 处理不同维度的数据
    if len(data.shape) == 3:
        # 3维数据：只取第一个特征（flow）
        data = data[:, :, 0]  # 形状变为 (T, N)
    elif len(data.shape) == 2:
        # 2维数据：已经是 (T, N) 格式，直接使用
        pass
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    print(f"Data loaded with shape: {data.shape}")  # 调试信息

    if time:
        num_data, num_sensor = data.shape
        data = np.expand_dims(data, axis=-1)
        data = data.tolist()

        for i in range(num_data):
            time = (i % 288) / 288
            for j in range(num_sensor):
                data[i][j].append(time)

        data = np.array(data)

    max_val = np.max(data)
    time_len = data.shape[0]
    seq_len = args.seq_len
    pre_len = args.pre_len
    split_ratio = args.split_ratio
    train_size = int(time_len * split_ratio)
    val_size = int(time_len * (1 - split_ratio) / 3)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:time_len]

    if args.labelrate != 100:
        # 固定取7天的数据量 (7天 * 24小时 * 12个时间步/小时 = 2016个时间步)
        days_to_use = 10
        time_steps_per_day = 24 * 12  # 假设每5分钟一个时间步
        new_train_size = days_to_use * time_steps_per_day
        
        # 确保不超过总数据量
        if new_train_size > train_size:
            new_train_size = train_size
            
        # 从开头取数据，保证数据连续性
        train_data = train_data[:new_train_size]

    train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()

    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i: i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(val_data) - seq_len - pre_len):
        val_X.append(np.array(val_data[i: i + seq_len]))
        val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i: i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

    if visualize:
        test_X = test_X[-288:]
        test_Y = test_Y[-288:]

    if args.labelrate != 0:
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    if args.labelrate != 0:
        max_xtrain = np.max(train_X)
        max_ytrain = np.max(train_Y)
    max_xval = np.max(val_X)
    max_yval = np.max(val_Y)
    max_xtest = np.max(test_X)
    max_ytest = np.max(test_Y)

    if args.labelrate != 0:
        min_xtrain = np.min(train_X)
        min_ytrain = np.min(train_Y)
    min_xval = np.min(val_X)
    min_yval = np.min(val_Y)
    min_xtest = np.min(test_X)
    min_ytest = np.min(test_Y)

    if args.labelrate != 0:
        max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

        # scaler = StandardScaler(mean=train_X[..., 0].mean(), std=train_X[..., 0].std())
        scaler = StandardScaler(mean=train_X.mean(), std=train_X.std())

        train_X = scaler.transform(train_X)
        train_Y = scaler.transform(train_Y)
    else:
        max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xval, min_yval, min_xtest, min_ytest)

    val_X = scaler.transform(val_X)
    val_Y = scaler.transform(val_Y)
    test_X = scaler.transform(test_X)
    test_Y = scaler.transform(test_Y)

    if args.labelrate != 0:
        max_xtrain = np.max(train_X)
        max_ytrain = np.max(train_Y)
    max_xval = np.max(val_X)
    max_yval = np.max(val_Y)
    max_xtest = np.max(test_X)
    max_ytest = np.max(test_Y)

    if args.labelrate != 0:
        min_xtrain = np.min(train_X)
        min_ytrain = np.min(train_Y)
    min_xval = np.min(val_X)
    min_yval = np.min(val_Y)
    min_xtest = np.min(test_X)
    min_ytest = np.min(test_Y)

    if args.labelrate != 0:
        max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

    else:
        max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xval, min_yval, min_xtest, min_ytest)

    print("train.shape", train_X.shape)
    return train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, scaler


def masked_loss(y_pred, y_true):
    # 创建掩码（不阻断梯度）

    mask = ((y_true > 0.01) & 
            ~torch.isnan(y_true) & ~torch.isnan(y_pred)).float()
    
    # 计算损失（保持梯度）
    mae_loss = torch.abs(y_pred - y_true)
    mse_loss = torch.square(y_pred - y_true)
    
    # MAPE安全计算（对分母使用detach避免不必要的梯度）
    safe_y_true = torch.where(y_true > 0.01, y_true, torch.ones_like(y_true)).detach()
    mape_loss = torch.abs(y_pred - y_true) / safe_y_true
    
    # 应用掩码
    mae_loss = mae_loss * mask
    mse_loss = mse_loss * mask
    mape_loss = mape_loss * mask
    
    # 计算均值（保持为张量）
    valid_points = mask.sum().clamp(min=1e-6)  # 避免除以0
    mae = mae_loss.sum() / valid_points
    rmse = torch.sqrt(mse_loss.sum() / valid_points)
    mape = mape_loss.sum() / valid_points
    
    return mae, rmse, mape