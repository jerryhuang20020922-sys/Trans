import argparse
import torch
import torch.nn.functional as F
import copy
import time
import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import contextlib
from utils.funcs import load_data, load_all_adj, masked_loss
from model import  DWTEnhancedSTGCN, SpeedPredictor, pyg_adj_to_edge_index, Domain_classifier_DG
import torch.nn as nn
from Series_Mix import MultiScaleTrendMixing, TemporalLSTM, TemporalCausalConv
from model import GCNModule
from model import BranchAttentionFusion

# === 新增：模块消融工具开始 ===
def _zero_like_output(output):
    if isinstance(output, tuple):
        return tuple(torch.zeros_like(o) if torch.is_tensor(o) else o for o in output)
    if torch.is_tensor(output):
        return torch.zeros_like(output)
    return output

def _make_zero_output_hook():
    def hook(module, inputs, output):
        return _zero_like_output(output)
    return hook

@contextlib.contextmanager
def ablate_module(module):
    hook = module.register_forward_hook(_make_zero_output_hook())
    try:
        yield
    finally:
        hook.remove()

def eval_val_target_mae_once():
    temporal_domain_classifier.eval()
    domain_classifier.eval()
    source_STGCN.eval()
    target_STGCN.eval()
    speed_predictor1.eval()
    speed_predictor2.eval()
    maes = []
    with torch.no_grad():
        t_iter = iter(t_val_dataloader.get_iterator())
        for s_feat, s_label in s_val_dataloader.get_iterator():
            try:
                t_feat, t_label = next(t_iter)
            except StopIteration:
                t_iter = iter(t_val_dataloader.get_iterator())
                t_feat, t_label = next(t_iter)

            s_feat = torch.FloatTensor(s_feat).to(device)
            s_label = torch.FloatTensor(s_label).to(device)
            t_feat = torch.FloatTensor(t_feat).to(device)
            t_label = torch.FloatTensor(t_label).to(device)

            s_fused, s_high_freq, s_low_freq = source_STGCN(s_feat, s_adj, s_causal_adj)
            t_fused, t_high_freq, t_low_freq = target_STGCN(t_feat, t_adj, t_causal_adj)

            t_pred = speed_predictor2(t_fused)

            if t_scaler is not None:
                t_pred = t_scaler.inverse_transform(t_pred)
                t_label = t_scaler.inverse_transform(t_label)

            mae_val, _, _ = masked_loss(t_pred, t_label)
            maes.append(mae_val.item())
    return float(np.mean(maes)) if len(maes) > 0 else float('nan')

def run_module_ablation():
    print("=== 模块消融评估（目标域验证） ===")
    base_mae = eval_val_target_mae_once()
    print(f"[ablation] baseline MAE={base_mae:.4f}")

    modules_to_check = {
        'src.global_residual_proj': source_STGCN.global_residual_proj,
        'src.global_residual_ln': source_STGCN.global_residual_ln,
        'src.high_residual_proj': source_STGCN.high_residual_proj,
        'src.low_residual_proj': source_STGCN.low_residual_proj,
        'tgt.global_residual_proj': target_STGCN.global_residual_proj,
        'tgt.global_residual_ln': target_STGCN.global_residual_ln,
        'tgt.high_residual_proj': target_STGCN.high_residual_proj,
        'tgt.low_residual_proj': target_STGCN.low_residual_proj,
        'shared_low_freq_lstm': shared_low_freq_lstm,
        'src_high_freq_temporal': source_high_freq_temporal,
        'tgt_high_freq_temporal': target_high_freq_temporal,
        'src_high_freq_spatial': source_high_freq_spatial,
        'tgt_high_freq_spatial': target_high_freq_spatial,
        'src_low_freq_spatial': source_low_freq_spatial,
        'tgt_low_freq_spatial': target_low_freq_spatial,
        'shared_attention_fusion': shared_attention_fusion,
        'speed_predictor2': speed_predictor2,
        'domain_classifier': domain_classifier,
        'temporal_domain_classifier': temporal_domain_classifier,
    }

    impact = []
    for name, mod in modules_to_check.items():
        try:
            with ablate_module(mod):
                mae = eval_val_target_mae_once()
            delta = mae - base_mae if (not np.isnan(mae) and not np.isnan(base_mae)) else float('nan')
            impact.append((name, delta, mae))
            print(f"[ablation] {name:30s} ΔMAE={delta:.4f} (MAE={mae:.4f})")
        except Exception as e:
            print(f"[ablation] {name:30s} 跳过，原因: {e}")

    # 排序并汇总
    impact = [x for x in impact if not np.isnan(x[1])]
    impact.sort(key=lambda x: x[1], reverse=True)
    print("=== 消融影响排序（ΔMAE降序） ===")
    for name, delta, mae in impact:
        print(f"{name:30s} ΔMAE={delta:.4f} | MAE={mae:.4f}")

# === 新增：模块消融工具结束 ===
# === 参数更新检查工具 ===
def module_checksum(module):
    with torch.no_grad():
        total = 0.0
        for p in module.parameters(recurse=True):
            if p is not None:
                total += p.detach().double().sum().item()
        return float(total)

def arg_parse(parser):
    parser = argparse.ArgumentParser()  # 在函数内初始化
    parser.add_argument('--dataset', type=str, default='4', help='dataset')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--division_seed', type=int, default=0, help='division_seed')
    parser.add_argument('--model', type=str, default='DASTNet', help='model')
    parser.add_argument('--labelrate', type=float, default=100, help='percent')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--gcn_hidden", type=int, default=64)
    parser.add_argument("--temporal_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--theta", type=float, default=1)
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument("--q", type=float, default=1)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pre_len", type=int, default=12)
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--normalize", type=bool, default=True)
    return parser.parse_args()

def train(dur, s_train_dataloader,t_train_dataloader, s_val_dataloader, t_val_dataloader, source_STGCN, target_STGCN, speed_predictor1, speed_predictor2, optimizer, total_step, start_step, domain_criterion, grl_on_this_epoch): 
    t0 = time.time()
    train_mae, val_mae, train_rmse, val_rmse, time_train_acc,  st_train_acc= list(), list(), list(), list(), list(), list()
    train_correct = 0
    
    # 收集整个epoch的损失用于返回平均值
    epoch_s_mae_list, epoch_t_mae_list = [], []
    epoch_s_rmse_list, epoch_t_rmse_list = [], []
    epoch_s_mape_list, epoch_t_mape_list = [], []

    # 统计：本epoch内开启GRL的batch数
    grl_on_batches = 0
    total_batches = 0
    # 统计：本epoch内MMD均值
    epoch_mmd_sum = 0.0
    epoch_mmd_count = 0

    domain_classifier.train()
    temporal_domain_classifier.train()
    source_STGCN.train()
    target_STGCN.train()
    speed_predictor1.train()
    speed_predictor2.train()

    
    # 目标域主导：保持目标域总步数不变；每个目标域batch配对K个源域batch
    '''
    s_iter = iter(s_train_dataloader.get_iterator())
    K = max(1, int(getattr(args, 'k_source_per_target', 1)))
    for i, (t_feat, t_label) in enumerate(t_train_dataloader.get_iterator()):
        # 清零梯度，防止跨 batch 累积
        optimizer.zero_grad(set_to_none=True)
        # 目标域当前批
        t_feat = torch.FloatTensor(t_feat).to(device)
        t_label = torch.FloatTensor(t_label).to(device)
        # 收集K个源域批（循环使用）
        s_feats = []
        s_labels = []
        for _ in range(K):
            try:
                s_feat, s_label = next(s_iter)
            except StopIteration:
                s_iter = iter(s_train_dataloader.get_iterator())
                s_feat, s_label = next(s_iter)
            s_feats.append(torch.FloatTensor(s_feat).to(device))
            s_labels.append(torch.FloatTensor(s_label).to(device))
        '''
    t_iter = iter(t_train_dataloader.get_iterator())
    for i, (s_feat, s_label) in enumerate(s_train_dataloader.get_iterator()):
        # 清零梯度，防止跨 batch 累积
        optimizer.zero_grad(set_to_none=True)
        # 获取目标域当前批，如果用完则循环从头开始
        try:
            t_feat, t_label = next(t_iter)
        except StopIteration:
            t_iter = iter(t_train_dataloader.get_iterator())
            t_feat, t_label = next(t_iter)
        # 目标域当前批
        t_feat = torch.FloatTensor(t_feat).to(device)
        t_label = torch.FloatTensor(t_label).to(device)

        # 按epoch决定是否开启GRL（整轮一致，由上一epoch的st_acc判定）
        #ST_Reverse = bool(grl_on_this_epoch)
        # 基于上一step的平衡域准确率决定是否开启对抗（初始False）
        if st_train_acc and st_train_acc[-1] > 0.6:
            ST_Reverse = True
        else:
            ST_Reverse = False
        # 统计GRL开启计数
        total_batches += 1
        if ST_Reverse:
            grl_on_batches += 1

        p = float(i + start_step) / total_step
        gamma = 1  # 更平缓的调度
        lambda_max = 1  # 降低对抗强度
        constant = lambda_max * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)  # [0, lambda_max]
 
        #print(Reverse)
        
        # 源域和目标域前向传播
        s_feat = torch.FloatTensor(s_feat).to(device)
        s_label = torch.FloatTensor(s_label).to(device)
        
        s_fused, s_high_freq, s_low_freq = source_STGCN(s_feat, s_adj, s_causal_adj)
        t_fused, t_high_freq, t_low_freq = target_STGCN(t_feat, t_adj, t_causal_adj)

        # 使用融合特征作为主要特征用于预测/对抗
        s_feat = s_fused
        t_feat = t_fused

        # 只有域分类准确率 > 40% 才启动梯度反转
        '''
        if st_train_acc and st_train_acc[-1] > 0.4:
            ST_Reverse = True
            print("梯度反转开启")
        '''
        # MMD 对齐低频特征（替代GAN）
        '''
        if s_low_freq is not None:
            s_repr = s_low_freq.permute(0, 2, 1).reshape(-1, s_low_freq.size(1))
        t_repr = t_low_freq.permute(0, 2, 1).reshape(-1, t_low_freq.size(1))
        '''
        # 线性时间MMD，限制样本上限避免OOM（例如 4096）
        #mmd_loss = mmd_rbf_linear(s_repr, t_repr, max_samples=8192)
        #epoch_mmd_sum += float(mmd_loss.item())
        #epoch_mmd_count += 1
        
        # 域对抗：源域和目标域分别计算域分类损失
        # 源域
        s_input_for_dc = s_low_freq if ST_Reverse else s_low_freq.detach()
        s_pred = domain_classifier(s_input_for_dc, constant, ST_Reverse)
        s_lab = torch.zeros(s_pred.size(0), dtype=torch.long).to(device)
        src_dom_loss = F.nll_loss(s_pred, s_lab, reduction='mean')
        s_pred_lab = s_pred.max(1, keepdim=True)[1]
        s_correct = s_pred_lab.eq(s_lab.view_as(s_pred_lab)).sum().item()
        s_acc = s_correct / max(1, s_lab.size(0))

        # 目标域：当前 batch 一次
        t_input_for_dc = t_low_freq if ST_Reverse else t_low_freq.detach()
        tgt_pred = domain_classifier(t_input_for_dc, constant, ST_Reverse)
        tgt_lab = torch.ones(tgt_pred.size(0), dtype=torch.long).to(device)
        tgt_dom_loss = F.nll_loss(tgt_pred, tgt_lab, reduction='mean')
        tgt_pred_lab = tgt_pred.max(1, keepdim=True)[1]
        t_correct = tgt_pred_lab.eq(tgt_lab.view_as(tgt_pred_lab)).sum().item()
        t_acc = t_correct / max(1, tgt_lab.size(0))

        # 平衡准确率：源/目标分别算acc再取均值
        st_train_acc.append(0.5 * (s_acc + t_acc))

        # 平衡域损失：对域内取均值，再 0.5/0.5 合成
        domain_loss = 0.5 * (src_dom_loss + tgt_dom_loss)
        

        # 源域监督
        s_pred = speed_predictor1(s_feat)
        if s_scaler is not None:
            s_pred = s_scaler.inverse_transform(s_pred)
            s_label = s_scaler.inverse_transform(s_label)
        s_mae_train, s_rmse_train, s_mape_train = masked_loss(s_pred, s_label)

        # 目标域监督
        t_pred = speed_predictor2(t_feat)
        if t_scaler is not None:
            t_pred = t_scaler.inverse_transform(t_pred)
            t_label = t_scaler.inverse_transform(t_label)

        t_mae_train, t_rmse_train, t_mape_train = masked_loss(t_pred, t_label)
        
        # 调试信息：检查第一个batch的数据
        if i == 0:
            print(f"[DEBUG] s_pred range: [{s_pred.min():.6f}, {s_pred.max():.6f}]")
            print(f"[DEBUG] s_label range: [{s_label.min():.6f}, {s_label.max():.6f}]")
            print(f"[DEBUG] t_pred range: [{t_pred.min():.6f}, {t_pred.max():.6f}]")
            print(f"[DEBUG] t_label range: [{t_label.min():.6f}, {t_label.max():.6f}]")
            print(f"[DEBUG] s_mae_train: {s_mae_train.item():.6f}")
            print(f"[DEBUG] t_mae_train: {t_mae_train.item():.6f}")

        # 收集整个epoch的损失

        epoch_s_mae_list.append(s_mae_train.item())
        epoch_t_mae_list.append(t_mae_train.item())
        epoch_s_rmse_list.append(s_rmse_train.item())
        epoch_t_rmse_list.append(t_rmse_train.item())
        epoch_s_mape_list.append(s_mape_train.item())
        epoch_t_mape_list.append(t_mape_train.item())

        mae_train = 0.5 * s_mae_train + 0.5 * t_mae_train
        
        #st_acc = st_train_acc[-1]

        # 优化域损失权重：实现动态权重调整
        # 添加域对抗损失监控
        # 平衡MAE损失和MMD对齐损失
        total_loss = mae_train
        #loss = total_loss
        loss = t_mae_train
        loss.backward()
        
        # 添加梯度裁剪防止梯度爆炸（使用更小的裁剪值）
        torch.nn.utils.clip_grad_norm_(source_STGCN.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(target_STGCN.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(speed_predictor1.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(speed_predictor2.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(domain_classifier.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(temporal_domain_classifier.parameters(), max_norm=10)
        
        optimizer.step()

        train_mae.append(t_mae_train.item())
        train_rmse.append(t_rmse_train.item())
        


    ###eval###
    # 切换到评估模式并关闭梯度
    temporal_domain_classifier.eval()
    domain_classifier.eval()
    source_STGCN.eval()
    target_STGCN.eval()
    speed_predictor1.eval()
    speed_predictor2.eval()
    with torch.no_grad():
        t_iter = iter(t_val_dataloader.get_iterator())  # 循环外初始化持久迭代器
        for i, (s_feat, s_label) in enumerate(s_val_dataloader.get_iterator()):
            try:
                t_feat, t_label = next(t_iter)
            except StopIteration:
                t_iter = iter(t_val_dataloader.get_iterator())  # 目标域数据用完后重置
                t_feat, t_label = next(t_iter)
            ###
            s_feat = torch.FloatTensor(s_feat).to(device)
            s_label = torch.FloatTensor(s_label).to(device)
            t_feat = torch.FloatTensor(t_feat).to(device)
            t_label = torch.FloatTensor(t_label).to(device)
            
            s_fused, s_high_freq, s_low_freq = source_STGCN(s_feat, s_adj, s_causal_adj)
            t_fused, t_high_freq, t_low_freq = target_STGCN(t_feat, t_adj, t_causal_adj)
            s_feat = s_fused  # 使用融合特征
            t_feat = t_fused  # 使用融合特征

            s_pred = speed_predictor1(s_feat)
            t_pred = speed_predictor2(t_feat)
            
            if s_scaler is not None:
                s_pred = s_scaler.inverse_transform(s_pred)
                s_label = s_scaler.inverse_transform(s_label)  # 移除对label的反标准化
            
            if t_scaler is not None:
                t_pred = t_scaler.inverse_transform(t_pred)
                t_label = t_scaler.inverse_transform(t_label)  # 移除对label的反标准化
            s_mae_val, s_rmse_val, s_mape_val = masked_loss(s_pred, s_label)
            t_mae_val, t_rmse_val, t_mape_val = masked_loss(t_pred, t_label)
            val_mae.append(t_mae_val.item())
            val_rmse.append(t_rmse_val.item())
    dur.append(time.time() - t0)

    # 本epoch GRL统计（无条件输出）
    '''
    print(f"本epoch开启GRL的batch数: {grl_on_batches}")
    if epoch_mmd_count > 0:
        print(f"本epoch MMD均值: {epoch_mmd_sum / epoch_mmd_count:.6f}")
    '''
    # 验证阶段目标域指标平均值（与测试口径一致）
    val_mae_avg = np.mean(val_mae) if len(val_mae) > 0 else float('nan')
    val_rmse_avg = np.mean(val_rmse) if len(val_rmse) > 0 else float('nan')

    # 返回整个epoch的平均损失
    train_epoch_t_mae_avg = np.mean(epoch_t_mae_list)
    train_epoch_t_rmse_avg = np.mean(epoch_t_rmse_list)

    
    # 返回：训练(目标域) + 验证(目标域) + 域分类准确率
    return train_epoch_t_mae_avg, train_epoch_t_rmse_avg, val_mae_avg, val_rmse_avg, np.mean(st_train_acc)


def model_train(args, source_STGCN, target_STGCN, SpeedPredictor1, SpeedPredictor2, optimizer):
    # 初始化训练过程记录变量
    dur = []                  # 存储每个epoch的训练耗时
    epoch = 1                 # 当前epoch计数器（从1开始）
    best = 999999999999999    # 初始化最佳验证损失（设为极大值）
    best_model_path = 'best_model.pth'  # 保存最佳模型的路径
    acc = list()
    cnt = 0                   # 早停计数器

    # 需要监测参数更新的模块（外部与模型内部）
    modules_to_check = {
        'src.global_residual_proj': source_STGCN.global_residual_proj,
        'src.global_residual_ln': source_STGCN.global_residual_ln,
        'src.high_residual_proj': source_STGCN.high_residual_proj,
        'src.low_residual_proj': source_STGCN.low_residual_proj,
        'tgt.global_residual_proj': target_STGCN.global_residual_proj,
        'tgt.global_residual_ln': target_STGCN.global_residual_ln,
        'tgt.high_residual_proj': target_STGCN.high_residual_proj,
        'tgt.low_residual_proj': target_STGCN.low_residual_proj,
        'shared_low_freq_lstm': shared_low_freq_lstm,
        'src_high_freq_temporal': source_high_freq_temporal,
        'tgt_high_freq_temporal': target_high_freq_temporal,
        'src_high_freq_spatial': source_high_freq_spatial,
        'tgt_high_freq_spatial': target_high_freq_spatial,
        'src_low_freq_spatial': source_low_freq_spatial,
        'tgt_low_freq_spatial': target_low_freq_spatial,
        'shared_attention_fusion': shared_attention_fusion,
        'speed_predictor1': speed_predictor1,
        'speed_predictor2': speed_predictor2,
        'domain_classifier': domain_classifier,
        'temporal_domain_classifier': temporal_domain_classifier,
        'adapted_mlp': adapted_mlp,
    }

    # 计算训练相关参数
    step_per_epoch = t_train_dataloader.get_num_batch()  # 获取每个epoch的批次数
    total_step = 100 * step_per_epoch                  # 增加总训练步数
    warmup_epochs = 0  # warm-up阶段

    start_time = time.time()
    end_time = time.time()

    prev_epoch_st_acc = 0  # 初始化为高值，第一轮通常不开GRL由你决定
    # 记录曲线
    history_train_mae = []
    history_train_rmse = []
    history_val_mae = []
    history_val_rmse = []

    while epoch <= args.epoch:  # 当未达到最大epoch数时继续训练
        start_step = epoch * step_per_epoch  # 计算当前epoch的起始步数

        # 记录epoch开始时的参数校验和
        start_checksums = {name: module_checksum(mod) for name, mod in modules_to_check.items()}

            # 联合训练阶段：加入域对抗
        # 基于上一epoch的st_acc决定本epoch是否开启GRL
        grl_on_this_epoch = 1 if prev_epoch_st_acc > 0.5 else 0
        mae_train, rmse_train, mae_val, rmse_val, st_train_acc = train(
            dur, s_train_dataloader, t_train_dataloader, s_val_dataloader, t_val_dataloader,
            source_STGCN, target_STGCN, speed_predictor1, speed_predictor2, optimizer, total_step, start_step, domain_criterion, grl_on_this_epoch
            )

        # 记录epoch结束时的参数校验和并打印更新状态
        end_checksums = {name: module_checksum(mod) for name, mod in modules_to_check.items()}
        
        '''
        print(f"[ParamUpdate][Epoch {epoch}] 模块参数是否更新：")
        for name in modules_to_check.keys():
            updated = abs(end_checksums[name] - start_checksums[name]) > 0
            print(f"  - {name}: {'UPDATED' if updated else 'NO-CHANGE'} (Δ={end_checksums[name]-start_checksums[name]:.6e})")
        '''
        print(f'Epoch {epoch} | mae_train: {mae_train: .4f} | rmse_train: {rmse_train: .4f} | mae_val: {mae_val: .4f} | rmse_val: {rmse_val: .4f} | st_acc: {st_train_acc: .4f} | Time(s) {dur[-1]: .4f}')

        # 追加历史
        history_train_mae.append(mae_train)
        history_train_rmse.append(rmse_train)
        history_val_mae.append(mae_val)
        history_val_rmse.append(rmse_val)
        prev_epoch_st_acc = st_train_acc
        

        # 早停机制：检查验证损失是否改善
        if mae_val <= best:   # 当前验证损失优于历史最佳
            best = mae_val
            cnt = 0
            # 保存当前最佳模型
            print(f"保存最佳模型到 {best_model_path}")
            torch.save({
                'source_STGCN': source_STGCN.state_dict(),
                'target_STGCN': target_STGCN.state_dict(),
                'speed_predictor1': speed_predictor1.state_dict(),
                'speed_predictor2': speed_predictor2.state_dict(),
                'temporal_domain_classifier': temporal_domain_classifier.state_dict(),
                'domain_classifier': domain_classifier.state_dict(),
                'epoch': epoch,
                'best_mae_val': best
            }, best_model_path)
        else:  # 验证损失未改善
            cnt += 1  # 早停计数器递增

        # 早停条件检查：达到耐心值或超过最大epoch
        if cnt == args.patience or epoch > args.epoch:
            print(f'Stop!!')  # 训练终止提示
            print(f'Avg acc: {np.mean(acc)}')  # 输出平均准确率
            break  # 终止训练循环

        epoch += 1            # epoch计数器递增

    # 训练完成提示
    print("训练完成！")

    # === 绘制并保存损失曲线 ===
    try:
        if len(history_train_mae) > 0:
            plt.figure()
            plt.plot(history_train_mae, label='train_mae')
            plt.plot(history_val_mae, label='val_mae')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('MAE over epochs')
            plt.legend()
            plt.tight_layout()
            plt.savefig('mae_curve.png')
            print('保存损失曲线: mae_curve.png')

        if len(history_train_rmse) > 0:
            plt.figure()
            plt.plot(history_train_rmse, label='train_rmse')
            plt.plot(history_val_rmse, label='val_rmse')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.title('RMSE over epochs')
            plt.legend()
            plt.tight_layout()
            plt.savefig('rmse_curve.png')
            print('保存损失曲线: rmse_curve.png')
        plt.close('all')
    except Exception as e:
        print(f'绘制损失曲线失败: {e}')
    return

def test():
    # 加载最佳模型
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):
        print(f"加载最佳模型从 {best_model_path}")
        checkpoint = torch.load(best_model_path)
        source_STGCN.load_state_dict(checkpoint['source_STGCN'])
        target_STGCN.load_state_dict(checkpoint['target_STGCN'])
        speed_predictor1.load_state_dict(checkpoint['speed_predictor1'])
        speed_predictor2.load_state_dict(checkpoint['speed_predictor2'])
        temporal_domain_classifier.load_state_dict(checkpoint['temporal_domain_classifier'])
        domain_classifier.load_state_dict(checkpoint['domain_classifier'])
        print(f"最佳模型在 epoch {checkpoint['epoch']} 保存，验证 MAE: {checkpoint['best_mae_val']:.4f}")
    else:
        print(f"未找到最佳模型文件 {best_model_path}，使用当前模型进行测试")

    temporal_domain_classifier.eval()
    domain_classifier.eval()
    source_STGCN.eval()
    target_STGCN.eval()
    speed_predictor1.eval()
    speed_predictor2.eval()

    # 测试目标域
    test_mape_3, test_rmse_3, test_mae_3 = list(), list(), list()
    test_mape_6, test_rmse_6, test_mae_6 = list(), list(), list()
    test_mape_12, test_rmse_12, test_mae_12 = list(), list(), list()
    all_predictions = []
    all_labels = []
    
    print("=== 目标域测试 ===")
    for i, (t_feat, t_label) in enumerate(t_test_dataloader.get_iterator()):
        with torch.no_grad():  # 关键修复：禁用梯度计算
            feat = torch.FloatTensor(t_feat).to(device)
            label = torch.FloatTensor(t_label).to(device)
            
            t_edge_index = pyg_adj_to_edge_index(t_adj).to(device)
            feat, t_high_freq, t_low_freq = target_STGCN(feat, t_adj, t_causal_adj)
            pred = speed_predictor2(feat)
            
            if t_scaler is not None:
                pred = t_scaler.inverse_transform(pred)
                label = t_scaler.inverse_transform(label)
            
            # 调试：送入loss前的整体分布
            try:
                print(f"[TEST batch {i}] pred  min={pred.min().item():.3f} max={pred.max().item():.3f} mean={pred.mean().item():.3f} std={pred.std().item():.3f}")
                print(f"[TEST batch {i}] label min={label.min().item():.3f} max={label.max().item():.3f} mean={label.mean().item():.3f} std={label.std().item():.3f}")
            except Exception:
                pass
            
            # 收集所有batch的预测和标签用于整体性能计算
            all_predictions.append(pred)
            all_labels.append(label)
            
            # 计算特定步骤的性能指标
            step_3 = pred[:, 2, :]  # 第3步 
            step_6 = pred[:, 5, :]  # 第6步
            step_12 = pred[:, 11, :] # 第12步

            label_3 = label[:, 2, :]
            label_6 = label[:, 5, :]
            label_12 = label[:, 11, :]

            mae_test_3, rmse_test_3, mape_test_3 = masked_loss(step_3, label_3)
            mae_test_6, rmse_test_6, mape_test_6 = masked_loss(step_6, label_6)
            mae_test_12, rmse_test_12, mape_test_12 = masked_loss(step_12, label_12)
            
            # 调试：分步分布
            try:
                print(f"[TEST batch {i}] step3  pred mean={step_3.mean().item():.3f} std={step_3.std().item():.3f} | label mean={label_3.mean().item():.3f} std={label_3.std().item():.3f}")
                print(f"[TEST batch {i}] step6  pred mean={step_6.mean().item():.3f} std={step_6.std().item():.3f} | label mean={label_6.mean().item():.3f} std={label_6.std().item():.3f}")
                print(f"[TEST batch {i}] step12 pred mean={step_12.mean().item():.3f} std={step_12.std().item():.3f} | label mean={label_12.mean().item():.3f} std={label_12.std().item():.3f}")
            except Exception:
                pass

            test_mae_3.append(mae_test_3.item())
            test_rmse_3.append(rmse_test_3.item())
            test_mape_3.append(mape_test_3.item())

            test_mae_6.append(mae_test_6.item())
            test_rmse_6.append(rmse_test_6.item())
            test_mape_6.append(mape_test_6.item())

            test_mae_12.append(mae_test_12.item())
            test_rmse_12.append(rmse_test_12.item())
            test_mape_12.append(mape_test_12.item())
            
            # 清理当前batch的GPU内存
            del feat, pred, label, step_3, step_6, step_12, label_3, label_6, label_12
            del mae_test_3, rmse_test_3, mape_test_3, mae_test_6, rmse_test_6, mape_test_6, mae_test_12, rmse_test_12, mape_test_12

    # 计算特定步骤的平均性能指标
    test_rmse_3 = np.mean(test_rmse_3)
    test_mae_3 = np.mean(test_mae_3)
    test_mape_3 = np.mean(test_mape_3)

    test_rmse_6 = np.mean(test_rmse_6)
    test_mae_6 = np.mean(test_mae_6)
    test_mape_6 = np.mean(test_mape_6)

    test_rmse_12 = np.mean(test_rmse_12)
    test_mae_12 = np.mean(test_mae_12)
    test_mape_12 = np.mean(test_mape_12)

    print(f'step3  mae: {test_mae_3: .2f},  step3  rmse: {test_rmse_3: .2f},  step3  mape: {test_mape_3 : .2f}')
    print(f'step6  mae: {test_mae_6: .2f},  step6  rmse: {test_rmse_6: .2f},  step6  mape: {test_mape_6 : .2f}')
    print(f'step12 mae: {test_mae_12: .2f},  step12 rmse: {test_rmse_12: .2f},  step12 mape: {test_mape_12 : .2f}')

    # 取平均，而不是直接计算拼接后的整体损失，确保与训练/验证阶段一致
    batch_overall_mae = []
    batch_overall_rmse = []
    batch_overall_mape = []
    
    for i in range(len(all_predictions)):
        pred_batch = all_predictions[i]  # [B, 12, N]
        label_batch = all_labels[i]      # [B, 12, N]
        # 使用与训练/验证相同的masked_loss函数，确保mask规则一致
        batch_mae, batch_rmse, batch_mape = masked_loss(pred_batch, label_batch)
        batch_overall_mae.append(batch_mae.item())
        batch_overall_rmse.append(batch_rmse.item())
        batch_overall_mape.append(batch_mape.item())
    
    # 计算所有batch的整体性能平均值
    overall_mae = np.mean(batch_overall_mae)
    overall_rmse = np.mean(batch_overall_rmse)
    overall_mape = np.mean(batch_overall_mape)
    
    print(f'=== 整体12步平均 ===')
    print(f'整体 MAE:  {overall_mae: .2f}')
    print(f'整体 RMSE: {overall_rmse: .2f}')
    print(f'整体 MAPE: {overall_mape: .2f}')

    print("测试结束")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    return


parser = argparse.ArgumentParser()  # 初始化解析器
args = arg_parse(parser)           # 传入解析器对象
args.dataset="7"
args.labelrate = 100  # 确保源域数据不被截断
s_train_dataloader, s_val_dataloader, s_test_dataloader, s_causal_adj, s_adj, s_scaler = load_data(args)
args.dataset="4"
args.labelrate = 100
t_train_dataloader, t_val_dataloader, t_test_dataloader, t_causal_adj, t_adj, t_scaler = load_data(args)

'''
# === 根据数据集自动计算 K(每个目标域batch配对的源域batch数)===
try:
    s_steps = s_train_dataloader.get_num_batch()
    t_steps = t_train_dataloader.get_num_batch()
    t_steps = max(1, int(t_steps))
    s_steps = max(1, int(s_steps))
    auto_k = max(1, (s_steps + t_steps - 1) // t_steps)  # 等价于 ceil(s_steps / t_steps)
    args.k_source_per_target = auto_k
    print(f"自动计算K: 源域批次数={s_steps}, 目标域批次数={t_steps}, k_source_per_target={auto_k}")
except Exception as e:
    print(f"自动计算K失败,使用默认K=1,原因: {e}")
    args.k_source_per_target = 1
'''

device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")  # 设置训练设备（优先GPU）
print(f'device: {device}')  # 打印当前使用的设备


# 设置随机种子确保可复现性
torch.manual_seed(args.seed)  # 设置PyTorch随机种子
np.random.seed(args.seed)     # 设置NumPy随机种子

# 标签率上限处理（确保不超过100%）
if args.labelrate > 100:
    args.labelrate = 100  # 限制标签率最大值为100

# 初始化域分类器及其损失函数
domain_criterion = nn.NLLLoss()  # 负对数似然损失（用于域分类任务，与log_softmax配合使用）
domain_classifier = Domain_classifier_DG(num_class=2, encode_dim=args.output_dim)
temporal_domain_classifier = Domain_classifier_DG(num_class=2, encode_dim=args.output_dim)

# 设备转移与状态初始化
domain_classifier = domain_classifier.to(device)  # 将域分类器移至指定设备
temporal_domain_classifier = temporal_domain_classifier.to(device)  # 将域分类器移至指定设备
state = None, None  # 初始化模型状态

'''
# 备份原始训练参数
bak_epoch = args.epoch  # 备份原始epoch数
bak_val = args.val      # 备份原始验证标志
bak_test = args.test    # 备份原始测试标志
'''
print(f'开始训练')
print(f'正在初始化数据迭代器...')

# 添加调试信息
print(f'源域验证数据加载器: {s_val_dataloader}')
print(f'目标域验证数据加载器: {t_val_dataloader}')


# 创建距离矩阵（从邻接矩阵）
# 确保邻接矩阵在正确的设备上
s_adj = s_adj.to(device)
t_adj = t_adj.to(device)

# 选择使用原始STGCN还是DWT增强的STGCN

print("使用原始STGCN模型")
# 共享的时间模块
# 创建共享的低频LSTM模块，用于源域和目标域之间的低频特征共享
shared_low_freq_lstm = TemporalLSTM(
    input_dim=1,
    lstm_hidden=args.lstm_hidden,
    output_dim=args.temporal_dim
).to(device)

# 2. 不共享的高频因果卷积模块（源域和目标域各自独立）
source_high_freq_temporal = TemporalCausalConv(
    input_dim=1, hidden_dim=args.lstm_hidden, output_dim=args.temporal_dim, kernel_size=3
).to(device)
target_high_freq_temporal = TemporalCausalConv(
    input_dim=1, hidden_dim=args.lstm_hidden, output_dim=args.temporal_dim, kernel_size=3
).to(device)

# 修改为共享的高频LSTM模块
shared_high_freq_temporal = TemporalCausalConv(
    input_dim=1, hidden_dim=args.lstm_hidden, output_dim=args.temporal_dim, kernel_size=3
).to(device)

# 3. 不共享的普通图卷积模块（高频分支）
source_high_freq_spatial = GCNModule(
    input_dim=args.temporal_dim, gcn_hidden=args.gcn_hidden, output_dim=args.output_dim, use_causal=False
).to(device)
target_high_freq_spatial = GCNModule(
    input_dim=args.temporal_dim, gcn_hidden=args.gcn_hidden, output_dim=args.output_dim, use_causal=False
).to(device)

# 4. 不共享的有向图卷积模块（低频分支）
source_low_freq_spatial = GCNModule(
    input_dim=args.temporal_dim, gcn_hidden=args.gcn_hidden, output_dim=args.output_dim, use_causal=True
).to(device)
target_low_freq_spatial = GCNModule(
    input_dim=args.temporal_dim, gcn_hidden=args.gcn_hidden, output_dim=args.output_dim, use_causal=True
).to(device)

# 5. 共享的注意力聚合模块
shared_attention_fusion = BranchAttentionFusion(output_dim=args.output_dim).to(device)

# 初始化源域和目标域的DWTEnhancedSTGCN模型（不包含内部模块）
source_STGCN = DWTEnhancedSTGCN(
    input_dim=args.input_dim,
    lstm_hidden=args.lstm_hidden,
    temporal_dim=args.temporal_dim,
    gcn_hidden=args.gcn_hidden,
    output_dim=args.output_dim,
    args=args
).to(device)

target_STGCN = DWTEnhancedSTGCN(
    input_dim=args.input_dim,
    lstm_hidden=args.lstm_hidden,
    temporal_dim=args.temporal_dim,
    gcn_hidden=args.gcn_hidden,
    output_dim=args.output_dim,
    args=args
).to(device)

# === 外部模块赋值 ===
# 1. 共享的低频LSTM
source_STGCN.low_freq_temporal = shared_low_freq_lstm
target_STGCN.low_freq_temporal = shared_low_freq_lstm

# 2. 不共享的高频因果卷积模块

'''
source_STGCN.high_freq_temporal = source_high_freq_temporal
target_STGCN.high_freq_temporal = target_high_freq_temporal
'''

source_STGCN.high_freq_temporal = shared_high_freq_temporal
target_STGCN.high_freq_temporal = shared_high_freq_temporal


# 3. 不共享的高频普通图卷积模块
source_STGCN.high_freq_spatial = source_high_freq_spatial
target_STGCN.high_freq_spatial = target_high_freq_spatial

# 4. 不共享的低频有向图卷积模块
source_STGCN.low_freq_spatial = source_low_freq_spatial
target_STGCN.low_freq_spatial = target_low_freq_spatial

# 5. 共享的注意力聚合模块
source_STGCN.attention_fusion = shared_attention_fusion
target_STGCN.attention_fusion = shared_attention_fusion

adapted_mlp = nn.Sequential(
    nn.Linear(args.output_dim, 32),  # 输入维度匹配STGCN输出
    nn.ReLU(),
    nn.Linear(32, 12),            # 输出单个速度值
)
speed_predictor1 = SpeedPredictor(adapted_mlp).to(device)
speed_predictor2 = SpeedPredictor(adapted_mlp).to(device)

# 配置优化器参数组，包含所有外部模块的参数
optimizer = optim.SGD([
    # 共享组件参数
    {'params': shared_low_freq_lstm.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': shared_attention_fusion.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': shared_high_freq_temporal.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},

    # 源域独立组件参数
    {'params': source_high_freq_temporal.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': source_high_freq_spatial.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': source_low_freq_spatial.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},

    # 目标域独立组件参数
    {'params': target_high_freq_temporal.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': target_high_freq_spatial.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': target_low_freq_spatial.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},

    # STGCN 内部层（确保更新）
    {'params': source_STGCN.global_residual_proj.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': source_STGCN.global_residual_ln.parameters(),   'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': source_STGCN.high_residual_proj.parameters(),   'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': source_STGCN.low_residual_proj.parameters(),    'lr': args.learning_rate, 'weight_decay': 1e-4},

    {'params': target_STGCN.global_residual_proj.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': target_STGCN.global_residual_ln.parameters(),   'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': target_STGCN.high_residual_proj.parameters(),   'lr': args.learning_rate, 'weight_decay': 1e-4},
    {'params': target_STGCN.low_residual_proj.parameters(),    'lr': args.learning_rate, 'weight_decay': 1e-4},

    # 其他组件参数
    {'params': adapted_mlp.parameters(), 'lr': args.learning_rate},
    {'params': temporal_domain_classifier.parameters(), 'lr': args.learning_rate * 0.1},  # 提高域分类器学习率
    {'params': domain_classifier.parameters(), 'lr': args.learning_rate * 0.1}
], lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

# 说明：通过外部模块创建和赋值的方式，我们可以精确控制每个组件的共享策略
# 共享组件只添加一次参数，避免重复优化

# 训练当前数据集
print("开始调用model_train...")
model_train(args, source_STGCN, target_STGCN, speed_predictor1, speed_predictor2, optimizer)
print("model_train完成,开始测试...")
test()
print("测试完成")

# === 新增：执行模块消融评估开始 ===
try:
    run_module_ablation()
except Exception as _e:
    print(f"模块消融评估失败: {_e}")
# === 新增：执行模块消融评估结束 ===

