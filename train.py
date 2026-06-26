# -*- coding: utf-8 -*-
"""
CNN-GRU 训练脚本 - 数据驱动 & 弹性正则化版
功能特点：
1. 【去除先验】完全移除先验注入逻辑，彻底由模型通过数据自我学习特征权重。
2. 【正则化隔离】面向特征采用全局 L2 (Weight Decay)；面向卷积核采用可切换的 L1+L2, L2, L1 机制。
3. 【卷积核保底】利用 L2 最小化概率平方和的数学特性，强制进行卷积核权重保底，防止退化。
"""
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import numpy as np
import pandas as pd
import h5py
import warnings
import gc
warnings.filterwarnings("ignore")

from model import GeoSpatialTemporalNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================= 配置字典 =================
DATASET_ROOT = r"F:\PaperFiles_1\预转换张量文件\TrainV2.2"
OUTPUT_ROOT = r"F:\PaperFiles_1\Output\CNN_GRU\TrainV2.2\Train_PureDataDrivenL1_3-5-7-9" # 建议修改输出目录名以作区分
CSV_ROOT = os.path.join(DATASET_ROOT, "CSV") 

REGIONS_CONFIG = {
    'A': {
        'name': 'Region_A', 'prefix': 'A_train_fold_', 'test_h5': os.path.join(DATASET_ROOT, "A_test_normalized.h5"),
        'cnn_kernel_sizes': [3, 5, 7, 9]
    },
    'B': {
        'name': 'Region_B', 'prefix': 'B_train_fold_', 'test_h5': os.path.join(DATASET_ROOT, "B_test_normalized.h5"),
        'cnn_kernel_sizes': [3, 5, 7, 9]
    },
    'C': {
        'name': 'Region_C', 'prefix': 'C_train_fold_', 'test_h5': os.path.join(DATASET_ROOT, "C_test_normalized.h5"),
        'cnn_kernel_sizes': [3, 5, 7, 9]
    }
}

TRAINING_PARAMS = { 
    'n_folds': 10,
    'epochs': 300,
    'patience': 30,
    'num_workers': 0,
    'random_seed': 42,
    
    # === 🚀 核心：隔离的正则化参数 ===
    'feature_l2_weight_decay': 1e-4,     # 作用于 CNN/GRU 特征权重的全局 L2 正则化
    'attention_reg_method': 'L2',        # 🚀 可切换参数: ['L1+L2', 'L2', 'L1']
    'attention_l1_lambda': 5e-4,         # 控制卷积核选择的稀疏性
    'attention_l2_lambda': 1e-3,         # 控制卷积核权重的保底阈值
    
    'gru_hidden_size': 128, 
    'cnn_output_features': 256,
    'batch_size': 256,
    'learning_rate': 5e-4,
}

STATIC_NAMES = ["RSI_Band1", "RSI_Band2", "RSI_Band3", "RSI_Band4", "RSI_Band5", "RSI_Band6", "RSI_Band7", 
                "DEM", "Slope", "Aspect", "TPI", "TWI", "VRM", "Prec", "LST"]
DYNAMIC_NAMES = ["EVI_Max", "EVI_Mean", "EVI_Std", "LSWI_Max", "LSWI_Mean", "LSWI_Std", "SAVI_Max", "SAVI_Mean", "SAVI_Std"]

FEATURE_SWITCHES = {
    "SPACE": {
        "RSI_Band1": True, "RSI_Band2": True, "RSI_Band3": True, "RSI_Band4": True, "RSI_Band5": True, "RSI_Band6": True, "RSI_Band7": True,
        "DEM": True, "Slope": True, "Aspect": False, "TPI": False, "TWI": False, "VRM": False, "Prec": True, "LST": True
    },
    "TIME_MASTER_SWITCH": True, 
    "TIME": {
        "EVI_Max": True, "EVI_Mean": True, "EVI_Std": True,
        "LSWI_Max": True, "LSWI_Mean": True, "LSWI_Std": True,
        "SAVI_Max": False, "SAVI_Mean": False, "SAVI_Std": False
    }
}

ACTIVE_STATIC_INDICES = [i for i, name in enumerate(STATIC_NAMES) if FEATURE_SWITCHES["SPACE"].get(name, False)]
USE_TIME = FEATURE_SWITCHES["TIME_MASTER_SWITCH"]
ACTIVE_DYNAMIC_INDICES = [i for i, name in enumerate(DYNAMIC_NAMES) if FEATURE_SWITCHES["TIME"].get(name, False)] if USE_TIME else []

MODELS_DIR = os.path.join(OUTPUT_ROOT, "Models")
for r in ['A', 'B', 'C']: os.makedirs(os.path.join(MODELS_DIR, r), exist_ok=True)

class InMemoryGeoDataset(Dataset):
    def __init__(self, h5_path, desc="Train/Val"):
        with h5py.File(h5_path, 'r') as f:
            s_data = f['static_data'][:]
            d_data = f['dynamic_data'][:]
            l_data = f['labels'][:]
            self.fids = f['fids'][:]

        np.nan_to_num(s_data, copy=False, nan=0.0)
        np.nan_to_num(d_data, copy=False, nan=0.0)

        s_data = s_data[:, ACTIVE_STATIC_INDICES, :, :]
        if USE_TIME and len(ACTIVE_DYNAMIC_INDICES) > 0:
            d_data = d_data[:, ACTIVE_DYNAMIC_INDICES, :]
        else:
            d_data = np.zeros((s_data.shape[0], 1, d_data.shape[2]))

        self.static_data = torch.from_numpy(s_data).float()
        self.dynamic_data = torch.from_numpy(d_data).float()
        self.labels = torch.from_numpy(l_data).float()

        self.static_channels = len(ACTIVE_STATIC_INDICES)
        self.dynamic_channels = len(ACTIVE_DYNAMIC_INDICES) if USE_TIME else 0
        self.static_data_shape = self.static_data.shape

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.static_data[idx], self.dynamic_data[idx], self.labels[idx], self.fids[idx]

def inverse_log1p(y_transformed):
    return np.expm1(np.clip(y_transformed, a_min=None, a_max=10))

class EarlyStopping:
    def __init__(self, patience=15, path='checkpoint.pth'):
        self.patience = patience; self.counter = 0; self.best_score = None; self.early_stop = False
        self.path = path; self.best_metrics = {}

    def __call__(self, val_r2, val_metrics, model, epoch, optimizer):
        if self.best_score is None or val_r2 > self.best_score:
            self.best_score = val_r2
            self.save_checkpoint(val_r2, val_metrics, model, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

    def save_checkpoint(self, val_r2, val_metrics, model, epoch):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'cnn_kernel_sizes': model.cnn_kernel_sizes, 'use_time': model.use_time,
            'feature_switches': FEATURE_SWITCHES,
            'active_static_indices': ACTIVE_STATIC_INDICES,
            'active_dynamic_indices': ACTIVE_DYNAMIC_INDICES
        }
        torch.save(checkpoint, self.path)
        self.best_metrics = val_metrics.copy(); self.best_metrics['epoch'] = epoch

# ================= 主训练逻辑 =================
def train_region(region_key, region_config, params, device):
    all_folds_cache = {i: InMemoryGeoDataset(os.path.join(DATASET_ROOT, f"{region_config['prefix']}{i}.h5")) for i in range(1, params['n_folds'] + 1)}
    test_dataset = InMemoryGeoDataset(region_config['test_h5'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory=True)
    
    first_ds = test_dataset
    print(f"[{region_key} 区] 配置确认: 静态通道 {first_ds.static_channels} | 动态通道 {first_ds.dynamic_channels}")
    print(f"[{region_key} 区] 弹性网络模式: {params['attention_reg_method']}")

    fold_metrics = [] 
    epoch_logs = []   
    attention_weight_logs = [] 
    test_base_df = pd.DataFrame({'FID': test_dataset.fids, 'Real_SOC': inverse_log1p(test_dataset.labels.numpy()), 'Region': region_key})
    csv_path = os.path.join(CSV_ROOT, f"{region_key}_Val_Features.csv")
    if os.path.exists(csv_path):
        val_csv_df = pd.read_csv(csv_path)
        coord_df = val_csv_df[['FID_Orig', 'Lon', 'Lat']].rename(columns={'FID_Orig': 'FID'})
        test_base_df = pd.merge(test_base_df, coord_df, on='FID', how='left')
    else:
        print(f"  [警告] 找不到坐标映射文件 {csv_path}，输出的 Lon/Lat 将为空！")
        test_base_df['Lon'], test_base_df['Lat'] = np.nan, np.nan
    valid_fold_preds = [] 

    for fold_id in range(params['n_folds']):
        print(f"\n--- {region_key} 区 : Fold {fold_id} ---")
        val_fold_num = fold_id + 1
        train_datasets_list = [all_folds_cache[j] for j in all_folds_cache if j != val_fold_num]
        
        train_loader = DataLoader(ConcatDataset(train_datasets_list), batch_size=params['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(all_folds_cache[val_fold_num], batch_size=params['batch_size'], shuffle=False)
        
        model = GeoSpatialTemporalNet(
            static_channels=first_ds.static_channels, 
            dynamic_channels=first_ds.dynamic_channels,
            use_time=(USE_TIME and first_ds.dynamic_channels > 0),
            cnn_kernel_sizes=region_config['cnn_kernel_sizes'], 
            gru_hidden_size=params['gru_hidden_size'],
            cnn_output_features=params['cnn_output_features'], 
            image_size=first_ds.static_data_shape[2]
        ).to(device)
        
        # 🚀 在优化器直接绑定面向全局特征权重的 L2 正则化 (Weight Decay)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['feature_l2_weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        
        checkpoint_path = os.path.join(MODELS_DIR, region_key, f"checkpoint_fold_{fold_id}.pth")
        early_stopping = EarlyStopping(patience=params['patience'], path=checkpoint_path)
        
        for epoch in range(1, params['epochs'] + 1):
            model.train()
            train_loss = 0.0
            for s_b, d_b, l_b, _ in train_loader:
                s_b, d_b, l_b = s_b.to(device, non_blocking=True), d_b.to(device, non_blocking=True), l_b.to(device, non_blocking=True).unsqueeze(1)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds, att_logits, att_weights = model(s_b, d_b)
                    mse_loss = criterion(preds, l_b)
                    
                    # 🚀 面向卷积核分布的弹性正则化隔离
                    attention_reg_loss = 0.0
                    if att_logits is not None and att_weights is not None:
                        if params['attention_reg_method'] == 'L1+L2':
                            loss_l1 = params['attention_l1_lambda'] * torch.mean(torch.abs(att_logits))
                            loss_l2 = params['attention_l2_lambda'] * torch.mean(att_weights ** 2)
                            attention_reg_loss = loss_l1 + loss_l2
                        elif params['attention_reg_method'] == 'L2':
                            attention_reg_loss = params['attention_l2_lambda'] * torch.mean(att_weights ** 2)
                        elif params['attention_reg_method'] == 'L1':  # 🚀 新增纯 L1 逻辑
                            attention_reg_loss = params['attention_l1_lambda'] * torch.mean(torch.abs(att_logits))
                            
                    loss = mse_loss + attention_reg_loss
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += mse_loss.item() * s_b.size(0)
            train_loss /= len(ConcatDataset(train_datasets_list))
            
            model.eval()
            v_p, v_t = [], []
            with torch.no_grad():
                for s_b, d_b, l_b, _ in val_loader:
                    p, _, _ = model(s_b.to(device), d_b.to(device))
                    v_p.extend(p.cpu().numpy().flatten()); v_t.extend(l_b.numpy())
            
            v_p_orig, v_t_orig = inverse_log1p(np.array(v_p)), inverse_log1p(np.array(v_t))
            val_r2 = r2_score(v_t_orig, v_p_orig)
            val_rmse = np.sqrt(mean_squared_error(v_t_orig, v_p_orig))
            val_mae = mean_absolute_error(v_t_orig, v_p_orig)
            
            val_metrics = {"RMSE": val_rmse, "R2": val_r2, "MAE": val_mae}
            early_stopping(val_r2, val_metrics, model, epoch, optimizer)
            
            epoch_logs.append({"Region": region_key, "Fold": fold_id, "Epoch": epoch, "Train_Loss": train_loss, "Val_R2": val_r2, "Val_RMSE": val_rmse, "Val_MAE": val_mae})
            
            print(f"  └─ Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val R2: {val_r2:.4f} | Val RMSE: {val_rmse:.3f}")
            scheduler.step(train_loss)
            if early_stopping.early_stop: break

        final_m = early_stopping.best_metrics
        status = "Dropped" if final_m["R2"] < 0 else "Valid"
        fold_metrics.append({"Region": region_key, "Fold": fold_id, "Status": status, "Best_Epoch": final_m.get("epoch"), "R2": final_m["R2"], "RMSE": final_m["RMSE"], "MAE": final_m["MAE"]})

        # --- 提取最终测试权重 ---
        if status == "Valid":
            model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'])
            model.eval()
            test_p = []
            test_weights_list = []
            with torch.no_grad():
                for s_b, d_b, _, _ in test_loader:
                    p, _, aw = model(s_b.to(device), d_b.to(device))
                    test_p.extend(p.cpu().numpy().flatten())
                    if aw is not None:
                        test_weights_list.append(aw.cpu().numpy())
            valid_fold_preds.append(inverse_log1p(np.array(test_p)))
            
            if test_weights_list:
                aw_concat = np.concatenate(test_weights_list, axis=0) 
                mean_aw = np.mean(aw_concat, axis=0)
                record = {'Region': region_key, 'Fold': fold_id, 'R2': final_m["R2"]}
                for i, k_size in enumerate(region_config['cnn_kernel_sizes']):
                    record[f'Weight_{k_size}x{k_size}'] = mean_aw[i]
                attention_weight_logs.append(record)
        else:
            if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

        gc.collect(); torch.cuda.empty_cache()

    if valid_fold_preds:
        test_base_df['Pred_SOC'] = np.mean(valid_fold_preds, axis=0)
        test_base_df['Abs_Error'] = np.abs(test_base_df['Real_SOC'] - test_base_df['Pred_SOC'])
    else:
        test_base_df['Pred_SOC'] = np.nan 
        test_base_df['Abs_Error'] = np.nan
    return pd.DataFrame(fold_metrics), pd.DataFrame(epoch_logs), test_base_df, pd.DataFrame(attention_weight_logs)

# ================= 主流程 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 特征矩阵已加载，使用设备: {device} | 弹性特征架构准备完毕！")
    
    all_fold_metrics, all_epoch_logs, all_test_preds, all_attention_weights = [], [], [], []

    for region_key in ['A', 'B', 'C']:
        f_mets, e_logs, t_preds, aw_logs = train_region(region_key, REGIONS_CONFIG[region_key], TRAINING_PARAMS, device)
        all_fold_metrics.append(f_mets)
        all_epoch_logs.append(e_logs)
        all_test_preds.append(t_preds)
        if not aw_logs.empty:
            all_attention_weights.append(aw_logs)

    print("\n>>> 正在生成 ModelPerformance.xlsx 性能报告...")
    df_folds = pd.concat(all_fold_metrics, ignore_index=True)
    df_logs = pd.concat(all_epoch_logs, ignore_index=True)
    
    with pd.ExcelWriter(os.path.join(OUTPUT_ROOT, "ModelPerformance.xlsx"), engine='openpyxl') as writer:
        df_folds[['Region', 'Fold', 'Status', 'Best_Epoch', 'R2']].to_excel(writer, sheet_name='R2', index=False)
        df_folds[['Region', 'Fold', 'Status', 'Best_Epoch', 'RMSE']].to_excel(writer, sheet_name='RMSE', index=False)
        df_folds[['Region', 'Fold', 'Status', 'Best_Epoch', 'MAE']].to_excel(writer, sheet_name='MAE', index=False)
        df_logs.to_excel(writer, sheet_name='TrainingLog', index=False)

    print(">>> 正在生成 ValPredResult.xlsx 集成预测对比报告...")
    df_test_A, df_test_B, df_test_C = all_test_preds[0], all_test_preds[1], all_test_preds[2]
    df_test_All = pd.concat([df_test_A, df_test_B, df_test_C], ignore_index=True)
    with pd.ExcelWriter(os.path.join(OUTPUT_ROOT, "ValPredResult.xlsx"), engine='openpyxl') as writer:
        cols_single = ['FID', 'Lon', 'Lat', 'Real_SOC', 'Pred_SOC', 'Abs_Error']
        cols_all = ['FID', 'Region', 'Lon', 'Lat', 'Real_SOC', 'Pred_SOC', 'Abs_Error']
        
        df_test_A[cols_single].to_excel(writer, sheet_name='Region_A', index=False)
        df_test_B[cols_single].to_excel(writer, sheet_name='Region_B', index=False)
        df_test_C[cols_single].to_excel(writer, sheet_name='Region_C', index=False)
        df_test_All[cols_all].to_excel(writer, sheet_name='All_Regions', index=False)

    print(">>> 正在生成 AttentionWeights.xlsx 卷积核权重动态分配报告...")
    if all_attention_weights:
        df_aw_all = pd.concat(all_attention_weights, ignore_index=True)
        with pd.ExcelWriter(os.path.join(OUTPUT_ROOT, "AttentionWeights.xlsx"), engine='openpyxl') as writer:
            for region_key in ['A', 'B', 'C']:
                df_aw_region = df_aw_all[df_aw_all['Region'] == region_key]
                if not df_aw_region.empty:
                    df_aw_region.to_excel(writer, sheet_name=f'Region_{region_key}', index=False)
    else:
        print("  [!] 无有效子模型，跳过生成 AttentionWeights.xlsx")

    print("🎉 所有任务执行完毕，模型与报告已就位！")

if __name__ == "__main__":
    main()