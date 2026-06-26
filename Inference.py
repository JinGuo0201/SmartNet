# -*- coding: utf-8 -*-
"""
CNN-GRU 推理脚本 - 终极自适应动态路由 & 独立单波段重构版 (Plug-and-Play)
特性：
1. 【独立掩膜与单波段读取】完美适配 7 个独立 RSI_Band.vrt，并以独立的 mask.vrt (1有效/0无效) 为几何基准。
2. 【零配置路由】直接从 .pth 权重文件中读取特征开关，无需在推理端手动对齐特征列表。
3. 【智能 I/O 熔断】若 .pth 记录不使用时间序列，则底层完全跳过动态 VRT 影像的硬盘读取，极速飙升。
"""
import os
import numpy as np
import torch
from osgeo import gdal, ogr
from datetime import datetime
import json
import warnings
import gc
import threading
import queue
from tqdm import tqdm

warnings.filterwarnings("ignore")
from model import GeoSpatialTemporalNet  # 确保你的模型脚本名为 model.py

# ================= 1. 基础配置区域 =================
# ----- 路径设置 (请根据你的实际情况修改) -----
FISHNET_PATH = r"F:\PaperFiles_1\StrictFishnet\TestFishnet\TestFishnet_Youyi.shp"
VRT_DIR = r"F:\PaperFiles_1\WholeVariables\Vrts"

# 🚀 核心适配：定义独立掩膜文件与单波段 RSI 列表
MASK_VRT = os.path.join(VRT_DIR, "mask.vrt")
RSI_VARS = [f"RSI_Band{i}.vrt" for i in range(1, 8)]
STATIC_VARS = ['DEM', 'Slope', 'Aspect', 'TPI', 'TWI', 'VRM', 'Prec', 'LST']
DYNAMIC_VARS = ['EVI_Max', 'EVI_Mean', 'EVI_Std', 'LSWI_Max', 'LSWI_Mean', 'LSWI_Std', 'SAVI_Max', 'SAVI_Mean', 'SAVI_Std']

OUTPUT_DIR = r"F:\PaperFiles_1\Output\CNN_GRU\TrainV2.2\Train_PriorInjectionV1\PredictionV4"
os.makedirs(OUTPUT_DIR, exist_ok=True) 
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "inference_progress_pid.json")

# 模型存放的主目录 (需包含 A/B/C 三个子文件夹)
MODEL_BASE_DIR = r"F:\PaperFiles_1\Output\CNN_GRU\TrainV2.2\Train_PriorInjectionV1\Models"
MODEL_PATHS = {
    'A': os.path.join(MODEL_BASE_DIR, "Global"),
    'B': os.path.join(MODEL_BASE_DIR, "B"),    
    'C': os.path.join(MODEL_BASE_DIR, "C"),    
}

# 预处理标准化文件路径
STATS_PATHS = {
    'A': r"F:\PaperFiles_1\预转换张量文件\TrainV2.2\A_normalization_stats.json",
    'B': r"F:\PaperFiles_1\预转换张量文件\TrainV2.2\B_normalization_stats.json",
    'C': r"F:\PaperFiles_1\预转换张量文件\TrainV2.2\C_normalization_stats.json",    
}

# ----- 核心推理参数 -----
WINDOW_SIZE = 9
NODATA_VALUE = -9999
OUTPUT_DTYPE = gdal.GDT_Float32
CHUNK_SIZE = 4096
INFERENCE_BATCH_SIZE = 3200
BUFFER_PIXELS = 30


# ================= 2. 核心功能类与函数 =================
class BatchPrefetcher:
    def __init__(self, valid_indices, valid_y, valid_x, cur_static, valid_dynamic_t, batch_size):
        self.valid_indices = valid_indices
        self.valid_y = valid_y
        self.valid_x = valid_x
        self.cur_static = cur_static  # 已完成硬件级切片
        self.valid_dynamic_t = valid_dynamic_t
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=2)
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            for idx in range(0, len(self.valid_indices), self.batch_size):
                end_idx = min(idx + self.batch_size, len(self.valid_indices))
                batch_y = self.valid_y[idx : end_idx]
                batch_x = self.valid_x[idx : end_idx]

                s_batch_np = np.stack([self.cur_static[:, y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] for y, x in zip(batch_y, batch_x)])
                s_batch_cpu = torch.from_numpy(s_batch_np).pin_memory()
                d_batch_cpu = self.valid_dynamic_t[idx : end_idx].pin_memory()

                self.queue.put((idx, end_idx, s_batch_cpu, d_batch_cpu))
        except Exception as e:
            print(f"\n[Prefetcher 异常] {e}")
        finally:
            self.queue.put(None)

    def __iter__(self): return self
    def __next__(self):
        item = self.queue.get()
        if item is None: raise StopIteration
        return item

def load_normalization_stats(stats_path):
    with open(stats_path, 'r') as f: return json.load(f)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed_pids', [])), data.get('partial_pids', {})
        except Exception: pass
    return set(), {}

def save_progress(completed_pids, partial_pids):
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'completed_pids': list(completed_pids), 'partial_pids': partial_pids}, f, indent=4)
    except: pass

# 🚀 核心自适应加载：直接从权重文件中提取配置
def load_models(region, device):
    models = []
    model_dir = MODEL_PATHS.get(region)
    stats_path = STATS_PATHS.get(region)
    
    if not model_dir or not os.path.exists(model_dir): return [], None, None
    if not stats_path or not os.path.exists(stats_path): return [], None, None
    
    stats = load_normalization_stats(stats_path)
    dropped_folds = []
    feature_config = None  
    
    for fold_id in range(10):
        checkpoint_path = os.path.join(model_dir, f"checkpoint_fold_{fold_id}.pth")
        
        if not os.path.exists(checkpoint_path): 
            dropped_folds.append(fold_id); continue
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model_use_time = checkpoint.get('use_time', True)
            
            if feature_config is None:
                feature_config = {
                    'use_time': model_use_time,
                    'active_static_indices': checkpoint.get('active_static_indices', list(range(15))),
                    'active_dynamic_indices': checkpoint.get('active_dynamic_indices', list(range(9)))
                }
            
            model = GeoSpatialTemporalNet(
                static_channels=len(feature_config['active_static_indices']),
                dynamic_channels=len(feature_config['active_dynamic_indices']) if model_use_time else 0,
                use_time=model_use_time,
                cnn_kernel_sizes=checkpoint.get('cnn_kernel_sizes', [3]), 
                gru_hidden_size=checkpoint.get('gru_hidden_size', 128),
                cnn_output_features=checkpoint.get('cnn_output_features', 256),
                image_size=WINDOW_SIZE
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        except Exception as e: 
            print(f"  [ERROR] 加载 Fold {fold_id} 模型失败: {e}")
            
    if dropped_folds: print(f"    [-] 区域 {region} 中 Fold {dropped_folds} 因未达标已被跳过。")
    return models, stats, feature_config

def get_fishnet_info(fishnet_path):
    ds = ogr.Open(fishnet_path)
    layer = ds.GetLayer()
    fishnets = []
    for feature in layer:
        weights = {'A': (feature.GetField('Weight_A') or 0) / 100.0, 'B': (feature.GetField('Weight_B') or 0) / 100.0, 'C': (feature.GetField('Weight_C') or 0) / 100.0}
        if sum(weights.values()) == 0: continue
        fishnets.append({'pid': str(feature.GetField('PID')), 'bbox': feature.GetGeometryRef().GetEnvelope(), 'weights': weights})
    return fishnets

def world_to_pixel(x, y, gt): return int((x - gt[0]) / gt[1]), int((y - gt[3]) / gt[5])

def read_raster_chunk_safe(ds, xoff, yoff, xsize, ysize, bands=None):
    raster_w, raster_h = ds.RasterXSize, ds.RasterYSize
    read_xoff, read_yoff = max(0, xoff), max(0, yoff)
    read_xend, read_yend = min(raster_w, xoff + xsize), min(raster_h, yoff + ysize)
    read_w, read_h = read_xend - read_xoff, read_yend - read_yoff
    
    num_bands = ds.RasterCount if bands is None else len(bands)
    bands_to_read = bands if bands is not None else range(1, num_bands + 1)
    
    if read_w <= 0 or read_h <= 0: return np.zeros((num_bands, ysize, xsize), dtype=np.float32)
        
    data = np.zeros((num_bands, read_h, read_w), dtype=np.float32)
    for i, b in enumerate(bands_to_read): data[i] = ds.GetRasterBand(b).ReadAsArray(read_xoff, read_yoff, read_w, read_h)
        
    pad_left, pad_top = max(0, -xoff), max(0, -yoff)
    pad_right, pad_bottom = max(0, (xoff + xsize) - raster_w), max(0, (yoff + ysize) - raster_h)
    
    if any([pad_left, pad_right, pad_top, pad_bottom]):
        data = np.pad(data, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    return data

def init_or_open_tif(out_path, orig_width, orig_height, out_gt, proj):
    if not os.path.exists(out_path):
        ds = gdal.GetDriverByName('GTiff').Create(out_path, orig_width, orig_height, 1, OUTPUT_DTYPE, ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
        ds.SetGeoTransform(out_gt)
        ds.SetProjection(proj)
        ds.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
        ds.GetRasterBand(1).Fill(NODATA_VALUE)
        ds.FlushCache()
        return ds
    return gdal.Open(out_path, gdal.GA_Update)

def apply_cross_ray_median_filter(pred_array, std_array, nodata_val, min_val=0.0, max_val=150.0, max_steps=50):
    extreme_mask = (pred_array != nodata_val) & ((pred_array < min_val) | (pred_array > max_val))
    extreme_coords = np.argwhere(extreme_mask)
    if len(extreme_coords) == 0: return pred_array, std_array
        
    height, width = pred_array.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    
    for y, x in extreme_coords:
        valid_preds, valid_stds = [], []
        for dy, dx in directions:
            for step in range(1, max_steps + 1):
                ny, nx = y + dy * step, x + dx * step
                if ny < 0 or ny >= height or nx < 0 or nx >= width: break
                val = pred_array[ny, nx]
                if val != nodata_val and min_val <= val <= max_val:
                    valid_preds.append(val); valid_stds.append(std_array[ny, nx])
                    break 
        if len(valid_preds) > 0:
            pred_array[y, x], std_array[y, x] = np.median(valid_preds), np.median(valid_stds)
        else:
            pred_array[y, x], std_array[y, x] = nodata_val, nodata_val
    return pred_array, std_array

# ================= 3. 渔网处理引擎 =================
def process_fishnet(fishnet_info, mask_ds, mask_gt, mask_proj, device, loaded_cache, completed_pids, partial_pids):
    pid, bbox, weights = fishnet_info['pid'], fishnet_info['bbox'], fishnet_info['weights']
    
    region_models, region_stats, region_configs = {}, {}, {}
    for r in ['A', 'B', 'C']:
        if weights[r] > 0 and loaded_cache.get(r):
            region_models[r], region_stats[r], region_configs[r] = loaded_cache[r]
            
    if not region_models: return False
        
    min_x, max_x, min_y, max_y = bbox
    orig_x_start, orig_y_start = world_to_pixel(min_x, max_y, mask_gt)
    orig_x_end, orig_y_end = world_to_pixel(max_x, min_y, mask_gt)
    orig_width, orig_height = orig_x_end - orig_x_start, orig_y_end - orig_y_start
    if orig_width <= 0 or orig_height <= 0: return False
    
    buf_x_start, buf_y_start = orig_x_start - BUFFER_PIXELS, orig_y_start - BUFFER_PIXELS
    buf_x_end, buf_y_end = orig_x_end + BUFFER_PIXELS, orig_y_end + BUFFER_PIXELS
    buf_width, buf_height = buf_x_end - buf_x_start, buf_y_end - buf_y_start
    
    pad = WINDOW_SIZE // 2
    num_chunks_y, num_chunks_x = (buf_height + CHUNK_SIZE - 1) // CHUNK_SIZE, (buf_width + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = num_chunks_y * num_chunks_x
    
    pred_out_path = os.path.join(OUTPUT_DIR, f"fishnet_{pid}_pred.tif")
    std_out_path = os.path.join(OUTPUT_DIR, f"fishnet_{pid}_std.tif")
    out_gt = (min_x, mask_gt[1], 0, max_y, 0, mask_gt[5])
    
    pred_ds = init_or_open_tif(pred_out_path, orig_width, orig_height, out_gt, mask_proj)
    std_ds = init_or_open_tif(std_out_path, orig_width, orig_height, out_gt, mask_proj)
    pred_band, std_band = pred_ds.GetRasterBand(1), std_ds.GetRasterBand(1)

    if pid not in partial_pids: partial_pids[pid] = []
    pbar_chunk = tqdm(total=total_chunks, desc=f" > PID {pid}", unit="chunk", leave=False)
    
    any_use_time = any(region_configs[r]['use_time'] for r in region_models)
    
    for y in range(0, buf_height, CHUNK_SIZE):
        for x in range(0, buf_width, CHUNK_SIZE):
            chunk_id = f"{y}_{x}"
            if chunk_id in partial_pids[pid]: pbar_chunk.update(1); continue
                
            cw, ch = min(CHUNK_SIZE, buf_width - x), min(CHUNK_SIZE, buf_height - y)
            global_x, global_y = buf_x_start + x, buf_y_start + y
            
            # 🚀 核心适配：使用 mask.vrt 读取中心有效像元 (严格判定 == 1 为有效)
            mask_center = read_raster_chunk_safe(mask_ds, global_x, global_y, cw, ch, bands=[1])[0]
            valid_mask = (mask_center == 1)
            
            valid_indices = np.where(valid_mask.flatten())[0]
            pbar_chunk.set_postfix({"XY": f"{y//CHUNK_SIZE},{x//CHUNK_SIZE}", "Valid": f"{len(valid_indices)}"})
            
            if len(valid_indices) == 0:
                partial_pids[pid].append(chunk_id); save_progress(completed_pids, partial_pids)
                pbar_chunk.update(1); continue
            
            # --- 🚀 核心适配：分别读取 7 个 RSI 与 8 个静态变量，合成 15 通道 ---
            s_xoff, s_yoff = global_x - pad, global_y - pad
            s_w, s_h = cw + 2 * pad, ch + 2 * pad
            
            static_data_list = []
            # 1. 读入 7 个单波段 RSI
            for rsi_var in RSI_VARS:
                v_ds = gdal.Open(os.path.join(VRT_DIR, rsi_var))
                static_data_list.append(read_raster_chunk_safe(v_ds, s_xoff, s_yoff, s_w, s_h, bands=[1])[0])
                v_ds = None
            
            # 2. 读入 8 个其他静态变量
            for var in STATIC_VARS:
                v_ds = gdal.Open(os.path.join(VRT_DIR, f"{var}.vrt"))
                static_data_list.append(read_raster_chunk_safe(v_ds, s_xoff, s_yoff, s_w, s_h, bands=[1])[0])
                v_ds = None
                
            static_data = np.stack(static_data_list)  # Shape: (15, H, W)
            
            # 🚀 智能熔断时序影像 I/O
            if any_use_time:
                dynamic_data = []
                for var in DYNAMIC_VARS:
                    v_ds = gdal.Open(os.path.join(VRT_DIR, f"{var}.vrt"))
                    dynamic_data.append(read_raster_chunk_safe(v_ds, global_x, global_y, cw, ch))
                    v_ds = None
                dynamic_data = np.stack(dynamic_data)
                time_steps = dynamic_data.shape[1]
            else:
                time_steps = 1  
            
            final_chunk_pred = np.zeros(len(valid_indices), dtype=np.float32)
            final_chunk_var = np.zeros(len(valid_indices), dtype=np.float32)
            w_sum = sum(weights[r] for r in region_models.keys())
            
            valid_y, valid_x = valid_indices // cw, valid_indices % cw

            # --- 分区模型自适应预测 ---
            for region, models in region_models.items():
                stats = region_stats[region]
                f_config = region_configs[region] 
                reg_weight = weights[region] / w_sum
                
                # [1] 静态特征标准化与硬件切片 (针对 15 个独立通道)
                static_means = np.array([stats['static'][i]['mean'] for i in range(15)], dtype=np.float32)[:, None, None]
                static_stds = np.array([stats['static'][i]['std'] for i in range(15)], dtype=np.float32)[:, None, None]
                
                cur_static = static_data.copy()
                invalid_mask_s = np.isnan(cur_static) | np.isinf(cur_static) | (cur_static < -1e30) | (cur_static == -9999)
                cur_static = np.where(invalid_mask_s, static_means, cur_static)
                cur_static = (cur_static - static_means) / (static_stds + 1e-8)
                cur_static = np.clip(np.nan_to_num(cur_static, nan=0.0), -10.0, 10.0)
                
                # 🔪 切片：只保留该模型训练时使用的静态通道
                cur_static = cur_static[f_config['active_static_indices'], :, :] 
                
                # [2] 动态特征标准化与硬件切片
                if f_config['use_time'] and len(f_config['active_dynamic_indices']) > 0:
                    dynamic_means = np.array([stats['dynamic'][i]['mean'] for i in range(9)], dtype=np.float32)[:, None, None, None]
                    dynamic_stds = np.array([stats['dynamic'][i]['std'] for i in range(9)], dtype=np.float32)[:, None, None, None]
                    
                    cur_dynamic = dynamic_data.copy()
                    invalid_mask_d = np.isnan(cur_dynamic) | np.isinf(cur_dynamic) | (cur_dynamic < -1e30) | (cur_dynamic == -9999)
                    cur_dynamic = np.where(invalid_mask_d, dynamic_means, cur_dynamic)
                    cur_dynamic = (cur_dynamic - dynamic_means) / (dynamic_stds + 1e-8)
                    cur_dynamic = np.clip(np.nan_to_num(cur_dynamic, nan=0.0), -10.0, 10.0)
                    
                    cur_dynamic = cur_dynamic[f_config['active_dynamic_indices'], :, :, :] 
                    d_tensor = torch.from_numpy(cur_dynamic).view(len(f_config['active_dynamic_indices']), time_steps, ch * cw).permute(2, 0, 1)
                    valid_dynamic_t = d_tensor[valid_indices]
                else:
                    valid_dynamic_t = torch.zeros((len(valid_indices), 1, 1), dtype=torch.float32)
                
                num_batches = (len(valid_indices) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
                pbar_batch = tqdm(total=num_batches, desc=f"   └─ Batch", unit="bt", leave=False, colour='cyan')
                reg_preds_matrix = np.zeros((len(models), len(valid_indices)), dtype=np.float32)
                
                prefetcher = BatchPrefetcher(valid_indices, valid_y, valid_x, cur_static, valid_dynamic_t, INFERENCE_BATCH_SIZE)
                
                for idx, end_idx, s_batch_cpu, d_batch_cpu in prefetcher:
                    s_batch = s_batch_cpu.to(device, non_blocking=True)
                    d_batch = d_batch_cpu.to(device, non_blocking=True)
                    
                    for m_idx, model in enumerate(models):
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            p, _, _ = model(s_batch, d_batch)
                            p_orig = torch.expm1(torch.clamp(p.squeeze(-1), max=10.0))
                            reg_preds_matrix[m_idx, idx:end_idx] = p_orig.cpu().numpy()
                            
                    pbar_batch.update(1)
                pbar_batch.close()
                
                final_chunk_pred += np.mean(reg_preds_matrix, axis=0) * reg_weight
                final_chunk_var += (np.std(reg_preds_matrix, axis=0)**2) * (reg_weight**2)
                del valid_dynamic_t, reg_preds_matrix; torch.cuda.empty_cache()
            
            # --- 组装输出与滤波 ---
            out_p, out_s = np.full((ch, cw), NODATA_VALUE, dtype=np.float32), np.full((ch, cw), NODATA_VALUE, dtype=np.float32)
            out_p.flat[valid_indices], out_s.flat[valid_indices] = final_chunk_pred, np.sqrt(final_chunk_var)
            out_p, out_s = apply_cross_ray_median_filter(out_p, out_s, nodata_val=NODATA_VALUE, min_val=0.0, max_val=150.0, max_steps=50)

            # --- 写盘逻辑 ---
            out_x, out_y = x - BUFFER_PIXELS, y - BUFFER_PIXELS
            write_x, write_y = max(0, out_x), max(0, out_y)
            write_end_x, write_end_y = min(orig_width, out_x + cw), min(orig_height, out_y + ch)
            write_w, write_h = write_end_x - write_x, write_end_y - write_y
            
            if write_w > 0 and write_h > 0:
                slice_x, slice_y = write_x - out_x, write_y - out_y
                pred_band.WriteArray(out_p[slice_y : slice_y + write_h, slice_x : slice_x + write_w], write_x, write_y)
                std_band.WriteArray(out_s[slice_y : slice_y + write_h, slice_x : slice_x + write_w], write_x, write_y)
                pred_ds.FlushCache(); std_ds.FlushCache()
            
            partial_pids[pid].append(chunk_id); save_progress(completed_pids, partial_pids)
            pbar_chunk.update(1); gc.collect()

    pbar_chunk.close()
    
    final_pred_array, final_std_array = pred_band.ReadAsArray(), std_band.ReadAsArray()
    valid_mask_final = (final_pred_array != NODATA_VALUE)
    valid_preds = final_pred_array[valid_mask_final]
    
    if len(valid_preds) > 0:
        print(f"  [√] 渔网完结 | 有效像元: {len(valid_preds)} 个")
        print(f"      ├─ SOC均值: {np.mean(valid_preds):.3f} ± {np.std(valid_preds):.3f}")
    else:
        print(f"  [!] 渔网完结 | 该区域无有效像元。")

    pred_band, std_band, pred_ds, std_ds = None, None, None, None
    if pid in partial_pids: del partial_pids[pid]
    completed_pids.add(pid); save_progress(completed_pids, partial_pids)
    
    return True

# ================= 4. 主流程 =================
def main():
    print("="*70)
    print("CNN-GRU 预测推理 | 全量自适应动态特征优化版")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n正在预热并加载全部分区模型与内部架构配置...")
    
    loaded_cache = {}
    for r in ['A', 'B', 'C']:
        models, stats, feature_config = load_models(r, device)
        if models:
            loaded_cache[r] = (models, stats, feature_config)
            time_status = "开启" if feature_config['use_time'] else "关闭"
            print(f"  [√] 区域 {r} | 子模型: {len(models)} 个 | 时序分支: [{time_status}] | 静态通道: {len(feature_config['active_static_indices'])} 个")
        else:
            print(f"  [!] 区域 {r} 未检测到任何有效子模型！")
            
    # 🚀 核心适配：以 mask.vrt 作为全局基准打开
    mask_ds = gdal.Open(MASK_VRT)
    if mask_ds is None: raise RuntimeError(f"无法打开掩膜文件: {MASK_VRT}")
    mask_gt = mask_ds.GetGeoTransform()
    mask_proj = mask_ds.GetProjection()
    
    fishnets = get_fishnet_info(FISHNET_PATH)
    
    completed_pids, partial_pids = load_progress()
    if completed_pids or partial_pids:
        print(f"\n[断点续传] 已完成 {len(completed_pids)} 个渔网，{len(partial_pids)} 个未完成。")
        
    print(f"\n开始并行自适应推理，共计 {len(fishnets)} 个渔网...")

    success_count, skip_count = 0, 0
    for idx, fn in enumerate(fishnets):
        pid = fn['pid']
        print(f"\n--- 进度: {idx+1}/{len(fishnets)} | 渔网 PID: {pid} ---")
        
        if pid in completed_pids:
            print(f"  [跳过] 渔网 {pid} 已完成。")
            skip_count += 1; continue
            
        try:
            # 传入 mask_ds 及其空间信息替代原有的 rsi_ds
            if process_fishnet(fn, mask_ds, mask_gt, mask_proj, device, loaded_cache, completed_pids, partial_pids):
                success_count += 1
        except Exception as e:
            print(f"  [致命错误] 渔网 {pid} 崩溃: {e}")

    print(f"\n{'='*70}\n推理任务结束 | 成功处理: {success_count} | 跳过: {skip_count} | 总计: {len(fishnets)}\n{'='*70}")

if __name__ == "__main__":
    main()