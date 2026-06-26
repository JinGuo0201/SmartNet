# -*- coding: utf-8 -*-
"""
全新 Offline Dataloader 引擎 (离线数据预处理与 H5 张量提取)
设计理念：完全替代传统的在线 Dataset 加载，提前完成异常清洗、划分、标准化与张量固化。
最新更新：
1. 参数区精细化隔离（基础/划分/清洗独立控制）。
2. 引入独立二值掩膜 (Mask.vrt)，1有效/0无效。
3. RSI 取消多波段合并形式，改为独立读取 RSI_Band1 ~ RSI_Band7.vrt。
"""

import os
import json
import numpy as np
import pandas as pd
import h5py
from osgeo import gdal, ogr, osr
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

# ====================================================================
# ======================= 1. 基础路径与核心配置 =======================
# ====================================================================
INPUT_SHP = r"F:\PaperFiles_1\B区样本处理工程\DataSetV8\B_All_Samples.shp"     # 原始样点 SHP
VRT_DIR = r"F:\PaperFiles_1\WholeVariables\Vrts"                              # VRT 所在总文件夹
OUTPUT_DIR = r"F:\PaperFiles_1\B区样本处理工程\DataSetV10"                         # 新纯净数据集输出目录
REGION_NAME = "B"

WINDOW_SIZE = 25          # 提取张量的空间窗口大小
TARGET_FIELD = "SOC"     # SHP中的目标预测字段
MAX_WORKERS = 14         # 多线程并发数

# ======================= 2. 数据集切分参数 ===========================
TRAIN_RATIO = 7          # 训练集与验证集比例 (7代表7:3)
FOLD_NUM = 10            # K-Fold 交叉验证折数

# ======================= 3. 异常值清洗参数 (可选开关) =================
ENABLE_OUTLIER_REMOVAL = False  # True: 开启异常清洗; False: 仅执行常规提取与划分
OLD_DATASET_DIR = r"F:\PaperFiles_1\C区样本处理工程\DataSetV0"  # 异常侦测对照库 (旧H5目录)
IQR_MULTIPLIER = 1.2            # IQR 异常界限乘数
CONTAMINATION = 0.2             # 孤立森林预期污染率

# ======================= 4. 变量组与掩膜配置 =========================
MASK_VRT = "mask.vrt"    # 独立掩膜文件，1代表有效，0代表Nodata (需在 VRT_DIR 下)

# RSI 改为 7 个单波段 VRT 独立输入
RSI_VARS = [f"RSI_Band{i}.vrt" for i in range(1, 8)]

STATIC_VARS = [
    ('DEM', 'DEM.vrt'), ('Slope', 'Slope.vrt'), ('Aspect', 'Aspect.vrt'),
    ('TPI', 'TPI.vrt'), ('TWI', 'TWI.vrt'), ('VRM', 'VRM.vrt'),
    ('Prec', 'Prec.vrt'), ('LST', 'LST.vrt'),
]
DYNAMIC_VARS = [
    ('EVI_Max', 'EVI_Max.vrt'), ('EVI_Mean', 'EVI_Mean.vrt'), ('EVI_Std', 'EVI_Std.vrt'),
    ('LSWI_Max', 'LSWI_Max.vrt'), ('LSWI_Mean', 'LSWI_Mean.vrt'), ('LSWI_Std', 'LSWI_Std.vrt'),
    ('SAVI_Max', 'SAVI_Max.vrt'), ('SAVI_Mean', 'SAVI_Mean.vrt'), ('SAVI_Std', 'SAVI_Std.vrt'),
]

NODATA_VALUE = -9999
EPSILON = 1e-8


# ====================================================================
# ======================= 第一部分：异常侦测机制 =======================
# ====================================================================
def detect_outliers_from_old_h5():
    old_h5_dir = os.path.join(OLD_DATASET_DIR, "H5")
    print(f"\n[1/6] 🕵️ 正在扫描旧目录以侦测异常点: {old_h5_dir}")
    if not os.path.exists(old_h5_dir): raise RuntimeError(f"找不到旧的 H5 目录：{old_h5_dir}")
        
    h5_files = [os.path.join(old_h5_dir, f) for f in os.listdir(old_h5_dir) if f.endswith('.h5')]
    if not h5_files: raise RuntimeError("未在旧目录中发现任何 H5 文件！")

    all_labels, all_fids, all_features = [], [], []
    total_samples = 0
    
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            labels = f['labels'][:]
            fids = f['fids'][:]
            center_idx = f['static_data'].shape[2] // 2
            center_static = f['static_data'][:, :, center_idx, center_idx]
            mean_dynamic = np.mean(f['dynamic_data'][:], axis=2)
            
            combined = np.concatenate([center_static, mean_dynamic], axis=1)
            all_labels.append(labels); all_fids.append(fids); all_features.append(combined)
            total_samples += len(fids)

    labels_merged, fids_merged, features_merged = np.concatenate(all_labels), np.concatenate(all_fids), np.concatenate(all_features)
    print(f"  > 成功聚合旧版 H5，总样本量: {total_samples}")

    outlier_fids = set()
    # 1. IQR 目标值检测
    q1, q3 = np.percentile(labels_merged, 25), np.percentile(labels_merged, 75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
    outlier_fids.update(fids_merged[np.where((labels_merged < lower_bound) | (labels_merged > upper_bound))[0]])
    
    # 2. 孤立森林多维特征检测
    iso_forest = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    predictions = iso_forest.fit_predict(np.nan_to_num(features_merged, nan=0.0))
    outlier_fids.update(fids_merged[np.where(predictions == -1)[0]])
    
    print(f"  > 诊断完毕！锁定异常 FID 共计 {len(outlier_fids)} 个。")
    return set(int(x) for x in outlier_fids)


# ====================================================================
# ======================= 第二部分：张量提取引擎 =======================
# ====================================================================
thread_local = threading.local()

def get_thread_vrt_ds(vrt_path):
    if not hasattr(thread_local, 'vrt_ds'): thread_local.vrt_ds = {}
    if vrt_path not in thread_local.vrt_ds: thread_local.vrt_ds[vrt_path] = gdal.Open(vrt_path)
    return thread_local.vrt_ds[vrt_path]

def world_to_pixel(x, y, gt):
    return int((x - gt[0]) / gt[1]), int((y - gt[3]) / gt[5])

def fill_nodata_nearest(data, nodata_value=NODATA_VALUE):
    result = data.copy()
    mask = np.isnan(result) | (np.abs(result) > 1e10) | np.isclose(result, nodata_value)
    if not np.any(mask): return result
    valid_y, valid_x = np.where(~mask)
    if len(valid_y) == 0:
        result[:] = 0 
        return result
    nodata_y, nodata_x = np.where(mask)
    for ny, nx in zip(nodata_y, nodata_x):
        dists = (valid_y - ny) ** 2 + (valid_x - nx) ** 2
        min_idx = np.argmin(dists)
        result[ny, nx] = result[valid_y[min_idx], valid_x[min_idx]]
    return result

def process_single_sample_h5(args):
    point, px, py, window_size = args
    half_win = window_size // 2
    
    # ---------------- 1. 独立二值掩膜验证 ----------------
    mask_ds = get_thread_vrt_ds(os.path.join(VRT_DIR, MASK_VRT))
    mask_data = mask_ds.GetRasterBand(1).ReadAsArray(px - half_win, py - half_win, window_size, window_size)
    if mask_data is None or np.all(mask_data == 0):
        return None # 1为有效，0为Nodata，若窗口全为0则直接判定无效
        
    static_windows = []
    
    # ---------------- 2. 依次读取独立的 RSI 波段 ----------------
    for rsi_file in RSI_VARS:
        ds = get_thread_vrt_ds(os.path.join(VRT_DIR, rsi_file))
        nd = ds.GetRasterBand(1).GetNoDataValue()
        data = ds.GetRasterBand(1).ReadAsArray(px - half_win, py - half_win, window_size, window_size)
        if data is None: return None
        data = data.astype(np.float32)
        if nd is not None: data[np.isclose(data, nd)] = NODATA_VALUE
        static_windows.append(fill_nodata_nearest(data))

    # ---------------- 3. 读取其他 Static 变量 ----------------
    for _, v_file in STATIC_VARS:
        ds = get_thread_vrt_ds(os.path.join(VRT_DIR, v_file))
        nd = ds.GetRasterBand(1).GetNoDataValue()
        data = ds.GetRasterBand(1).ReadAsArray(px - half_win, py - half_win, window_size, window_size)
        if data is None: return None
        data = data.astype(np.float32)
        if nd is not None: data[np.isclose(data, nd)] = NODATA_VALUE
        static_windows.append(fill_nodata_nearest(data))

    # ---------------- 4. 读取 Dynamic 变量 (中心像元提取) --------
    dynamic_values = []
    for _, v_file in DYNAMIC_VARS:
        ds = get_thread_vrt_ds(os.path.join(VRT_DIR, v_file))
        nd = ds.GetRasterBand(1).GetNoDataValue()
        data_3d = ds.ReadAsArray(px - half_win, py - half_win, window_size, window_size)
        if data_3d is None: return None
        data_3d = data_3d.astype(np.float32)
        vals = []
        for b in range(ds.RasterCount):
            band_data = data_3d[b]
            if nd is not None: band_data[np.isclose(band_data, nd)] = NODATA_VALUE
            filled_band = fill_nodata_nearest(band_data)
            vals.append(filled_band[half_win, half_win])
        dynamic_values.append(vals)

    return {
        'fid': point['fid'], 'target': point['target'],
        'static': np.array(static_windows, dtype=np.float32),
        'dynamic': np.array(dynamic_values, dtype=np.float32)
    }

def compute_stats(data):
    valid = data[np.isfinite(data) & (np.abs(data) < 1e10)]
    if len(valid) == 0: return 0.0, 1.0
    mean, std = np.mean(valid), np.std(valid)
    return float(mean), float(std) if std >= EPSILON else 1.0


# ====================================================================
# ======================= 第三部分：主控流水线 =========================
# ====================================================================
def main():
    print("="*80)
    print(f"🌍 独立 Offline Dataloader 预处理引擎 | 区域: {REGION_NAME}")
    print(f"🔧 异常值清洗模式: {'开启 (ON)' if ENABLE_OUTLIER_REMOVAL else '关闭 (OFF)'}")
    print("="*80)
    
    for d in ["H5", "CSV", "SHP"]: os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

    # 第一步：获取异常黑名单
    bad_fids = detect_outliers_from_old_h5() if ENABLE_OUTLIER_REMOVAL else set()
    if not ENABLE_OUTLIER_REMOVAL: print(f"\n[1/6] ⏭️ 异常值剔除功能已关闭，跳过旧数据扫描。")

    # 第二步：坐标转换与清洗过滤
    print(f"\n[2/6] 正在读取原始 SHP 并收集样点...")
    ds = ogr.Open(INPUT_SHP)
    layer = ds.GetLayer()
    source_srs = layer.GetSpatialRef()
    
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)
    if int(gdal.VersionInfo('VERSION_NUM')) >= 3000000:
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if source_srs: source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source_srs, target_srs) if source_srs else None
    
    all_points = []
    skipped_count = 0
    for feature in layer:
        fid = feature.GetFID()
        if fid in bad_fids: 
            skipped_count += 1
            continue
            
        soc = feature.GetField(TARGET_FIELD)
        if soc is None or np.isnan(soc): continue
        geom = feature.GetGeometryRef()
        x, y = geom.GetX(), geom.GetY()
        
        lon, lat = x, y
        if transform:
            geom_clone = geom.Clone()
            geom_clone.Transform(transform)
            lon, lat = geom_clone.GetX(), geom_clone.GetY()
            
        all_points.append({'fid': fid, 'x': x, 'y': y, 'lon': lon, 'lat': lat, 'target': soc})
    ds = None
    
    all_points.sort(key=lambda p: p['target'])
    for idx, pt in enumerate(all_points): pt['PID'] = idx + 1
    
    print(f"  > 收集完毕。拦截脏点: {skipped_count} 个。参与划分样点数: {len(all_points)} 个。")

    # 第三步：分配 Train/Val 与 Fold
    print("\n[3/6] 基于纯净数据执行切分逻辑...")
    train_pts, val_pts = [], []
    for i in range(0, len(all_points), 10):
        chunk = all_points[i:i+10]
        train_pts.extend(chunk[:TRAIN_RATIO])
        val_pts.extend(chunk[TRAIN_RATIO:])
        
    train_pts.sort(key=lambda p: p['target'], reverse=True)
    for i, pt in enumerate(train_pts):
        pt['set'] = 'Train'
        pt['fold'] = (i % FOLD_NUM) + 1 
    for pt in val_pts:
        pt['set'] = 'Val'
        pt['fold'] = 0

    final_points = train_pts + val_pts
    print(f"  > 切分完成: Train={len(train_pts)} ({FOLD_NUM} Fold), Val={len(val_pts)}")

    # 第四步：多线程并行提取
    print(f"\n[4/6] 🚀 开启 H5 张量提取引擎，并计算 Z-Score...")
    # 获取掩膜的地理变换参数以计算像素偏移
    mask_ds = gdal.Open(os.path.join(VRT_DIR, MASK_VRT))
    gt = mask_ds.GetGeoTransform()
    mask_ds = None
    
    results_dict = {}
    args_list = [(p, *world_to_pixel(p['x'], p['y'], gt), WINDOW_SIZE) for p in final_points]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_sample_h5, arg): arg[0]['fid'] for arg in args_list}
        for future in as_completed(futures):
            fid = futures[future]
            res = future.result()
            if res: results_dict[fid] = res

    # 聚合张量数据
    t_stat, t_dyn, t_lab, t_fids, t_fold_ids = [], [], [], [], []
    v_stat, v_dyn, v_lab, v_fids = [], [], [], []
    
    for pt in train_pts:
        if pt['fid'] in results_dict:
            r = results_dict[pt['fid']]
            t_stat.append(r['static']); t_dyn.append(r['dynamic']); t_lab.append(r['target'])
            t_fids.append(r['fid']); t_fold_ids.append(pt['fold'])
            
    for pt in val_pts:
        if pt['fid'] in results_dict:
            r = results_dict[pt['fid']]
            v_stat.append(r['static']); v_dyn.append(r['dynamic']); v_lab.append(r['target'])
            v_fids.append(r['fid'])
            
    t_stat, t_dyn = np.array(t_stat), np.array(t_dyn)
    t_lab, t_fids, t_fold_ids = np.array(t_lab), np.array(t_fids), np.array(t_fold_ids)
    v_stat, v_dyn = np.array(v_stat), np.array(v_dyn)
    v_lab, v_fids = np.array(v_lab), np.array(v_fids)

    # 执行标准化
    stats = {'static': [], 'dynamic': []}
    for i in range(t_stat.shape[1]):
        m, s = compute_stats(t_stat[:, i, :, :])
        stats['static'].append({'channel': i, 'mean': m, 'std': s})
        t_stat[:, i, :, :] = (t_stat[:, i, :, :] - m) / s
        if len(v_stat) > 0: v_stat[:, i, :, :] = (v_stat[:, i, :, :] - m) / s
        
    for i in range(t_dyn.shape[1]):
        m, s = compute_stats(t_dyn[:, i, :])
        stats['dynamic'].append({'channel': i, 'mean': m, 'std': s})
        t_dyn[:, i, :] = (t_dyn[:, i, :] - m) / s
        if len(v_dyn) > 0: v_dyn[:, i, :] = (v_dyn[:, i, :] - m) / s
        
    # SOC目标值平滑转换
    t_lab_log = np.log1p(np.maximum(t_lab, 0))
    v_lab_log = np.log1p(np.maximum(v_lab, 0))

    # 输出训练集 (分Fold写入)
    h5_out_dir = os.path.join(OUTPUT_DIR, "H5")
    for f_id in range(1, FOLD_NUM + 1):
        mask = (t_fold_ids == f_id)
        if not np.any(mask): continue
        with h5py.File(os.path.join(h5_out_dir, f"{REGION_NAME}_train_fold_{f_id}.h5"), 'w') as f:
            f.create_dataset('static_data', data=t_stat[mask], compression='gzip')
            f.create_dataset('dynamic_data', data=t_dyn[mask], compression='gzip')
            f.create_dataset('labels', data=t_lab_log[mask], compression='gzip')
            f.create_dataset('fids', data=t_fids[mask], compression='gzip')
            
    # 输出验证集与统计表
    with h5py.File(os.path.join(h5_out_dir, f"{REGION_NAME}_test_normalized.h5"), 'w') as f:
        f.create_dataset('static_data', data=v_stat, compression='gzip')
        f.create_dataset('dynamic_data', data=v_dyn, compression='gzip')
        f.create_dataset('labels', data=v_lab_log, compression='gzip')
        f.create_dataset('fids', data=v_fids, compression='gzip')
        
    with open(os.path.join(h5_out_dir, f"{REGION_NAME}_normalization_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)

    # 第五步与第六步：导出属性表 CSV 与物理 SHP (由于与原脚本逻辑相同，直接平移构建)
    print(f"\n[5/6] 正在提取中心像元构建 CSV 属性表...")
    table_data = []
    half_win = WINDOW_SIZE // 2
    for pt in final_points:
        if pt['fid'] not in results_dict: continue
        row = {
            'PID': pt['PID'], 'FID_Orig': pt['fid'], 'SOC': pt['target'], 
            'Set': pt['set'], 'Fold': pt['fold'], 
            'Lon': pt['lon'], 'Lat': pt['lat']
        }
        res = results_dict[pt['fid']]
        static_c = res['static'][:, half_win, half_win]
        dyn_vals = res['dynamic']
        for idx in range(7): row[f'RSI_B{idx+1}'] = static_c[idx]
        for idx, (name, _) in enumerate(STATIC_VARS): row[name] = static_c[7 + idx]
        for idx, (name, _) in enumerate(DYNAMIC_VARS):
            abbr = name.replace('Mean', 'Mn').replace('Max', 'Mx').replace('Std', 'Sd')
            for b in range(12): row[f'{abbr}_{b+1}'] = dyn_vals[idx, b]
        table_data.append(row)
        
    df = pd.DataFrame(table_data).sort_values(by='PID')
    df_train, df_val = df[df['Set'] == 'Train'], df[df['Set'] == 'Val']
    df_train.to_csv(os.path.join(OUTPUT_DIR, "CSV", f"{REGION_NAME}_Train_Features.csv"), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, "CSV", f"{REGION_NAME}_Val_Features.csv"), index=False)

    print(f"\n[6/6] 正在固化对应的独立 Shapefile...")
    def create_shapefile(df_subset, shp_name):
        shp_out = os.path.join(OUTPUT_DIR, "SHP", shp_name)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(shp_out): driver.DeleteDataSource(shp_out)
        out_ds = driver.CreateDataSource(shp_out)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        out_layer = out_ds.CreateLayer("data", srs, ogr.wkbPoint)
        
        for field in ["PID", "FID_Orig", "Fold"]: out_layer.CreateField(ogr.FieldDefn(field, ogr.OFTInteger))
        for field in ["SOC", "Lon", "Lat"]: out_layer.CreateField(ogr.FieldDefn(field, ogr.OFTReal))
        out_layer.CreateField(ogr.FieldDefn("Set", ogr.OFTString))
        for fn in df_subset.columns[7:]: out_layer.CreateField(ogr.FieldDefn(fn[:10], ogr.OFTReal))
            
        for row in df_subset.to_dict('records'):
            feat = ogr.Feature(out_layer.GetLayerDefn())
            geom = ogr.Geometry(ogr.wkbPoint)
            geom.AddPoint(row['Lon'], row['Lat'])
            feat.SetGeometry(geom)
            for k in ["PID", "FID_Orig", "SOC", "Set", "Fold", "Lon", "Lat"]: feat.SetField(k, row[k])
            for fn in df_subset.columns[7:]: feat.SetField(fn[:10], float(row[fn]))
            out_layer.CreateFeature(feat)
        out_ds = None

    create_shapefile(df_train, f"{REGION_NAME}_Train_Features.shp")
    create_shapefile(df_val, f"{REGION_NAME}_Val_Features.shp")
    create_shapefile(df, f"{REGION_NAME}_All_Samples.shp")

    print("\n" + "="*80)
    print("Dataloader 数据池构建与预加载任务完成")
    print("="*80)

if __name__ == "__main__":
    main()