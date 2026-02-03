import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from scipy.interpolate import interp1d

def process_recipe(file_path):
    # 读取数据
    points_data = []
    x_set = set()
    y_set = set()
    header_found = False
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        
        # 表头检测
        for row in reader:
            if not row:
                continue
                
            if len(row) >= 5:
                header_candidates = [col.strip().lower() for col in row[:5]]
                required_headers = {"point", "x-position", "x-speed", "y-position", "y-speed"}
                
                if required_headers.issubset(set(header_candidates)):
                    header_found = True
                    break
        
        if not header_found:
            print("错误：未找到有效表头")
            return None, None, None, None
        
        # 数据处理
        for row in reader:
            if not row:
                continue
                
            # 检查结束行 (放在数据提取之前)
            if len(row) >= 5:
                try:
                    # 精确匹配结束行 (0,0,0,0,0)
                    if all(abs(float(val)) < 1e-10 for val in row[:5]):
                        break
                except:
                    pass
            
            # 数据提取 (所有单位都是mm)
            try:
                point = int(row[0])
                x_pos = float(row[1])  # 单位为mm
                y_pos = float(row[3])  # 单位为mm
                y_speed = float(row[4])  # 单位为mm/s
                
                # 添加坐标验证和零速度检查
                if abs(x_pos) < 1e-10 or abs(y_pos) < 1e-10 or abs(y_speed) < 1e-10:
                    continue
                    
                # 坐标唯一性检查 (添加容差比较)
                coordinate_exists = any(
                    abs(existing[0]-x_pos) < 1e-10 and 
                    abs(existing[1]-y_pos) < 1e-10 
                    for existing in points_data
                )
                if coordinate_exists:
                    continue
                    
                points_data.append((x_pos, y_pos, y_speed))
                x_set.add(x_pos)
                y_set.add(y_pos)
            except (ValueError, IndexError):
                continue
    
    if not points_data:
        print("错误：未找到有效数据点")
        return None, None, None, None
    
    # 创建有序坐标列表 (所有单位mm)
    x_coords = sorted(x_set)
    y_coords = sorted(y_set)
    
    # 坐标精度检查
    min_x = min(x_coords) if x_coords else 0
    min_y = min(y_coords) if y_coords else 0
    print(f"原始数据范围 (单位mm): X: [{min(x_coords):.6f}, {max(x_coords):.6f}], Y: [{min_y:.6f}, {max(y_coords):.6f}]")
    if min_x < 1e-10 or min_y < 1e-10:
        print(f"警告：检测到接近零值坐标 (x_min={min_x:.6e}, y_min={min_y:.6e})")
    
    # === 1. 创建速度矩阵 ===
    speed_matrix = np.full((len(y_coords), len(x_coords)), np.nan)
    dwell_matrix = np.full((len(y_coords), len(x_coords)), np.nan)  # 原始网格的停留时间矩阵
    
    x_to_index = {x: idx for idx, x in enumerate(x_coords)}
    y_to_index = {y: idy for idy, y in enumerate(y_coords)}
    
    for x, y, speed in points_data:
        if x in x_to_index and y in y_to_index:
            idx_x = x_to_index[x]
            idx_y = y_to_index[y]
            speed_matrix[idx_y, idx_x] = speed
            # 停留时间 = 1/速度 (单位秒)
            dwell_matrix[idx_y, idx_x] = 1.0 / speed if abs(speed) > 1e-10 else np.nan
    
    # === 2. 生成速度分布热力图和CSV ===
    speed_heatmap, speed_csv = generate_heatmap(
        speed_matrix, x_coords, y_coords, 
        'Y-Speed (mm/s)', 'coolwarm', '_y_speed', file_path
    )
    
    # === 3. 创建1mm网格的停留时间分布 (单位mm) ===
    # 确定Y坐标的最小值和最大值（单位mm）
    min_y_val = min(y_coords)
    max_y_val = max(y_coords)
    
    # 创建1mm间隔的Y坐标网格 (单位mm)
    # 计算需要多少个点：从最小值到最大值，每1mm一个点
    num_points = int(np.round(max_y_val - min_y_val)) + 1
    y_coords_1mm = np.linspace(min_y_val, min_y_val + num_points - 1, num_points)
    
    print(f"停留时间网格 (单位mm): Y范围: [{y_coords_1mm[0]:.6f}, {y_coords_1mm[-1]:.6f}], 点数={len(y_coords_1mm)}")
    
    # 创建新的停留时间矩阵（1mm网格，单位秒）
    dwell_matrix_1mm = np.full((len(y_coords_1mm), len(x_coords)), np.nan)
    
    # 对每一列(x位置)进行Y方向的插值 (使用原始网格)
    for col_idx, x_val in enumerate(x_coords):
        # 获取当前列的非NaN数据
        valid_mask = ~np.isnan(dwell_matrix[:, col_idx])
        valid_y = np.array(y_coords)[valid_mask]
        valid_dwell = dwell_matrix[valid_mask, col_idx]
        
        if len(valid_y) < 2:
            # 少于2个有效点，无法插值
            continue
            
        # 创建线性插值函数
        interp_fn = interp1d(
            valid_y, valid_dwell, 
            kind='linear', 
            bounds_error=False, 
            fill_value=np.nan
        )
        
        # 在当前列上进行插值到1mm网格
        dwell_matrix_1mm[:, col_idx] = interp_fn(y_coords_1mm)
    
    # === 4. 生成停留时间分布热力图和CSV ===
    dwell_heatmap, dwell_csv = generate_heatmap(
        dwell_matrix_1mm, x_coords, y_coords_1mm, 
        'Dwell Time (s)', 'viridis', '_y_dwell_time', file_path,
        is_dwell_time=True
    )
    
    return speed_heatmap, speed_csv, dwell_heatmap, dwell_csv

def generate_heatmap(matrix, x_coords, y_coords, data_label, cmap_name, suffix, file_path, is_dwell_time=False):
    """生成热力图并保存为图像和CSV文件"""
    # 生成文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    heatmap_file = f"{base_name}{suffix}_heatmap.png"
    csv_file = f"{base_name}{suffix}_distribution.csv"
    
    # 生成热力图
    plt.figure(figsize=(12, 9))
    
    if is_dwell_time:
        # 停留时间用对数色标，因为值可能差异很大
        norm = 'log' if np.nanmin(matrix) > 0 else None
        cmap = 'viridis'
    else:
        norm = None
        cmap = 'coolwarm'
    
    # 计算数据范围（忽略NaN）
    try:
        vmin = np.nanmin(matrix)
        vmax = np.nanmax(matrix)
    except:
        vmin = 0
        vmax = 1
    
    # 热力图绘图
    img = plt.pcolormesh(
        x_coords, y_coords, matrix, 
        shading='auto', 
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )
    
    # 颜色条设置
    cbar = plt.colorbar(img, shrink=0.8)
    
    if is_dwell_time:
        title_label = 'Dwell Time Distribution (1mm grid)'
        cbar.set_label(f'Dwell Time Range (s): [{vmin:.4e}, {vmax:.4e}]')
        plt.ylabel('Y-Position (mm)')
    else:
        title_label = 'Y-Speed Distribution'
        cbar.set_label(f'Y-Speed Range (mm/s): [{vmin:.4f}, {vmax:.4f}]')
        plt.ylabel('Y-Position (mm)')
    
    plt.title(title_label, fontsize=14, pad=20)
    plt.xlabel('X-Position (mm)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 设置科学计数法格式化
    plt.gca().ticklabel_format(axis='both', style='sci', scilimits=(-3, 4))
    plt.gca().xaxis.get_offset_text().set_fontsize(10)
    plt.gca().yaxis.get_offset_text().set_fontsize(10)
    
    plt.tight_layout()
    plt.savefig(heatmap_file, dpi=150)
    plt.close()
    print(f"{title_label}热力图已保存为: {heatmap_file}")
    
    # 生成CSV文件
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # CSV表头 - 特别处理第一列显示"Coordinate" 
        header_row = ['Coordinate (mm)'] + [f"{x:.6f}" for x in x_coords]
        writer.writerow(header_row)
        
        # 数据体 - Y值在每行的第一列
        for i, y_val in enumerate(y_coords):
            row_data = [f"{y_val:.6f}"]  # 位置保留6位小数
            for j in range(len(x_coords)):
                val = matrix[i, j]
                if is_dwell_time and np.isnan(val):
                    row_data.append('NaN')  # 停留时间NaN特殊处理
                elif val is not None and not np.isnan(val):
                    # 根据值大小智能格式化
                    if abs(val) < 1e-6 or abs(val) > 1e6:  # 非常大或非常小的值
                        row_data.append(f"{val:.6e}")
                    elif abs(val) < 1e-3:
                        row_data.append(f"{val:.10f}")
                    else:
                        row_data.append(f"{val:.6f}")
                else:
                    row_data.append('')  # 数值为空
            writer.writerow(row_data)
    
    print(f"{title_label} CSV文件已保存为: {csv_file}")
    
    return heatmap_file, csv_file

# 使用示例
if __name__ == '__main__':
    input_file = 'recipe.csv'  # 替换为您的文件路径
    results = process_recipe(input_file)
    
    if all(result is not None for result in results):
        speed_heatmap, speed_csv, dwell_heatmap, dwell_csv = results
        print("\n处理成功!")
        print(f"Y轴速度分布: 热力图={speed_heatmap}, CSV={speed_csv}")
        print(f"停留时间分布: 热力图={dwell_heatmap}, CSV={dwell_csv}")
    else:
        print("处理失败，请检查输入文件")
