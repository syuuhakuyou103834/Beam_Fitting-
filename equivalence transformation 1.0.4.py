import numpy as np
from scipy.interpolate import RectBivariateSpline
import pandas as pd
import math
import os

def interpolate_grid(data, factor=10):
    """将网格分辨率从1mm提升到0.1mm"""
    n = data.shape[0]
    x = np.linspace(-15, 15, n)
    y = np.linspace(-15, 15, n)
    
    # 计算新网格点数：从31点计算应得301点
    n_new = (n - 1) * factor + 1
    
    interp_fn = RectBivariateSpline(x, y, data)
    x_fine = np.linspace(-15, 15, n_new)
    y_fine = np.linspace(-15, 15, n_new)
    return interp_fn(x_fine, y_fine)

def fwhm_1d(profile):
    """计算一维分布的真实FWHM"""
    max_val = np.max(profile)
    if max_val <= 1e-6:
        return 0
    
    half_max = max_val * 0.5
    above_half = np.where(profile >= half_max)[0]
    
    if len(above_half) < 2:
        return 0
    
    left_idx = above_half[0]
    right_idx = above_half[-1]
    
    # 左侧线性插值
    if left_idx > 0:
        left_diff = profile[left_idx] - profile[left_idx - 1]
        left_interp = left_idx - (profile[left_idx] - half_max) / (left_diff + 1e-10)
    else:
        left_interp = left_idx
        
    # 右侧线性插值
    if right_idx < len(profile) - 1:
        right_diff = profile[right_idx] - profile[right_idx + 1]
        right_interp = right_idx + (profile[right_idx] - half_max) / (right_diff + 1e-10)
    else:
        right_interp = right_idx
        
    return (right_interp - left_interp) * 0.1  # 转换为mm单位

def directional_fwhm(grid, angle_deg):
    """计算指定方向上的FWHM"""
    size = grid.shape[0]
    center = (size - 1) / 2  # 中心点坐标（可能是浮点数）
    angle_rad = math.radians(angle_deg)
    
    # 增加采样密度
    max_len = int(15 * math.sqrt(2) * 10) * 2  # 每边扩展
    line_points = []
    
    # 沿方向均匀采样密集点
    for d in np.linspace(-max_len, max_len, max_len * 2 + 1):
        # 转换为网格坐标
        x = center + d * math.cos(angle_rad) * 0.1  # 步长0.1mm
        y = center + d * math.sin(angle_rad) * 0.1
        
        if 0 <= x < size-1 and 0 <= y < size-1:
            i, j = int(x), int(y)
            di, dj = x - i, y - j
            
            # 双线性插值
            val = (1-di)*(1-dj)*grid[i,j] 
            val += di*(1-dj)*grid[i+1,j] 
            val += (1-di)*dj*grid[i,j+1] 
            val += di*dj*grid[i+1,j+1]
            line_points.append(val)
    
    return fwhm_1d(np.array(line_points))

def fwhm_map(grid):
    """计算所有指定方向上的FWHM"""
    angles = [0, 15, 45, 75, 90]
    return [directional_fwhm(grid, angle) for angle in angles]

def ensure_monotonicity(grid, strictness=0.95):
    """确保网格满足物理规律：中心强，边缘弱，单峰"""
    size = grid.shape[0]
    center = size // 2
    center_val = grid[center, center]
    
    # 计算每个点到中心的距离
    distances = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dist = math.sqrt((i-center)**2 + (j-center)**2)
            distances[i, j] = dist
    
    # 按距离排序所有点
    flat_indices = np.argsort(distances.ravel())
    sorted_vals = grid.ravel()[flat_indices]
    sorted_dists = distances.ravel()[flat_indices]
    
    # 强制增加单调性：外围点的值不大于内圈点的值
    max_val_so_far = float('-inf')
    for idx in range(len(sorted_vals)):
        current_val = sorted_vals[idx]
        current_dist = sorted_dists[idx]
        
        # 跳过距离为0的中心点
        if abs(current_dist) < 1e-3:
            max_val_so_far = current_val
            continue
            
        # 如果当前距离变化不大，跳过
        if idx > 0 and abs(current_dist - sorted_dists[idx-1]) < 1e-3:
            if current_val > max_val_so_far:
                max_val_so_far = current_val
            continue
        
        # 如果当前点值大于内圈最大值，则限制其值
        if current_val > max_val_so_far:
            if max_val_so_far > 0:
                # 允许轻微突破但设置上限
                new_val = min(current_val, max_val_so_far * (1 + 0.1/(current_dist+1)))
            else:
                new_val = current_val
            sorted_vals[idx] = new_val
            max_val_so_far = new_val
        else:
            max_val_so_far = current_val
    
    # 重构网格
    new_grid = np.zeros_like(grid)
    new_grid.ravel()[flat_indices] = sorted_vals
    return new_grid

def equivalence_transformation(data):
    """等价变换算法：减少FWHM同时保持行列和不变"""
    size = data.shape[0]
    center = size // 2
    
    # 计算原始行列和
    orig_row_sums = np.sum(data, axis=1)
    orig_col_sums = np.sum(data, axis=0)
    total_sum = np.sum(orig_row_sums)
    
    # 计算每个点的"影响力权重"（中心点权重高，边缘点权重低）
    weights = np.zeros_like(data)
    for i in range(size):
        for j in range(size):
            # 曼哈顿距离 + 对角线距离的混合度量
            dist = math.sqrt((i-center)**2 + (j-center)**2)
            weights[i, j] = 1 / (1 + dist**1.5)  # 指数增加权重差异
    
    # 原始FWHM
    high_res = interpolate_grid(data)
    fwhms = fwhm_map(high_res)
    print(f"Initial FWHM: {[round(f, 2) for f in fwhms]}")
    
    # 如果主要方向FWHM已经达标则跳过变换
    if max(fwhms[0], fwhms[-1], fwhms[1], fwhms[3]) < 3.5:
        print("FWHM already within limits. No transformation needed.")
        return data, orig_row_sums, orig_col_sums
    
    iterations = 0
    max_iter = 30
    print("Starting equivalence transformation...")
    
    # 克隆原始数据作为参考
    working_data = data.copy()
    
    while max(fwhms[0], fwhms[-1]) >= 3.5 and iterations < max_iter:
        iterations += 1
        
        # 动态调整中心点增强幅度（根据当前最大FWHM）
        max_fwhm = max(fwhms)
        boost_factor = min(0.2 + (max_fwhm - 3.5) * 0.02, 0.25)
        center_boost = boost_factor * total_sum * 0.0008
        
        # 中心点增强
        working_data[center, center] += center_boost
        
        # 计算权重总值用于分配减幅（排除中心点）
        weight_total = np.sum(weights) - weights[center, center]
        
        # 按权重比例减少外围点值
        reduction_per_weight = center_boost / (weight_total + 1e-10)
        for i in range(size):
            for j in range(size):
                if i != center or j != center:
                    reduction_amount = reduction_per_weight * weights[i, j]
                    # 保证值不低于初始值的70%或0.0
                    min_val = max(0.70 * data[i, j], 0.0)
                    working_data[i, j] = max(min_val, working_data[i, j] - reduction_amount)
        
        # 行总和矫正（确保每行和不变）
        for r in range(size):
            current_row_sum = np.sum(working_data[r, :])
            diff = orig_row_sums[r] - current_row_sum
            
            # 差值超过0.1%时才校正
            if abs(diff) > orig_row_sums[r] * 0.001:
                # 计算每列的权重（当前值越大权重越高）
                row_weights = working_data[r, :] / (current_row_sum + 1e-10)
                
                # 按比例分配差值
                adjustment = diff * row_weights
                working_data[r, :] += adjustment
        
        # 列总和矫正（确保每列和不变）
        for c in range(size):
            current_col_sum = np.sum(working_data[:, c])
            diff = orig_col_sums[c] - current_col_sum
            
            # 差值超过0.1%时才校正
            if abs(diff) > orig_col_sums[c] * 0.001:
                col_weights = working_data[:, c] / (current_col_sum + 1e-10)
                adjustment = diff * col_weights
                working_data[:, c] += adjustment
        
        # 强制物理属性（单峰分布）
        working_data = ensure_monotonicity(working_data)
        
        # 计算新FWHM
        high_res = interpolate_grid(working_data)
        new_fwhms = fwhm_map(high_res)
        
        # 打印进度（重点关注0°和90°）
        print(f"Iteration {iterations}: FWHM [0°={new_fwhms[0]:.2f}, 90°={new_fwhms[4]:.2f}]")
        
        # 更新FWHM
        fwhms = new_fwhms
        
        # 提前终止条件：主要方向FWHM达标
        if max(fwhms[0], fwhms[1], fwhms[3], fwhms[4]) < 3.5:
            break
    
    # 返回变换后数据和原始行列和
    return working_data, orig_row_sums, orig_col_sums

def calculate_row_col_sums(data):
    """计算行和与列和"""
    row_sums = np.sum(data, axis=1)
    col_sums = np.sum(data, axis=0)
    return row_sums, col_sums

def save_row_col_report(file_path, pre_row, pre_col, post_row, post_col, tolerance=1e-3):
    """保存行和列和的报告并检查守恒性"""
    n = len(pre_row)
    with open(file_path, 'w') as f:
        f.write("=== Row and Column Sum Report ===\n")
        f.write(f"Tolerance for conservation check: ±{tolerance}\n\n")
        
        f.write("Row Sums (Before -> After):\n")
        max_row_diff = 0.0
        for i in range(n):
            diff = abs(pre_row[i] - post_row[i])
            max_row_diff = max(max_row_diff, diff)
            status = "Conserved" if diff < tolerance else f"Difference: {diff:.6f}"
            f.write(f"Row {i+1:2d}: {pre_row[i]:10.6f} -> {post_row[i]:10.6f} | {status}\n")
        
        f.write(f"\nMax row difference: {max_row_diff:.6f}\n")
        
        f.write("\nColumn Sums (Before -> After):\n")
        max_col_diff = 0.0
        for j in range(n):
            diff = abs(pre_col[j] - post_col[j])
            max_col_diff = max(max_col_diff, diff)
            status = "Conserved" if diff < tolerance else f"Difference: {diff:.6f}"
            f.write(f"Col {j+1:2d}: {pre_col[j]:10.6f} -> {post_col[j]:10.6f} | {status}\n")
        
        f.write(f"\nMax column difference: {max_col_diff:.6f}\n")
        
        f.write("\n=== Conservation Summary ===\n")
        if max_row_diff < tolerance and max_col_diff < tolerance:
            f.write("SUCCESS: All row and column sums conserved within tolerance.")
        else:
            f.write("WARNING: Some row or column sums changed beyond tolerance!")

def main():
    print("=== Beam Profile Transformation ===")
    
    # 0. 获取当前脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "raw_beamprofile.csv")
    
    # 1. 从CSV文件读取数据
    print(f"Reading data from {input_file}...")
    
    # 读取数据（假设CSV文件是纯数据，没有表头）
    grid_1mm = pd.read_csv(input_file, header=None).values
    
    # 验证数据大小
    if grid_1mm.shape != (31, 31):
        print(f"Error: Expected 31x31 grid, found {grid_1mm.shape}")
        return
    
    print("Data loaded successfully.")
    
    # 2. 计算原始行和与列和
    orig_row_sums, orig_col_sums = calculate_row_col_sums(grid_1mm)
    
    # 3. 将数据范围从-15到15标记
    distances_1mm = np.linspace(-15, 15, 31)
    
    # 4. 插值到0.1mm分辨率
    high_res_grid = interpolate_grid(grid_1mm)
    
    # 5. 计算变换前FWHM
    angles = [0, 15, 45, 75, 90]
    original_fwhm = []
    for angle in angles:
        fwhm_val = directional_fwhm(high_res_grid, angle)
        original_fwhm.append(round(fwhm_val, 4))
    
    print("\n=== ORIGINAL FWHM (mm) ===")
    for angle, fwhm_val in zip(angles, original_fwhm):
        print(f"Angle {angle:2d}°: {fwhm_val:.2f}")
    
    # 6. 等价变换
    transformed_1mm, pre_row_sums, pre_col_sums = equivalence_transformation(grid_1mm)
    
    # 7. 计算变换后的行和与列和
    post_row_sums, post_col_sums = calculate_row_col_sums(transformed_1mm)
    
    # 8. 生成变换后的高分辨率网格
    transformed_high_res = interpolate_grid(transformed_1mm)
    
    # 9. 计算变换后FWHM
    transformed_fwhm = []
    for angle in angles:
        fwhm_val = directional_fwhm(transformed_high_res, angle)
        transformed_fwhm.append(round(fwhm_val, 4))
    
    print("\n=== TRANSFORMED FWHM (mm) ===")
    for angle, fwhm_val in zip(angles, transformed_fwhm):
        print(f"Angle {angle:2d}°: {fwhm_val:.2f}")
    
    # 10. 输出结果文件
    # 计算高分辨率网格范围
    n_high = transformed_high_res.shape[0]
    fine_points = np.linspace(-15, 15, n_high)
    
    # 高分辨率CSV (0.1mm) - 包含坐标
    pd.DataFrame(
        transformed_high_res, 
        index=np.round(fine_points, 1),
        columns=np.round(fine_points, 1)
    ).to_csv("transformed_high_res.csv", float_format='%.4f')
    
    print("Saved high-resolution (0.1mm) data to transformed_high_res.csv (with coordinates)")
    
    # 低分辨率CSV (1mm) - 仅数值矩阵，不包含坐标
    pd.DataFrame(transformed_1mm).to_csv(
        "transformed_low_res.csv", 
        header=False,     # 不保存列索引
        index=False,      # 不保存行索引
        float_format='%.4f'
    )
    
    print("Saved low-resolution (1mm) data to transformed_low_res.csv (numerical matrix only)")
    
    # 11. 生成FWHM报告
    with open("fwhm_report.txt", "w") as f:
        f.write("=== FWHM Analysis Report ===\n\n")
        
        f.write("Original FWHM (mm):\n")
        for angle, val in zip(angles, original_fwhm):
            f.write(f"{angle:2d}°: {val:.4f}, ")
        
        f.write("\n\nTransformed FWHM (mm):\n")
        for angle, val in zip(angles, transformed_fwhm):
            f.write(f"{angle:2d}°: {val:.4f}, ")
        
        f.write("\n\n=== Transformation Details ===\n")
        f.write(f"Original grid: 31×31 points (1mm resolution)\n")
        f.write(f"Transformed grid: 31×31 points (1mm resolution)\n")
        f.write(f"High-res grid: {n_high}×{n_high} points (0.1mm resolution)\n")
        
        # 计算FWHM变化百分比
        fwhm_changes = []
        for orig, trans in zip(original_fwhm, transformed_fwhm):
            if orig > 0:
                change_percent = (trans - orig) / orig * 100
                fwhm_changes.append(change_percent)
            else:
                fwhm_changes.append(0)
        
        f.write("\nFWHM Change Percentage (%):\n")
        for angle, change in zip(angles, fwhm_changes):
            f.write(f"{angle:2d}°: {change:.1f}%, ")
    
    print("Saved FWHM report to fwhm_report.txt")
    
    # 12. 生成行列和报告
    save_row_col_report(
        "row_col_sum_report.txt", 
        pre_row_sums, pre_col_sums, 
        post_row_sums, post_col_sums
    )
    print("Saved row/column sum report to row_col_sum_report.txt")

if __name__ == "__main__":
    main()
