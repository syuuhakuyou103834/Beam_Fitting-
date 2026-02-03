import numpy as np
import pandas as pd
import os

def load_and_process_data(filename):
    """加载数据并进行平移重采样处理"""
    df = pd.read_csv(filename, header=None, names=['position', 'depth'])
    positions = df['position'].values
    depths = df['depth'].values
    
    # 找到深度最大值的位置
    max_idx = np.argmax(depths)
    max_pos = positions[max_idx]
    offset = max_pos  # 需要平移的偏移量
    
    # 平移位置并创建新位置网格 (-15到15mm, 0.25mm步长)
    new_positions = positions - offset
    fine_grid = np.arange(-15, 15.01, 0.25)
    fine_depths = np.interp(fine_grid, new_positions, depths, 
                            left=0, right=0)  # 超出范围填充0
    
    # 在1mm网格上重采样 (-15到15mm, 31个点)
    coarse_grid = np.linspace(-15, 15, 31)
    coarse_depths = np.interp(coarse_grid, fine_grid, fine_depths)
    
    return coarse_depths

# 加载和处理X轴运动数据（Y截面）
x_crosssect = load_and_process_data('x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv')

# 加载和处理Y轴运动数据（X截面）
y_crosssect = load_and_process_data('y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv')
y_crosssect = y_crosssect.reshape(-1, 1)  # 转换为列向量

# 计算必要参数
sum_of_x_crosssect = np.sum(x_crosssect)
unit_of_x_crosssect = x_crosssect / sum_of_x_crosssect
one_over_31_x_crosssect = x_crosssect / 31.0

# 初始化beamprofile (31x31)
beamprofile = np.tile(one_over_31_x_crosssect, (31, 1))

# 创建命名字典用于追踪每行/列
row_vectors = {}
col_vectors = {}
for i in range(31):
    y_val = 15 - i  # y坐标: 第0行=15mm, 第30行=-15mm
    row_vectors[f"{i+1}th_row_y_is_{int(y_val)}mm"] = beamprofile[i, :].copy()
    
for j in range(31):
    x_val = -15 + j  # x坐标: 第0列=-15mm, 第30列=15mm
    col_vectors[f"{j+1}th_col_x_is_{int(x_val)}mm"] = beamprofile[:, j].copy()

# 迭代重构过程
processed_rows = set()
epsilon = 1e-6

for iteration in range(30):
    # 计算当前卷积值
    convolution_of_y_dir = np.sum(beamprofile, axis=0)  # 列求和 (行向量)
    convolution_of_x_dir = np.sum(beamprofile, axis=1).reshape(-1, 1)  # 行求和 (列向量)
    
    # 计算差异
    difference_of_x_crosssect = convolution_of_y_dir - x_crosssect
    difference_of_y_crosssect = convolution_of_x_dir - y_crosssect
    
    # 将足够小的差异设为0
    diff_abs = np.abs(difference_of_y_crosssect)
    difference_of_y_crosssect[diff_abs < epsilon] = 0
    
    # 打印当前差异状态
    print(f"\n迭代 {iteration+1} - Y方向差异值:")
    for i, diff in enumerate(difference_of_y_crosssect[:, 0]):
        y_val = 15 - i
        print(f"y_{i}: {y_val}mm → {diff:.8f}")
    
    # 找到最大差异值及行索引
    # 排除已处理的行和接近0的行
    valid_indices = np.where(difference_of_y_crosssect[:, 0] != 0)[0]
    
    # 如果所有行都已处理，结束迭代
    if len(valid_indices) == 0:
        break
    
    # 找出最大真实值（正数优先）
    max_val = -np.inf
    max_idx = -1
    for idx in valid_indices:
        val = difference_of_y_crosssect[idx, 0]
        if val > max_val:
            max_val = val
            max_idx = idx
        elif np.isclose(val, max_val, atol=epsilon):
            # 并列最大值时，选择距离中心更远（y坐标绝对值更大）的行
            current_y = abs(15 - max_idx)
            candidate_y = abs(15 - idx)
            # 或者两个都在边缘(15mm或-15mm)，选择索引小的行
            if candidate_y > current_y or (candidate_y == current_y and idx < max_idx):
                max_val = val
                max_idx = idx
    
    row_idx = max_idx
    max_diff = max_val
    y_val = 15 - row_idx
    
    print(f"处理行 {row_idx+1}: y = {y_val}mm, 最大差异 = {max_diff:.8f}")
    
    # 行名索引
    row_name = f"{row_idx+1}th_row_y_is_{int(y_val)}mm"
    
    # 步骤6：调整该行数据
    adjustment = max_diff * unit_of_x_crosssect
    beamprofile[row_idx, :] -= adjustment
    row_vectors[row_name] = beamprofile[row_idx, :].copy()
    
    # 标记该行为已处理
    processed_rows.add(row_idx)
    
    # 计算需要补偿的行数
    non_zero_rows = [i for i in range(31) if difference_of_y_crosssect[i] != 0 and i != row_idx]
    n_rows = len(non_zero_rows)
    
    # 步骤7：其他行补偿调整
    if n_rows > 0:
        compensation = (max_diff / n_rows) * unit_of_x_crosssect
        
        for i in non_zero_rows:
            beamprofile[i, :] += compensation
            y_val_i = 15 - i
            row_name_i = f"{i+1}th_row_y_is_{int(y_val_i)}mm"
            row_vectors[row_name_i] = beamprofile[i, :].copy()
    
    # 更新列向量
    for j in range(31):
        x_val = -15 + j
        col_name = f"{j+1}th_col_x_is_{int(x_val)}mm"
        col_vectors[col_name] = beamprofile[:, j].copy()

# 处理最后一行（中心行）
convolution_of_x_dir = np.sum(beamprofile, axis=1).reshape(-1, 1)
difference_of_y_crosssect = convolution_of_x_dir - y_crosssect

# 找到未处理的最后一行（中心行y=0）
center_idx = 15
if center_idx not in processed_rows:
    center_diff = difference_of_y_crosssect[center_idx, 0]
    print(f"\n处理中心行: y=0mm, 差异={center_diff:.8f}")
    
    # 调整中心行
    center_row_name = "16th_row_y_is_0mm"
    adjustment = center_diff * unit_of_x_crosssect
    beamprofile[center_idx, :] -= adjustment
    row_vectors[center_row_name] = beamprofile[center_idx, :].copy()
    
    # 补偿其他30行
    compensation = (center_diff / 30) * unit_of_x_crosssect
    
    for i in range(31):
        if i != center_idx:
            y_val_i = 15 - i
            row_name_i = f"{i+1}th_row_y_is_{int(y_val_i)}mm"
            beamprofile[i, :] += compensation
            row_vectors[row_name_i] = beamprofile[i, :].copy()
    
    # 更新列向量
    for j in range(31):
        x_val = -15 + j
        col_name = f"{j+1}th_col_x_is_{int(x_val)}mm"
        col_vectors[col_name] = beamprofile[:, j].copy()

# 创建行向量名称的列表用于输出
rows_output = []
for i in range(31):
    y_val = 15 - i
    row_name = f"{i+1}th_row_y_is_{int(y_val)}mm"
    rows_output.append(beamprofile[i, :])

rows_output = np.vstack(rows_output)

# 输出最终结果
output_df = pd.DataFrame(rows_output)
x_labels = [f'{-15 + i}mm' for i in range(31)]
output_df.columns = x_labels
output_df.index = [f'{15 - i}mm' for i in range(31)]

output_df.to_csv('reconstructed_beamprofile.csv', index_label='Y\\X')
print("\n重构完成！结果已保存为 reconstructed_beamprofile.csv")
