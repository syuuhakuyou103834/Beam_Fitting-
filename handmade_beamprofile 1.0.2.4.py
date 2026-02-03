import numpy as np
import pandas as pd
import os

# 创建迭代过程输出目录
os.makedirs('iteration', exist_ok=True)

def load_and_resample_csv(filepath, total_points=31, shift_range=15):
    # 读取CSV文件，没有表头
    data = pd.read_csv(filepath, header=None)
    positions = data.iloc[:, 0].values
    depths = data.iloc[:, 1].values

    # 找到深度最大值的位置
    peak_idx = np.argmax(depths)
    center_offset = positions[peak_idx]  # 计算中心偏移量
    
    # 平移所有位置让最大值位于0点
    shifted_positions = positions - center_offset
    
    # 创建新的插值坐标（1mm间隔）
    x_new = np.linspace(-shift_range, shift_range, total_points)
    
    # 使用线性插值获取新位置的数据（边界外填充0）
    interp_depths = np.interp(
        x_new, 
        shifted_positions, 
        depths, 
        left=0.0, 
        right=0.0
    )

    # 返回插值后数据、平移偏移量和位置坐标
    return interp_depths, center_offset, x_new

def save_shifted_profile(positions, depths, filename):
    """保存平移后的截面深度分布"""
    df = pd.DataFrame({
        'Position (mm)': positions,
        'Etching Depth': depths
    })
    df.to_csv(filename, index=False)
    print(f"已保存平移后的截面数据至: {filename}")

def save_beamprofile_with_diffs(beam_profile, x_profile, y_profile, iteration, folder='iteration'):
    """保存beamprofile矩阵并添加差异信息"""
    rows, cols = beam_profile.shape
    
    # 计算行和（卷积x方向）和列和（卷积y方向）
    row_sums = np.sum(beam_profile, axis=1)
    col_sums = np.sum(beam_profile, axis=0)
    
    # 计算与实验值的差异
    col_diffs = col_sums - x_profile  # 列和与x_crosssect差异
    row_diffs = row_sums - y_profile  # 行和与y_crosssect差异
    
    # 扩展矩阵添加差异列
    extended_beam = np.zeros((rows + 1, cols + 1))
    extended_beam[:rows, :cols] = beam_profile
    
    # 添加列差异（最后一行）
    extended_beam[rows, :cols] = col_diffs
    
    # 添加行差异（最后一列）
    extended_beam[:rows, cols] = row_diffs
    
    # 创建数据框（添加行列标签）
    df = pd.DataFrame(extended_beam)
    
    # 添加行列标签
    y_labels = [f"y={15-i}mm" for i in range(rows)] + ['Y-Conv Diff']
    df.insert(0, 'Row/Y-Position', y_labels)
    
    x_labels = [f"x={i-15}mm" for i in range(cols)] + ['X-Conv Diff']
    df.columns = ['Position'] + x_labels
    
    # 保存到文件
    # 处理迭代次数的显示
    iter_str = f"{iteration:03d}" if isinstance(iteration, int) else iteration
    filename = os.path.join(folder, f"beamprofile_iteration_{iter_str}.csv")
    df.to_csv(filename, index=False)
    print(f"已保存迭代{iter_str}的beamprofile至: {filename}")

def save_initial_profiles(x_pos, x_data, y_pos, y_data, filename="initial_shifted_profiles.csv"):
    """保存用于计算循环的初始截面数据"""
    # 反转y_data以匹配beamprofile的行顺序
    reversed_y_data = y_data[::-1]
    
    # 创建数据框
    df = pd.DataFrame({
        'Position (mm)': np.concatenate([x_pos, y_pos]),
        'Etching Depth': np.concatenate([x_data, reversed_y_data]),
        'Profile Type': ['x_crosssect'] * len(x_pos) + ['y_crosssect'] * len(y_pos)
    })
    
    # 保存到文件
    df.to_csv(filename, index=False)
    print(f"已保存初始截面数据至: {filename}")

def reconstruct_beam_profile():
    iteration_count = 0
    
    # ===== STEP 1 & 2: 加载数据并平移 =====
    x_profile, x_offset, x_positions = load_and_resample_csv(
        'x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv'
    )
    y_profile_raw, y_offset, y_positions = load_and_resample_csv(
        'y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv'
    )
    
    # 保存平移后的截面数据
    save_shifted_profile(x_positions, x_profile, 'translated_x_crosssection.csv')
    save_shifted_profile(y_positions, y_profile_raw, 'translated_y_crosssection.csv')
    
    # 保存组合的初始截面数据
    save_initial_profiles(
        x_positions, x_profile, 
        y_positions, y_profile_raw
    )
    
    # ===== STEP 3: 重组y_crosssect（行向量） =====
    # 注意：原始y_profile是-15~15（升序），需反转为15~-15（匹配beamprofile的行序）
    y_profile = y_profile_raw[::-1]  # 第0行对应y=15，第30行对应y=-15
    
    # ====== STEP 4 & 5: 初始化beamprofile及相关向量 ======
    beam_profile = np.zeros((31, 31))
    
    # 计算归一化因子
    one_over_31_x = x_profile / 31.0
    sum_x = np.sum(x_profile)
    unit_of_x_crosssect = x_profile / sum_x
    
    # STEP 6: 初始填充beamprofile
    for i in range(31):
        beam_profile[i, :] = one_over_31_x
    
    # 保存初始状态（第0次迭代）
    save_beamprofile_with_diffs(beam_profile, x_profile, y_profile, iteration_count)
    
    # 行向量字典（按题目要求命名）
    row_vectors = {
        f"{i+1}th_row_y_is_{15-i}mm": beam_profile[i,:].copy()
        for i in range(31)
    }
    
    # ====== 重构核心函数 - 模拟需求中的步骤 ======
    def calculate_diffs():
        """STEP 7-9: 计算行和、列和、卷积差"""
        # 行和（卷积x方向）
        row_sums = np.sum(beam_profile, axis=1)
        # 列和（卷积y方向）
        col_sums = np.sum(beam_profile, axis=0)
        
        # 计算差异向量
        diff_x = col_sums - x_profile  # 与x_crosssect的差 (difference_of_x_crosssect)
        diff_y = row_sums - y_profile  # 与y_crosssect的差 (difference_of_y_crosssect)
        
        # 数字小于1e-6强制设为0 (STEP 11要求)
        diff_y[np.abs(diff_y) < 1e-6] = 0
        
        return diff_y, diff_x
    
    # 标记已处理的行
    processed_rows = np.full(31, False)
    
    # 迭代直到只剩下最后一行未处理
    while np.sum(~processed_rows) > 1:
        iteration_count += 1
        print(f"\n===== 开始迭代 {iteration_count} =====")
        
        # 计算当前差异
        diff_y, diff_x = calculate_diffs()
        
        # STEP 10: 找出最大差异行 (最需要处理的行)
        # 只考虑未处理的行
        unprocessed_indices = np.where(~processed_rows)[0]
        candidate_diffs = diff_y[unprocessed_indices]
        
        # 找到最大正差异值（真实大小）
        max_val = np.max(candidate_diffs)
        max_candidates = unprocessed_indices[abs(diff_y[unprocessed_indices]-max_val) < 1e-6]
        
        # 处理多个同最大值的情形（优先边缘行）
        if len(max_candidates) > 1:
            # 计算每个候选行到中心的距离（边缘度）
            dist_to_center = np.abs(max_candidates - 15)
            # 选择最远的边缘行
            selected_row = max_candidates[np.argmax(dist_to_center)]
        else:
            selected_row = max_candidates[0] if len(max_candidates) == 1 else unprocessed_indices[0]
        
        max_diff_val = diff_y[selected_row]
        y_position = 15 - selected_row
        print(f"处理行: y={y_position}mm (第{selected_row+1}行), 差异值={max_diff_val:.6f}")
        
        # STEP 11: 调整目标行
        beam_profile[selected_row] -= max_diff_val * unit_of_x_crosssect
        processed_rows[selected_row] = True
        row_vectors[f"{selected_row+1}th_row_y_is_{y_position}mm"] = beam_profile[selected_row]
        
        # 强制设为0（STEP 11要求）
        diff_y[selected_row] = 0
        
        # 更新差异
        diff_y, diff_x = calculate_diffs()
        
        # STEP 12: 计算需要补偿的行数
        unprocessed_rows = np.where(~processed_rows)[0]
        n_compensation_rows = len(unprocessed_rows)
        
        if n_compensation_rows > 0:
            # 对所有未处理行进行补偿
            compensation = (max_diff_val / n_compensation_rows) * unit_of_x_crosssect
            for row_idx in unprocessed_rows:
                beam_profile[row_idx] += compensation
                # 更新行向量
                y_pos = 15 - row_idx
                row_vectors[f"{row_idx+1}th_row_y_is_{y_pos}mm"] = beam_profile[row_idx]
        
        # 保存当前迭代状态
        save_beamprofile_with_diffs(beam_profile, x_profile, y_profile, iteration_count)
        
        # STEP 14检查点：如果只剩一行未处理，跳出循环
        unprocessed_rows = np.where(~processed_rows)[0]
        if len(unprocessed_rows) == 1:
            print("检测到只剩一行未处理，将执行最终优化")
            break
    
    # ==== STEP 14: 专门处理最后一行 =====
    if len(np.where(~processed_rows)[0]) == 1:
        iteration_count += 1
        print(f"\n===== 开始最终优化迭代 {iteration_count} =====")
        
        # 找出最后未处理的行（中心行）
        last_row_index = np.where(~processed_rows)[0][0]
        y_pos_value = 15 - last_row_index
        
        # 计算当前差异
        diff_y, diff_x = calculate_diffs()
        
        # 获取该行的差异值
        max_final_diff = diff_y[last_row_index]
        print(f"处理最后中心行: y={y_pos_value}mm (第{last_row_index+1}行), 差异值={max_final_diff:.6f}")
        
        # STEP 14-A: 修正中心行
        beam_profile[last_row_index] -= max_final_diff * unit_of_x_crosssect
        processed_rows[last_row_index] = True
        row_vectors[f"{last_row_index+1}th_row_y_is_{y_pos_value}mm"] = beam_profile[last_row_index]
        
        # STEP 14-B: 对所有其他30行进行补偿
        compensation = (max_final_diff / 30) * unit_of_x_crosssect
        for row_idx in range(31):
            if row_idx != last_row_index:
                beam_profile[row_idx] += compensation
                # 更新行向量
                y_pos = 15 - row_idx
                row_vectors[f"{row_idx+1}th_row_y_is_{y_pos}mm"] = beam_profile[row_idx]
        
        # 保存最后一次迭代状态
        save_beamprofile_with_diffs(beam_profile, x_profile, y_profile, iteration_count)
    
    # ==== 最终输出前处理：负数修正 ====
    print("\n检查beamprofile中是否有负值...")
    
    # 检查矩阵中的负值
    negative_count = np.sum(beam_profile < 0)
    min_value = np.min(beam_profile)
    
    if negative_count > 0:
        print(f"发现 {negative_count} 个负值元素，最小值={min_value:.10f}")
        print("正在进行非负处理（将负数设为0）...")
        
        # 克隆原始数据用于记录变化
        original_beam = beam_profile.copy()
        
        # 标记需要调整的位置
        negative_mask = beam_profile < 0
        negative_positions = np.argwhere(negative_mask)
        
        # 打印负数位置
        print(f"负数元素位置(y, x):")
        for pos in negative_positions:
            y_idx, x_idx = pos
            y_value = 15 - y_idx
            x_value = x_idx - 15
            value = beam_profile[y_idx, x_idx]
            print(f"    y={y_value:+d}mm, x={x_value:+d}mm: {value:.10f}")
        
        # 将负数设为0
        beam_profile[negative_mask] = 0
        
        # 计算并显示修正量
        correction_amount = -np.sum(original_beam[negative_mask])
        print(f"负数修正总量（绝对值之和）: {correction_amount:.10f}")
    else:
        print(f"未发现负数元素，最小值={min_value:.10f}")
    
    # ===== 更新行向量 =====
    print("更新行向量字典...")
    for i in range(31):
        y_pos = 15 - i
        row_vectors[f"{i+1}th_row_y_is_{y_pos}mm"] = beam_profile[i, :]
    
    # ==== 最终输出 ====
    # 计算最终差异
    print("\n===== 最终结果 =====")
    final_diff_y, final_diff_x = calculate_diffs()
    print(f"最终y_convdiff: {np.round(final_diff_y, 6)}")
    print(f"最终x_convdiff: {np.round(final_diff_x, 6)}")
    
    # 保存最终beamprofile
    save_beamprofile_with_diffs(beam_profile, x_profile, y_profile, "final")
    np.savetxt('reconstructed_beamprofile.csv', beam_profile, delimiter=',')
    print('已保存最终beamprofile到: reconstructed_beamprofile.csv')
    
    # ==== 额外保存负数位置信息 ====
    if negative_count > 0:
        # 准备负数位置数据
        negative_y = []
        negative_x = []
        negative_values = []
        for pos in negative_positions:
            y_idx, x_idx = pos
            y_value = 15 - y_idx
            x_value = x_idx - 15
            value = original_beam[y_idx, x_idx]
            negative_y.append(y_value)
            negative_x.append(x_value)
            negative_values.append(value)
        
        # 创建负数位置数据框
        neg_df = pd.DataFrame({
            'Y Position (mm)': negative_y,
            'X Position (mm)': negative_x,
            'Original Value': negative_values
        })
        neg_df.to_csv('negative_corrections.csv', index=False)
        print('已保存负数修正信息至: negative_corrections.csv')
    
    return beam_profile, row_vectors

# 执行重构
if __name__ == '__main__':
    final_beam, row_vector_dict = reconstruct_beam_profile()
