import numpy as np
import pandas as pd

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

    return interp_depths, center_offset  # 返回插值后数据和平移偏移量

def reconstruct_beam_profile():
    # ===== STEP 1 & 2: 加载数据并平移 =====
    x_profile, _ = load_and_resample_csv(
        'x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv'
    )
    y_profile_raw, y_shift = load_and_resample_csv(
        'y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv'
    )
    
    # ===== STEP 3: 重组y_crosssect（行向量） =====
    # 注意：原始y_profile是-15~15（升序），需反转为15~-15（匹配beamprofile的行序）
    y_profile = y_profile_raw[::-1]  # 第0行对应y=15，第30行对应y=-15
    
    # ====== STEP 4 & 5: 初始化beamprofile及相关向量 ======
    beam_profile = np.zeros((31, 31))
    
    # 计算归一化因子
    one_over_31_x = x_profile / 31.0
    sum_x = np.sum(x_profile)
    unit_x = x_profile / sum_x
    
    # STEP 6: 初始填充beamprofile
    for i in range(31):
        beam_profile[i, :] = one_over_31_x
    
    # 行向量字典（按题目要求命名）
    row_vectors = {
        f"{i+1}th_row_y_is_{15-i}mm": beam_profile[i,:].copy()
        for i in range(31)
    }
    
    # ===== STEP 7-9: 计算行和、列和、卷积差 =====
    def calculate_and_display_diffs():
        nonlocal beam_profile
        # 行和（卷积x方向）
        row_sums = np.sum(beam_profile, axis=1)
        # 列和（卷积y方向）
        col_sums = np.sum(beam_profile, axis=0)
        
        # 计算差异向量
        diff_x = col_sums - x_profile  # 与x_crosssect的差
        diff_y = row_sums - y_profile  # 与y_crosssect的差
        
        # 按题目要求打印
        print("\n当前卷积差异:")
        print(f"x_crosssect差异: {np.round(diff_x, 4)}")
        print(f"y_crosssect差异: {np.round(diff_y, 4)}\n")
        
        return diff_y, col_sums, row_sums
    
    # ====== 迭代重构核心 ======
    adjusted = np.full(31, False)  # 记录已调整过的行
    
    while np.sum(~adjusted) > 0:  # 当还有未处理行时继续
        # 计算当前行和差异
        diff_y_vec, conv_y, row_sums = calculate_and_display_diffs()
        
        # STEP 10：查找最需要调整的行
        unadjusted_rows = np.where(~adjusted)[0]
        if len(unadjusted_rows) == 0:
            break
        
        # 在未调整行中找最大绝对差异（非0）
        # 策略：优先正值，再按边缘优先
        candidate_vals = diff_y_vec[unadjusted_rows]
        max_val = np.max(candidate_vals)  # 最大值
        
        # 处理多个同最大值的情形
        max_candidates = unadjusted_rows[diff_y_vec[unadjusted_rows] == max_val]
        if len(max_candidates) > 1:
            # 计算每个候选行到中心的距离（边缘度）
            edge_distances = np.abs(max_candidates - 15)
            max_edge_idx = np.argmax(edge_distances)
            selected_row = max_candidates[max_edge_idx]
            
            # 如果边缘度相同取索引小的
            if np.sum(edge_distances == edge_distances[max_edge_idx]) > 1:
                selected_row = np.min(
                    max_candidates[edge_distances == edge_distances[max_edge_idx]]
                )
        else:
            selected_row = max_candidates[0]
        
        # STEP 11：调整目标行
        max_diff_val = diff_y_vec[selected_row]
        if np.abs(max_diff_val) < 1e-3 and "最后一次处理" not in locals():
            continue  # 跳过已小于1e-6的优化项
            
        print(f"调整行: y={15-selected_row}mm, 差异值={max_diff_val:.6f}")
        
        beam_profile[selected_row] -= max_diff_val * unit_x
        adjusted[selected_row] = True
        row_vectors[f"{selected_row+1}th_row_y_is_{15-selected_row}mm"] = beam_profile[selected_row]
        
        # STEP 12：补偿其他行
        unadjusted_rows = np.where(~adjusted)[0]
        if len(unadjusted_rows) > 0:
            compensation = (max_diff_val / len(unadjusted_rows)) * unit_x
            for row_idx in unadjusted_rows:
                beam_profile[row_idx] += compensation
                # 更新行向量
                row_vectors[f"{row_idx+1}th_row_y_is_{15-row_idx}mm"] = beam_profile[row_idx]
        
        # 处理最后一行 (STEP 14)
        if np.sum(~adjusted) == 0 and len(unadjusted_rows) == 1:
            # 特殊处理最后一行（确保列和不变）
            row_idx = unadjusted_rows[0]
            max_final_diff = diff_y_vec[row_idx]
            beam_profile[row_idx] -= max_final_diff * unit_x
            compensation = (max_final_diff / 30) * unit_x
            for comp_idx in range(31):
                if comp_idx != row_idx:
                    beam_profile[comp_idx] += compensation
            adjusted[row_idx] = True
            break

    # ==== 最终输出 ====
    # 检查所有行是否完成处理
    print("="*50)
    print("重构完成！最终差异：")
    diff_y_vec, _, _ = calculate_and_display_diffs()
    
    # STEP 15: 输出csv
    np.savetxt('reconstructed_beamprofile.csv', beam_profile, delimiter=',')
    print('已保存beamprofile到: reconstructed_beamprofile.csv')
    
    # 返回带命名的行向量（按题目要求）
    return beam_profile, row_vectors

# 执行重构
if __name__ == '__main__':
    final_beam, row_vector_dict = reconstruct_beam_profile()
