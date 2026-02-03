import numpy as np
import pandas as pd
import os

def load_and_resample_csv(filepath, total_points=31, shift_range=15):
    data = pd.read_csv(filepath, header=None)
    positions = data.iloc[:, 0].values
    depths = data.iloc[:, 1].values

    # 找到深度最大值的位置
    peak_idx = np.argmax(depths)
    center_offset = positions[peak_idx]
    
    # 平移所有位置让最大值位于0点
    shifted_positions = positions - center_offset
    
    # 创建新的插值坐标（1mm间隔）
    x_new = np.linspace(-shift_range, shift_range, total_points)
    
    # 使用线性插值获取新位置的数据
    interp_depths = np.interp(
        x_new, 
        shifted_positions, 
        depths, 
        left=0.0, 
        right=0.0
    )

    return interp_depths, center_offset

def reconstruct_beam_profile_and_save_iterations():
    # 创建存放迭代结果的文件夹
    if not os.path.exists('beamprofile_iterations'):
        os.makedirs('beamprofile_iterations')
    
    # 加载数据并重采样
    x_profile, _ = load_and_resample_csv(
        'x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv'
    )
    y_profile_raw, y_shift = load_and_resample_csv(
        'y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv'
    )
    
    # 反转y_profile数据以匹配顺序
    y_profile = y_profile_raw[::-1]
    
    # 初始化beamprofile和相关向量
    beam_profile = np.zeros((31, 31))
    
    # 计算归一化因子
    one_over_31_x = x_profile / 31.0
    sum_x = np.sum(x_profile)
    unit_x = x_profile / sum_x
    
    # 初始填充beamprofile
    for i in range(31):
        beam_profile[i, :] = one_over_31_x
    
    # 创建行向量字典
    row_vectors = {}
    for i in range(31):
        y_val = 15 - i
        row_vectors[f"{i+1}th_row_y_is_{y_val}mm"] = beam_profile[i,:].copy()
    
    # 初始化迭代计数器和调整状态数组
    iteration = 0
    adjusted = np.zeros(31, dtype=bool)  # 跟踪哪些行已调整
    
    def save_current_beamprofile(iter_num):
        """保存当前beamprofile到文件"""
        filepath = f"beamprofile_iterations/iteration_{iter_num}_beamprofile.csv"
        df = pd.DataFrame(beam_profile)
        
        # 添加行索引名称和列索引名称
        rows = [f"{i+1}th_row_y_is_{15-i}mm" for i in range(31)]
        cols = [f"{j+1}th_col_x_is_{j-15}mm" for j in range(31)]
        
        df.index = rows
        df.columns = cols
        df.to_csv(filepath)
        print(f"已保存迭代 {iter_num} 的beamprofile到: {filepath}")
    
    # 保存初始状态 (迭代0)
    save_current_beamprofile(iteration)
    
    def calculate_differences():
        # 行和（卷积x方向）
        row_sums = np.sum(beam_profile, axis=1)
        # 列和（卷积y方向）
        col_sums = np.sum(beam_profile, axis=0)
        
        # 计算差异向量
        diff_x = col_sums - x_profile
        diff_y = row_sums - y_profile
        
        return diff_y, col_sums, row_sums
    
    # ====== 迭代重构核心 ======
    while np.sum(adjusted) < 31:  # 当还有未处理行时继续
        iteration += 1
        # 计算当前行和差异
        diff_y_vec, conv_y, row_sums = calculate_differences()
        
        # 查找最需要调整的行
        unadjusted_indices = np.where(~adjusted)[0]
        if len(unadjusted_indices) == 0:
            break
        
        # 在未调整行中找最大绝对差异
        max_val = -np.inf
        selected_row = -1
        
        for i in unadjusted_indices:
            # 跳过已被视为0的差异（但该行尚未调整）
            if abs(diff_y_vec[i]) < 1e-6:
                continue
                
            if diff_y_vec[i] > max_val:
                max_val = diff_y_vec[i]
                selected_row = i
            elif diff_y_vec[i] == max_val:
                # 处理相同差异值：优先选择离中心更远的行
                dist_current = abs(i - 15)
                dist_selected = abs(selected_row - 15)
                
                if dist_current > dist_selected:
                    selected_row = i
                elif dist_current == dist_selected:
                    # 离中心距离相同，选择索引值小的
                    selected_row = min(i, selected_row)
        
        if selected_row == -1:
            # 所有未调整行的差异值都近似0
            break
            
        max_diff_val = diff_y_vec[selected_row]
        
        print(f"\n迭代 {iteration}: 调整行 y={15-selected_row}mm, 差异值={max_diff_val:.6f}")
        
        # 调整目标行
        beam_profile[selected_row] -= max_diff_val * unit_x
        
        # 更新行向量
        y_val = 15 - selected_row
        row_vectors[f"{selected_row+1}th_row_y_is_{y_val}mm"] = beam_profile[selected_row]
        adjusted[selected_row] = True
            
        # 补偿其他未调整行（如果还有未调整行）
        other_rows = np.where(~adjusted)[0]
        if len(other_rows) > 0:
            # 计算补偿量
            compensation_per_row = (max_diff_val / len(other_rows)) * unit_x
            
            # 应用补偿到所有未调整行
            for i in other_rows:
                beam_profile[i] += compensation_per_row
                
                # 更新行向量
                y_val = 15 - i
                row_vectors[f"{i+1}th_row_y_is_{y_val}mm"] = beam_profile[i]
        
        # 保存当前迭代的beamprofile
        save_current_beamprofile(iteration)
        
        # 打印简要信息
        print(f"已调整行数: {np.sum(adjusted)}/31")
    
    # 最终保存
    iteration += 1
    save_current_beamprofile(iteration)
    
    # 输出最终差异
    final_diff_y, _, _ = calculate_differences()
    print("\n最终y方向卷积差异:", np.round(final_diff_y, 6))
    
    # 最终输出
    save_current_beamprofile("final")
    
    return beam_profile, row_vectors

# 执行重构
if __name__ == '__main__':
    print("开始重构离子束形状...")
    final_beam, row_vector_dict = reconstruct_beam_profile_and_save_iterations()
    print("重构完成! 所有beamprofile已保存在beamprofile_iterations文件夹中")
