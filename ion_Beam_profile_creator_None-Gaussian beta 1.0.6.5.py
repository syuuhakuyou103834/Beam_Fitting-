import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
import matplotlib as mpl

# ====================== 修复中文字体支持 ======================
def setup_plotting():
    """配置绘图环境，解决中文字体问题"""
    # 使用系统字体
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'],
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 300
    })
    
setup_plotting()

class BeamEfficiencyOptimizer:
    def __init__(self, beam_traced_x_axis, beam_traced_y_axis, initial_guess_path, grid_bound=15.0, grid_points=31):
        """初始化优化器"""
        # 创建日志文件
        self.log_file = open("beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("离子束刻蚀效率优化引擎启动 (单峰约束版)")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"输入文件说明:")
        self.log(f" - 离子束沿X轴移动时的Y方向截面数据: {beam_traced_x_axis}")
        self.log(f" - 离子束沿Y轴移动时的X方向截面数据: {beam_traced_y_axis}")
        self.log("=" * 30)
        
        # 网格系统
        self.grid_bound = grid_bound
        self.grid_points = grid_points = 31
        self.grid = np.linspace(-grid_bound, grid_bound, grid_points)
        self.grid_spacing = 2 * grid_bound / (grid_points - 1)
        
        # 加载初始猜测
        self.initial_guess_path = initial_guess_path
        self.load_initial_beam(initial_guess_path)
        
        # 加载实验数据
        self.beam_traced_x_axis = self.load_experimental_data(beam_traced_x_axis)  # 沿X轴移动 (测量Y截面)
        self.beam_traced_y_axis = self.load_experimental_data(beam_traced_y_axis)  # 沿Y轴移动 (测量X截面)
        
        # 优化参数
        self.scan_velocity = 30.0  # 扫描速度 (mm/s)
        self.opt_radius = 15.0  # 优化半径 (mm)，覆盖15mm半径
        self.max_mutations = 4  # 变异候选数

        # 创建优化掩膜
        self.create_optimization_mask()
        
        # 中心点位置
        self.center_i, self.center_j = self.find_center(self.initial_beam)
        self.log(f"初始中心点位置: ({self.grid[self.center_i]:.2f}, {self.grid[self.center_j]:.2f})")
        
        # 创建距离矩阵（以中心为原点）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_center = np.sqrt(
            (xx - self.grid[self.center_i])**2 + 
            (yy - self.grid[self.center_j])**2
        )
        
        # 创建11个阶段的点集掩膜
        self.num_stages = 11
        self.stage_masks = []  # 存储阶段信息
        self.create_stage_masks()
        
        self.stage_ends = []  # 存储阶段结束点的迭代索引
        
        # 历史记录
        self.history = {
            "iteration": [],
            "abs_error": [],
            "rel_err_x": [],
            "rel_err_y": [],
            "max_val": [],
            "peak_violations": []
        }
        self.optimized_beam = self.initial_beam / self.max_val  # 初始优化束流
        
        # 应用初始约束确保中心辐射状分布
        self.optimized_beam, violations = self.enforce_radial_constraints(self.optimized_beam, strict=True)
        self.log(f"初始径向约束完成，修正峰值违规: {violations}处")
        
        # 应用初始行/列单峰约束
        self.optimized_beam, row_violations, col_violations = self.enforce_unimodal_row_col_constraints(self.optimized_beam)
        self.log(f"初始单峰约束完成，行违规: {row_violations}处, 列违规: {col_violations}处")

    def log(self, message):
        """记录带时间戳的日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_file.write(log_entry + "\n")
        self.log_file.flush()

    def find_center(self, beam_matrix):
        """找到束流中心点位置"""
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        return center_i, center_j

    def load_initial_beam(self, file_path):
        """加载初始束流分布"""
        self.log(f"加载初始束流猜测: {file_path}")
        try:
            self.initial_beam = np.genfromtxt(file_path, delimiter=",")
            if self.initial_beam.shape != (31, 31):
                self.log(f"错误: 初始束流尺寸应为31x31，实际为{self.initial_beam.shape}")
                raise ValueError("初始束流尺寸不匹配")
                
            self.max_val = np.max(self.initial_beam)
            if self.max_val == 0:
                raise ValueError("最大刻蚀速率为零，无效输入")
                
            # 归一化束流(保持比例)
            self.optimized_beam = self.initial_beam / self.max_val
            self.log(f"最大刻蚀速率: {self.max_val:.2f} nm/s")
            
            # 检查初始束流的单峰性
            row_violations, col_violations = self.check_unimodal_violations(self.optimized_beam)
            if row_violations + col_violations > 0:
                self.log(f"警告: 初始束流存在行/列峰违规: {row_violations}行, {col_violations}列")
            
        except Exception as e:
            self.log(f"加载失败: {str(e)}")
            raise

    def load_experimental_data(self, file_path):
        """加载实验轮廓数据"""
        self.log(f"加载实验数据: {file_path}")
        try:
            # 自动检测CSV格式
            data = np.loadtxt(file_path, delimiter=",")
            
            # 确保数据有位置和值两列
            if data.shape[1] != 2:
                # 尝试处理单列数据
                data = data.reshape((-1, 2))
                
            self.log(f"加载数据点: {len(data)} 个, 位置范围: [{data[0,0]:.2f}, {data[-1,0]:.2f}] mm")
            return data
        except Exception as e:
            self.log(f"加载失败: {str(e)}")
            raise

    def create_optimization_mask(self):
        """创建优化区域掩膜"""
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.distance_from_center = np.sqrt(xx**2 + yy**2)
        self.optimization_mask = (self.distance_from_center <= self.opt_radius)
        self.log(f"优化区域包含 {np.sum(self.optimization_mask)} 个点 (半径 ≤ {self.opt_radius}mm)")

    def create_stage_masks(self):
        """创建11个阶段的点集掩膜"""
        self.stage_masks = []
        total_mask = np.zeros((self.grid_points, self.grid_points), dtype=bool)
        
        # 生成所有网格坐标
        xx, yy = np.meshgrid(range(self.grid_points), range(self.grid_points), indexing='ij')
        
        # 第一阶段 (集合1): x=0 或 y=0 的点
        stage0_mask = ((np.abs(self.grid[xx]) < 1e-5) | (np.abs(self.grid[yy]) < 1e-5))
        self.stage_masks.append(stage0_mask)
        total_mask = total_mask | stage0_mask
        self.log(f"集合1 (阶段0): x=0或y=0的点, 包含点数: {np.sum(stage0_mask)}")
        
        # 第二阶段到第十阶段 (集合2~集合10): |x|=k 或 |y|=k 的点，排除之前阶段
        for k in range(1, 10):
            k_value = float(k)
            # 定义位置阈值（考虑浮点误差）
            threshold = self.grid_spacing / 2
            # 找到坐标绝对值约为k的位置
            stage_mask = (
                (np.abs(np.abs(self.grid[xx]) - k_value) < threshold) | 
                (np.abs(np.abs(self.grid[yy]) - k_value) < threshold)
            )
            # 排除之前阶段已选择的点
            stage_mask = stage_mask & ~total_mask
            self.stage_masks.append(stage_mask)
            total_mask = total_mask | stage_mask
            self.log(f"集合{k+1} (阶段{k}: |x|={k}或|y|={k}的点, 包含点数: {np.sum(stage_mask)}")
        
        # 第十一阶段 (集合11): 剩余所有点
        stage10_mask = ~total_mask
        self.stage_masks.append(stage10_mask)
        self.log(f"集合11 (阶段10): 剩余点, 包含点数: {np.sum(stage10_mask)}")
        
        # 当前阶段索引
        self.current_stage_idx = 0

    def create_interpolator(self, beam_matrix):
        """创建双线性插值器"""
        return RegularGridInterpolator(
            (self.grid, self.grid),
            beam_matrix * self.max_val,  # 反归一化
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
    
    def enforce_radial_constraints(self, beam_matrix, strict=False):
        """
        强制执行径向约束 - 强化版本
        strict=True: 使用更严格的约束
        返回修正后的矩阵和修正点数量
        """
        rows, cols = beam_matrix.shape
        
        # 更新中心点位置（如果最大值位置改变）
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        self.center_i, self.center_j = center_i, center_j
        center_pos = (self.grid[center_i], self.grid[center_j])
        max_val = beam_matrix[center_i, center_j]
        
        # 记录调整点数
        modified_points = 0
        
        # 1. 确保中心点位于物理中心
        center_distance = np.sqrt(center_pos[0]**2 + center_pos[1]**2)
        if center_distance > 1.5:
            self.log(f"中心点({center_pos[0]:.2f}, {center_pos[1]:.2f})偏离过大({center_distance:.2f}mm)，调整到附近最高点")
            
            # 在-1.0~1.0mm范围内寻找最大值
            region_i = np.where((np.abs(self.grid) < 1.0))[0]
            region_j = np.where((np.abs(self.grid) < 1.0))[0]
            
            if region_i.size > 0 and region_j.size > 0:
                region_values = beam_matrix[region_i[:, None], region_j]
                max_in_center = np.max(region_values)
                max_idx = np.unravel_index(np.argmax(region_values), (region_i.size, region_j.size))
                center_i, center_j = region_i[max_idx[0]], region_j[max_idx[1]]
                max_val = max_in_center
                center_pos = (self.grid[center_i], self.grid[center_j])
                self.log(f"新中心点位置: ({center_pos[0]:.2f}, {center_pos[1]:.2f}), 值={max_val:.4f}")
        
        # 2. 创建距离矩阵（以中心点为原点）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt((xx - center_pos[0])**2 + (yy - center_pos[1])**2)
        
        # 3. 中心区域（半径≤5mm）禁止零值点
        inner_region = r <= 5.0
        zero_points_mask = beam_matrix < max_val * 0.01
        inner_zero_points = np.logical_and(inner_region, zero_points_mask)
        zero_points_count = np.sum(inner_zero_points)
        
        if zero_points_count > 0:
            base_value = max_val * 0.02
            beam_matrix[inner_zero_points] = base_value
            modified_points += zero_points_count
            self.log(f"禁止零值点: 设置 {zero_points_count} 个点至少为 {base_value:.4f}")
        
        # 4. 沿射线方向强制单调递减
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)  # 36个方向
        decay_factor = 0.96  # 每单位距离的衰减因子
        
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            max_radius = self.opt_radius * 1.2
            
            # 计算射线上的点
            ray_points = []
            distances = np.linspace(0, max_radius, 100)  # 100个点
            for dist in distances:
                x_pos = center_pos[0] + dx * dist
                y_pos = center_pos[1] + dy * dist
                # 找到最近的网格点
                x_idx = np.argmin(np.abs(self.grid - x_pos))
                y_idx = np.argmin(np.abs(self.grid - y_pos))
                ray_points.append((x_idx, y_idx, dist))
            
            # 移除重复点
            unique_points = {}
            for x_idx, y_idx, dist in ray_points:
                key = (x_idx, y_idx)
                if key not in unique_points or dist < unique_points[key][0]:
                    unique_points[key] = (dist, x_idx, y_idx)
            
            # 按距离排序
            sorted_points = sorted(unique_points.values(), key=lambda x: x[0])
            
            # 应用单调递减约束
            prev_val = max_val
            for idx, (dist, x_idx, y_idx) in enumerate(sorted_points):
                current_val = beam_matrix[x_idx, y_idx]
                
                # 计算衰减目标值
                target_val = max_val * np.exp(-dist * (1 - decay_factor) / decay_factor)
                
                # 确保当前值不超过前一个点且符合目标值
                if current_val > min(prev_val, target_val * 1.2):
                    new_val = min(prev_val * 0.99, target_val * (1.0 + 0.1 * np.random.randn()))
                    beam_matrix[x_idx, y_idx] = max(0, min(new_val, prev_val))
                    modified_points += 1
                
                # 更新前一个值
                prev_val = beam_matrix[x_idx, y_idx]
        
        # 5. 确保所有值非负
        beam_matrix = np.maximum(beam_matrix, 0)
        
        return beam_matrix, modified_points

    def enforce_unimodal_row_col_constraints(self, beam_matrix):
        """
        强制执行行/列单峰约束，确保每个行和列只有一个峰值
        返回修正后的矩阵和行违规数、列违规数
        """
        rows, cols = beam_matrix.shape
        center_i, center_j = self.center_i, self.center_j
        
        row_violations = 0
        col_violations = 0
        
        # 对每一行应用单峰约束
        for i in range(rows):
            row = beam_matrix[i, :].copy()
            current_peak = center_j  # 初始假设峰值为中心列
            
            # 寻找实际峰值位置
            peak_idx = np.argmax(row)
            
            # 检查峰值位置是否合理（应在中心附近）
            if abs(peak_idx - center_j) > 3:  # 允许偏离中心点3个位置
                self.log(f"行{i}峰值偏离较大: 在位置{peak_idx} (中心:{center_j})")
                peak_idx = center_j  # 强制设为中心点位置
            
            # 检查峰值是否为局部极值点
            if not self.is_peak(row, peak_idx):
                # 调整到最近的合格位置
                peak_idx = self.find_nearest_peak(row, peak_idx)
                row[peak_idx] = np.max(row)  # 恢复最大值
                row_violations += 1
            
            # 确保从左到峰值位置非递减
            for j in range(1, peak_idx + 1):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1] * 0.99  # 轻微递减允许
                    row_violations += 1
            
            # 确保从峰值位置到右端非递增
            for j in range(peak_idx, cols - 1):
                if row[j + 1] > row[j]:
                    row[j + 1] = row[j] * 0.99  # 轻微递减允许
                    row_violations += 1
            
            beam_matrix[i, :] = row
        
        # 对每一列应用单峰约束
        for j in range(cols):
            col = beam_matrix[:, j].copy()
            current_peak = center_i  # 初始假设峰值为中心行
            
            # 寻找实际峰值位置
            peak_idx = np.argmax(col)
            
            # 检查峰值位置是否合理
            if abs(peak_idx - center_i) > 3:
                self.log(f"列{j}峰值偏离较大: 在位置{peak_idx} (中心:{center_i})")
                peak_idx = center_i
            
            # 检查峰值是否为局部极值点
            if not self.is_peak(col, peak_idx):
                peak_idx = self.find_nearest_peak(col, peak_idx)
                col[peak_idx] = np.max(col)
                col_violations += 1
            
            # 确保从上到峰值位置非递减
            for i in range(1, peak_idx + 1):
                if col[i] < col[i - 1]:
                    col[i] = col[i - 1] * 0.99
                    col_violations += 1
            
            # 确保从峰值位置到下端非递增
            for i in range(peak_idx, rows - 1):
                if col[i + 1] > col[i]:
                    col[i + 1] = col[i] * 0.99
                    col_violations += 1
            
            beam_matrix[:, j] = col
        
        return beam_matrix, row_violations, col_violations

    def is_peak(self, array, index):
        """
        检查给定索引是否为局部极大值点
        """
        n = len(array)
        # 边界检查
        if index == 0:
            return array[index] >= array[index + 1]
        elif index == n - 1:
            return array[index] >= array[index - 1]
        else:
            return (array[index] >= array[index - 1] and 
                    array[index] >= array[index + 1])

    def find_nearest_peak(self, array, current_index):
        """
        查找最近的合格峰值位置
        """
        n = len(array)
        # 向左找
        left_peak = current_index
        for i in range(current_index - 1, -1, -1):
            if self.is_peak(array, i):
                left_peak = i
                break
        
        # 向右找
        right_peak = current_index
        for i in range(current_index + 1, n):
            if self.is_peak(array, i):
                right_peak = i
                break
        
        # 返回最近的峰值位置
        left_dist = current_index - left_peak
        right_dist = right_peak - current_index
        
        if left_dist <= right_dist:
            return left_peak
        else:
            return right_peak

    def check_unimodal_violations(self, beam_matrix):
        """
        检查行/列单峰违规情况
        返回行违规数，列违规数
        """
        rows, cols = beam_matrix.shape
        row_violations = 0
        col_violations = 0
        
        # 检查行违规
        for i in range(rows):
            row = beam_matrix[i, :]
            peak = np.argmax(row)
            # 检查左边是否递减
            for j in range(1, peak):
                if row[j] < row[j-1]:
                    row_violations += 1
            # 检查右边是否递增
            for j in range(peak, cols-1):
                if row[j+1] > row[j]:
                    row_violations += 1
        
        # 检查列违规
        for j in range(cols):
            col = beam_matrix[:, j]
            peak = np.argmax(col)
            # 检查上边是否递减
            for i in range(1, peak):
                if col[i] < col[i-1]:
                    col_violations += 1
            # 检查下边是否递增
            for i in range(peak, rows-1):
                if col[i+1] > col[i]:
                    col_violations += 1
        
        return row_violations, col_violations

    def simulate_etching(self, beam_matrix, direction):
        """
        模拟指定方向的刻蚀轮廓
        direction = "x": 表示束沿Y轴方向移动 (测量X方向轮廓)
        direction = "y": 表示束沿X轴方向移动 (测量Y方向轮廓)
        """
        interpolator = self.create_interpolator(beam_matrix)
        profile = np.zeros_like(self.grid)
        
        # 创建路径点
        if direction == "x":  # 束沿Y轴移动 -> 测量沿X方向的轮廓
            for j in range(len(self.grid)):
                y_pos = self.grid[j]
                path_points = np.column_stack((self.grid, np.full_like(self.grid, y_pos)))
                etch_rates = interpolator(path_points)
                profile[j] = trapezoid(etch_rates, dx=self.grid_spacing)
        else:  # direction == "y": 束沿X轴移动 -> 测量沿Y方向的轮廓
            for i in range(len(self.grid)):
                x_pos = self.grid[i]
                path_points = np.column_stack((np.full_like(self.grid, x_pos), self.grid))
                etch_rates = interpolator(path_points)
                profile[i] = trapezoid(etch_rates, dx=self.grid_spacing)
        
        # 归一化
        max_profile = np.max(profile)
        return profile / max_profile if max_profile > 0 else profile

    def calculate_error(self, sim_x_scan, sim_y_scan):
        """
        计算模拟结果与实验数据的绝对误差
        使用整个区域的误差进行计算
        """
        # 处理束沿Y移动时的X方向轮廓
        exp_x_data = self.beam_traced_y_axis
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        
        # 归一化实验数据
        exp_x_max = np.max(exp_x_val) if np.max(exp_x_val) > 0 else 1.0
        exp_x_norm = exp_x_val / exp_x_max
        
        # 插值模拟数据到实验点位置
        sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_scan)
        
        # 计算X误差
        abs_dev_x = np.abs(sim_x_interp - exp_x_norm)
        
        # 处理束沿X移动时的Y方向轮廓
        exp_y_data = self.beam_traced_x_axis
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        
        # 归一化实验数据
        exp_y_max = np.max(exp_y_val) if np.max(exp_y_val) > 0 else 1.0
        exp_y_norm = exp_y_val / exp_y_max
        
        # 插值模拟数据到实验点位置
        sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_scan)
        
        # 计算Y误差
        abs_dev_y = np.abs(sim_y_interp - exp_y_norm)
        
        # 只有在实验数据非零时计算真正相对误差
        non_zero_x = exp_x_norm > 1e-5
        non_zero_y = exp_y_norm > 1e-5
        
        rel_err_x = np.mean(abs_dev_x[non_zero_x] / exp_x_norm[non_zero_x]) * 100 if np.any(non_zero_x) else 100.0
        rel_err_y = np.mean(abs_dev_y[non_zero_y] / exp_y_norm[non_zero_y]) * 100 if np.any(non_zero_y) else 100.0
        
        # 计算综合绝对误差
        abs_err_x = np.mean(abs_dev_x)
        abs_err_y = np.mean(abs_dev_y)
        abs_error = (abs_err_x + abs_err_y) / 2
        
        return abs_error, rel_err_x, rel_err_y

    def mutate_beam(self, beam_matrix, magnitude, sim_x, sim_y):
        """
        基于误差分析的定向变异策略 - 针对坐标位置点集优化
        策略：如果当前位置的仿真值小于实验值，增加束流值；反之减小
        """
        new_beam = beam_matrix.copy()
        current_mask = self.stage_masks[self.current_stage_idx]
        
        # 计算当前位置的仿真值与实验值的误差
        errors = self.calculate_point_errors(beam_matrix, sim_x, sim_y)
        
        # 获取当前区域的索引
        indices = np.where(current_mask)
        
        # 变异概率设置
        mutation_probability = 0.8 if self.current_stage_idx < 5 else 0.7
        
        # 中心值
        center_val = beam_matrix[self.center_i, self.center_j]
        
        for idx in range(len(indices[0])):
            i, j = indices[0][idx], indices[1][idx]
            
            # 以一定概率跳过变异
            if np.random.rand() > mutation_probability:
                continue
            
            # 计算误差方向（1表示需要增加，-1表示需要减少）
            error_direction = 1 if errors[i, j] < 0 else -1
            
            # 变异幅度（带随机性）- 靠近中心的点变异幅度大
            dist_to_center = self.r_center[i, j]
            dist_factor = np.exp(-dist_to_center / 6.0)
            effective_magnitude = magnitude * dist_factor * (0.8 + 0.4 * np.random.rand())
            
            # 应用变异
            mutation = error_direction * effective_magnitude
            new_val = new_beam[i, j] + mutation
            
            # 确保值在合理范围内
            if dist_to_center < 5.0:  # 靠近中心区域
                new_val = max(0, min(1.0, new_val))
            elif dist_to_center < 10.0:  # 中间区域
                new_val = max(0, min(0.7, new_val))
            else:  # 远离中心区域
                new_val = max(0, min(0.4, new_val))
                
            new_beam[i, j] = new_val
        
        # 应用径向约束
        radial_modified = 0
        new_beam, radial_modified = self.enforce_radial_constraints(new_beam, strict=True)
        
        # 应用行/列单峰约束
        row_violations, col_violations = 0, 0
        new_beam, row_violations, col_violations = self.enforce_unimodal_row_col_constraints(new_beam)
        
        return new_beam, radial_modified, row_violations, col_violations

    def calculate_point_errors(self, beam_matrix, sim_x, sim_y):
        """
        计算网格点上每一点的误差
        策略：比较仿真值和实验值的差异
        """
        # 网格点上的坐标
        positions = self.grid  # 网格点坐标
        
        # 计算X方向的误差
        exp_x_data = self.beam_traced_y_axis
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        exp_x_max = np.max(exp_x_val)
        exp_x_norm = exp_x_val / exp_x_max
        
        # 插值实验数据到网格位置
        exp_x_grid = np.interp(positions, exp_x_pos, exp_x_norm)
        
        # 计算Y方向的误差
        exp_y_data = self.beam_traced_x_axis
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        exp_y_max = np.max(exp_y_val)
        exp_y_norm = exp_y_val / exp_y_max
        
        # 插值实验数据到网格位置
        exp_y_grid = np.interp(positions, exp_y_pos, exp_y_norm)
        
        # 计算每个网格点的误差（平均X和Y方向的误差）
        errors = np.zeros((self.grid_points, self.grid_points))
        
        # 预先计算全局误差 - 提高效率
        x_err_global = sim_x - exp_x_grid
        y_err_global = sim_y - exp_y_grid
        
        for i in range(self.grid_points):
            for j in range(self.grid_points):
                # 获取当前点的位置
                pos_x = self.grid[i]
                pos_y = self.grid[j]
                
                # 计算高斯权重
                sigma = 1.5  # 带宽参数
                x_weight = np.exp(-(positions - pos_x)**2 / (2 * sigma**2))
                y_weight = np.exp(-(positions - pos_y)**2 / (2 * sigma**2))
                
                # 加权误差
                x_weighted_err = np.sum(x_err_global * x_weight) / np.sum(x_weight)
                y_weighted_err = np.sum(y_err_global * y_weight) / np.sum(y_weight)
                
                # 综合误差
                errors[i, j] = (x_weighted_err + y_weighted_err) / 2
        
        return errors

    def calculate_magnitude(self, current_iter, max_iters, stage_idx):
        """动态计算变异幅度"""
        # 基础公式
        min_mag = 0.015  # 最小变异幅度
        max_mag = 0.35   # 最大变异幅度
        
        # 阶段自适应调整
        stage_factor = 1.0 - (stage_idx / (self.num_stages - 1)) * 0.6
        
        # 迭代衰减
        decay_factor = max(0.2, 1.0 - current_iter / max_iters)
        
        # 基础幅度
        magnitude = max_mag * stage_factor * decay_factor
        
        # 添加随机波动
        magnitude *= 0.7 + np.random.rand() * 0.6
        
        return max(magnitude, min_mag)

    def optimize_stage(self, current_matrix, stage_idx, max_iterations):
        """执行单个阶段的优化"""
        self.current_stage_idx = stage_idx
        
        # 获取当前阶段描述
        if stage_idx == 0:
            stage_name = "集合1 (坐标轴点)"
        elif stage_idx < 10:
            stage_name = f"集合{stage_idx+1} (|坐标|={stage_idx}点)"
        else:
            stage_name = "集合11 (剩余点)"
        
        self.log(f"\n=== 开始优化阶段 {stage_idx+1}/{self.num_stages}: {stage_name} ===")
        self.log(f"当前阶段包含点数: {np.sum(self.stage_masks[stage_idx])}")
        
        # 检查当前矩阵的单峰性
        row_violations, col_violations = self.check_unimodal_violations(current_matrix)
        peak_violations = row_violations + col_violations
        self.history["peak_violations"].append(peak_violations)
        self.log(f"阶段开始时峰违规: {row_violations}行, {col_violations}列")
        
        # 获取当前最佳模拟轮廓
        try:
            sim_x = self.simulate_etching(current_matrix, "x")
            sim_y = self.simulate_etching(current_matrix, "y")
            abs_error, rel_err_x, rel_err_y = self.calculate_error(sim_x, sim_y)
        except Exception as e:
            self.log(f"模拟误差计算失败: {str(e)}")
            raise
        
        self.log(f"阶段初始误差: 绝对={abs_error:.4f}, 束X动时Y向误差={rel_err_x:.1f}%, 束Y动时X向误差={rel_err_y:.1f}%")
        
        best_abs_error = abs_error
        best_matrix = current_matrix.copy()
        stagnation_count = 0
        
        # 冻结前一阶段的区域
        frozen_mask = np.zeros_like(self.optimization_mask, dtype=bool)
        for i in range(stage_idx):
            frozen_mask = frozen_mask | self.stage_masks[i]
        
        for iteration in range(1, max_iterations + 1):
            # 计算变异幅度
            magnitude = self.calculate_magnitude(iteration, max_iterations, stage_idx)
            
            self.log(f"迭代 {iteration}: 阶段{stage_idx+1}/{self.num_stages}, 变异幅度={magnitude:.5f}")
            
            # 创建变异候选
            try:
                candidate, radial_modified, row_violations, col_violations = self.mutate_beam(
                    best_matrix, magnitude, sim_x, sim_y)
            except Exception as e:
                self.log(f"变异失败: {str(e)}")
                break
            
            # 记录约束应用情况
            self.log(f"约束应用: 径向调整={radial_modified}点, 行违规={row_violations}, 列违规={col_violations}")
            
            # 冻结之前优化过的区域
            candidate[frozen_mask] = best_matrix[frozen_mask]
            
            # 评估候选
            try:
                cand_sim_x = self.simulate_etching(candidate, "x")
                cand_sim_y = self.simulate_etching(candidate, "y")
                cand_abs_error, cand_rel_err_x, cand_rel_err_y = self.calculate_error(cand_sim_x, cand_sim_y)
            except Exception as e:
                self.log(f"评估候选失败: {str(e)}")
                break
            
            # 检查是否改进
            if cand_abs_error < best_abs_error:
                improvement = best_abs_error - cand_abs_error
                self.log(f"改进: Δ={improvement:.5f}, 新绝对误差={cand_abs_error:.4f}")
                self.log(f"相对误差变化: 束X动时Y向={cand_rel_err_x - self.history['rel_err_x'][-1]:+.1f}%, "
                        f"束Y动时X向={cand_rel_err_y - self.history['rel_err_y'][-1]:+.1f}%")
                
                best_abs_error = cand_abs_error
                best_matrix = candidate.copy()
                sim_x, sim_y = cand_sim_x, cand_sim_y
                stagnation_count = 0
                
                # 更新历史记录
                iter_num = self.history["iteration"][-1] + 1 if len(self.history["iteration"]) > 0 else 1
                self.history["iteration"].append(iter_num)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(cand_rel_err_x)
                self.history["rel_err_y"].append(cand_rel_err_y)
                
                # 检查优化后的单峰性
                row_violations, col_violations = self.check_unimodal_violations(best_matrix)
                peak_violations = row_violations + col_violations
                self.history["peak_violations"].append(peak_violations)
                self.log(f"峰违规检查: {row_violations}行, {col_violations}列")
                
                # 保存阶段结果
                self.optimized_beam = best_matrix
            else:
                stagnation_count += 1
                self.log(f"无改进，维持绝对误差: {best_abs_error:.4f}, 已连续 {stagnation_count} 次")
            
            # 检查收敛
            if best_abs_error < 0.01:
                self.log(f"阶段收敛：绝对误差{best_abs_error:.4f} < 0.01")
                break
            elif stagnation_count >= max(2, max_iterations//2):  # 提前退出机制
                self.log(f"连续{stagnation_count}次无改进，结束当前阶段优化")
                break
        
        # 最终确保无峰违规
        row_violations, col_violations = self.check_unimodal_violations(best_matrix)
        if row_violations + col_violations > 0:
            best_matrix, _, _ = self.enforce_unimodal_row_col_constraints(best_matrix)
            self.log(f"阶段结束前强制修正峰违规: 行={row_violations}, 列={col_violations}")
        
        self.log(f"阶段完成: 最佳绝对误差={best_abs_error:.4f}")
        return best_matrix

    def run_optimization(self):
        """运行11阶段优化过程"""
        # 初始评估
        current_matrix = self.optimized_beam.copy()
        sim_x = self.simulate_etching(current_matrix, "x")
        sim_y = self.simulate_etching(current_matrix, "y")
        abs_error0, rel_err_x0, rel_err_y0 = self.calculate_error(sim_x, sim_y)
        
        # 初始峰违规检查
        row_violations0, col_violations0 = self.check_unimodal_violations(current_matrix)
        peak_violations0 = row_violations0 + col_violations0
        
        # 添加初始状态到历史记录
        self.history["iteration"].append(0)
        self.history["abs_error"].append(abs_error0)
        self.history["rel_err_x"].append(rel_err_x0)
        self.history["rel_err_y"].append(rel_err_y0)
        self.history["peak_violations"].append(peak_violations0)
        
        start_time = time.time()
        
        # 设置每个阶段的迭代次数
        stage_iterations = [8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 5]
        
        # 执行11个阶段优化
        for stage_idx in range(self.num_stages):
            max_iter = stage_iterations[stage_idx] if stage_idx < len(stage_iterations) else 4
            current_matrix = self.optimize_stage(current_matrix, stage_idx, max_iter)
            self.stage_ends.append(len(self.history["iteration"]) - 1)
        
        # 最终评估
        self.optimized_beam = current_matrix
        optimized_beam_full = self.optimized_beam * self.max_val
        
        # 保存结果
        np.savetxt("optimized_beam_distribution.csv", optimized_beam_full, delimiter=",")
        
        # 最终误差评估
        sim_x = self.simulate_etching(self.optimized_beam, "x")
        sim_y = self.simulate_etching(self.optimized_beam, "y")
        final_abs_error, final_rel_err_x, final_rel_err_y = self.calculate_error(sim_x, sim_y)
        
        # 最终峰违规检查
        row_violations, col_violations = self.check_unimodal_violations(self.optimized_beam)
        peak_violations = row_violations + col_violations
        
        elapsed_time = time.time() - start_time
        self.log(f"\n优化完成! 总迭代次数: {len(self.history['iteration'])}")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        self.log(f"最终绝对误差: {final_abs_error:.4f} (初始={abs_error0:.4f})")
        self.log(f"最终相对误差: 束X动时Y向={final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%)")
        self.log(f"              束Y动时X向={final_rel_err_y:.1f}% (初始={rel_err_y0:.1f}%)")
        self.log(f"最终峰违规: {row_violations}行, {col_violations}列")
        
        # 绘制径向分布验证图
        self.plot_radial_distribution(optimized_beam_full)
        
        # 绘制单峰性验证图
        self.plot_unimodal_verification(optimized_beam_full)
        
        return self.optimized_beam, final_rel_err_x, final_rel_err_y

    def plot_unimodal_verification(self, beam_full):
        """绘制行/列单峰性验证图"""
        try:
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle("束流分布行/列单峰性验证", fontsize=16)
            
            # 1. 示例行分布
            ax1 = plt.subplot(221)
            row_idx = self.grid_points // 2  # 中间行
            row_data = beam_full[row_idx, :]
            ax1.plot(self.grid, row_data, 'b-', linewidth=1.5)
            ax1.scatter(self.grid, row_data, c='r', s=20)
            
            # 标注峰值位置
            peak_j = np.argmax(row_data)
            ax1.axvline(self.grid[peak_j], color='g', linestyle='--', alpha=0.7)
            ax1.scatter([self.grid[peak_j]], [row_data[peak_j]], s=100, marker='x', c='g')
            
            ax1.set_title(f"Y={self.grid[row_idx]:.1f}mm 行束流分布")
            ax1.set_xlabel("X位置 (mm)")
            ax1.set_ylabel("刻蚀速率 (nm/s)")
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # 2. 示例列分布
            ax2 = plt.subplot(222)
            col_idx = self.grid_points // 2  # 中间列
            col_data = beam_full[:, col_idx]
            ax2.plot(self.grid, col_data, 'b-', linewidth=1.5)
            ax2.scatter(self.grid, col_data, c='r', s=20)
            
            # 标注峰值位置
            peak_i = np.argmax(col_data)
            ax2.axvline(self.grid[peak_i], color='g', linestyle='--', alpha=0.7)
            ax2.scatter([self.grid[peak_i]], [col_data[peak_i]], s=100, marker='x', c='g')
            
            ax2.set_title(f"X={self.grid[col_idx]:.1f}mm 列束流分布")
            ax2.set_xlabel("Y位置 (mm)")
            ax2.set_ylabel("刻蚀速率 (nm/s)")
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # 3. 峰偏离度图
            ax3 = plt.subplot(223)
            peak_deviations = np.zeros((self.grid_points, self.grid_points))
            center_x = self.grid[self.center_j]
            center_y = self.grid[self.center_i]
            
            # 计算每个点的峰偏离度
            for i in range(self.grid_points):
                row = beam_full[i, :]
                row_peak = np.argmax(row)
                peak_deviations[i, :] = np.abs(self.grid - self.grid[row_peak])
            
            # 绘制
            im3 = ax3.imshow(peak_deviations, cmap="viridis", extent=[-15, 15, -15, 15])
            plt.colorbar(im3, ax=ax3, label="峰位置偏离 (mm)")
            ax3.scatter([center_x], [center_y], s=100, marker='*', c='r', label='中心点')
            ax3.set_title("行峰位置偏离中心图")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            ax3.legend()
            
            # 4. 违规历史曲线
            if len(self.history["iteration"]) > 1:
                ax4 = plt.subplot(224)
                iterations = self.history["iteration"]
                abs_errors = self.history["abs_error"]
                peak_violations = self.history["peak_violations"]
                
                # 左侧Y轴 - 绝对误差
                ax4.plot(iterations, abs_errors, "k-", label="绝对误差")
                ax4.set_xlabel("迭代次数")
                ax4.set_ylabel("绝对误差", color='k')
                ax4.tick_params(axis='y', labelcolor='k')
                ax4.grid(True)
                
                # 右侧Y轴 - 峰违规
                ax4b = ax4.twinx()
                ax4b.plot(iterations, peak_violations, "r--", label="峰违规次数")
                ax4b.set_ylabel("峰违规次数", color='r')
                ax4b.tick_params(axis='y', labelcolor='r')
                ax4b.set_ylim(-1, max(peak_violations)*1.2)
                
                # 标记阶段转换
                for i, end_idx in enumerate(self.stage_ends):
                    ax4.axvline(end_idx, color=f'C{i}', linestyle='--', alpha=0.7)
                
                # 图例
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4b.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                ax4.set_title("峰违规历史")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig("unimodal_verification.png", bbox_inches='tight')
            self.log("束流单峰性验证图已保存")
            plt.close(fig)
        except Exception as e:
            self.log(f"单峰性验证图创建失败: {str(e)}")

    def plot_radial_distribution(self, beam_full):
        """绘制径向分布验证图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 创建距离数组
            r = self.r_center.flatten()
            values = beam_full.flatten()
            
            # 添加中心点
            ax.scatter(0, beam_full[self.center_i, self.center_j], 
                      color='red', s=100, marker='*', 
                      label='中心点值')
            
            # 绘制所有点随距离的分布
            ax.scatter(r, values, alpha=0.5, s=20)
            
            # 计算平均值
            bins = np.linspace(0, 15, 20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            radial_avg = []
            for i in range(len(bins)-1):
                mask = (r >= bins[i]) & (r < bins[i+1])
                if np.any(mask):
                    radial_avg.append(np.mean(values[mask]))
                else:
                    radial_avg.append(0)
            
            # 绘制平均趋势线
            ax.plot(bin_centers, radial_avg, 'r-', linewidth=3, 
                   label='径向平均值')
            
            # 添加指数衰减参考线
            decay_ref = radial_avg[0] * np.exp(-bin_centers/5)
            ax.plot(bin_centers, decay_ref, 'b--', linewidth=2.5,
                   label='指数衰减参考 (λ=5mm)')
            
            ax.set_title("束流强度径向分布")
            ax.set_xlabel("到中心点的距离 (mm)")
            ax.set_ylabel("刻蚀速率 (nm/s)")
            ax.grid(True)
            ax.legend()
            
            fig.savefig("beam_radial_distribution.png", dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.log("束流径向分布验证图已保存")
        except Exception as e:
            self.log(f"径向分布图保存失败: {str(e)}")
    
    def visualize_results(self):
        """可视化优化结果"""
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle("离子束刻蚀效率优化结果 (单峰约束版)", fontsize=16)
            
            # 1. 原始束流分布
            ax1 = plt.subplot(231)
            im1 = ax1.imshow(self.initial_beam, cmap="viridis", 
                            extent=[-15, 15, -15, 15])
            
            # 标记位置点集阶段
            for i, mask in enumerate(self.stage_masks):
                indices = np.where(mask)
                if len(indices[0]) > 0:
                    x = self.grid[indices[0]]
                    y = self.grid[indices[1]]
                    ax1.scatter(x, y, s=40, marker=f'.', 
                               alpha=0.7, label=f'集合{i+1}')
            
            ax1.set_title("初始束流分布")
            plt.colorbar(im1, ax=ax1, label="刻蚀速率 (nm/s)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 2. 优化后束流分布
            optimized_beam_full = self.optimized_beam * self.max_val
            ax2 = plt.subplot(232)
            im2 = ax2.imshow(optimized_beam_full, cmap="viridis", 
                            extent=[-15, 15, -15, 15])
            # 添加等高线
            cont = ax2.contour(self.grid, self.grid, optimized_beam_full,
                              levels=10, colors='w', linewidths=0.7, alpha=0.7)
            ax2.clabel(cont, fmt='%1.1f', colors='k', fontsize=8)
            ax2.set_title("优化后束流分布")
            plt.colorbar(im2, ax=ax2, label="刻蚀速率 (nm/s)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            
            # 3. 束沿Y移动时的X方向轮廓
            ax3 = plt.subplot(233)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")
            
            ax3.scatter(
                self.beam_traced_y_axis[:, 0], 
                self.beam_traced_y_axis[:, 1]/np.max(self.beam_traced_y_axis[:, 1]), 
                c="g", s=30, alpha=0.6, label="实验数据 (束沿Y)"
            )
            ax3.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            ax3.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            ax3.set_title("束沿Y移动时的X轴截面")
            ax3.set_xlabel("X位置 (mm)")
            ax3.set_ylabel("归一化刻蚀深度")
            ax3.grid(True)
            ax3.legend()
            
            # 4. 束沿X移动时的Y方向轮廓
            ax4 = plt.subplot(234)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")
            
            ax4.scatter(
                self.beam_traced_x_axis[:, 0], 
                self.beam_traced_x_axis[:, 1]/np.max(self.beam_traced_x_axis[:, 1]), 
                c="g", s=30, alpha=0.6, label="实验数据 (束沿X)"
            )
            ax4.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            ax4.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            ax4.set_title("束沿X移动时的Y轴截面")
            ax4.set_xlabel("Y位置 (mm)")
            ax4.set_ylabel("归一化刻蚀深度")
            ax4.grid(True)
            ax4.legend()
            
            # 5. 误差收敛曲线
            if len(self.history["iteration"]) > 1:
                ax5 = plt.subplot(235)
                iterations = self.history["iteration"]
                abs_errors = self.history["abs_error"]
                rel_err_x = self.history["rel_err_x"]
                rel_err_y = self.history["rel_err_y"]
                
                # 左侧Y轴 - 绝对误差
                ax5.plot(iterations, abs_errors, "ko-", label="绝对误差")
                ax5.set_xlabel("迭代次数")
                ax5.set_ylabel("绝对误差", color='k')
                ax5.tick_params(axis='y', labelcolor='k')
                ax5.grid(True)
                
                # 标记阶段转换
                for i, end_idx in enumerate(self.stage_ends):
                    ax5.axvline(end_idx, color=f'C{i}', linestyle='--', alpha=0.7, label=f"阶段{i+1}结束")
                
                # 右侧Y轴 - 相对误差
                ax5b = ax5.twinx()
                ax5b.plot(iterations, rel_err_x, "r--", label="束X动时Y向误差")
                ax5b.plot(iterations, rel_err_y, "g--", label="束Y动时X向误差")
                ax5b.set_ylabel("相对误差 (%)", color='b')
                ax5b.tick_params(axis='y', labelcolor='b')
                
                # 创建组合图例
                lines, labels = ax5.get_legend_handles_labels()
                lines2, labels2 = ax5b.get_legend_handles_labels()
                ax5.legend(lines + lines2, labels + labels2, loc='upper right')
                
                ax5.set_title("误差收敛曲线")
            
            # 6. 峰违规验证图
            if os.path.exists("unimodal_verification.png"):
                from matplotlib import image
                ax6 = plt.subplot(236)
                img = image.imread("unimodal_verification.png")
                ax6.imshow(img)
                ax6.axis('off')
                ax6.set_title("单峰性验证")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig("beam_optimization_results.png", bbox_inches='tight')
            self.log("优化结果可视化已保存")
            plt.close(fig)
        except Exception as e:
            self.log(f"可视化失败: {str(e)}")
    
    def finalize(self):
        """结束优化并关闭日志"""
        self.log("\n优化完成!")
        self.log("结果已保存至:")
        self.log(f"  - optimized_beam_distribution.csv (优化后束流分布)")
        self.log(f"  - beam_optimization_results.png (可视化结果)")
        self.log(f"  - beam_radial_distribution.png (径向分布验证)")
        self.log(f"  - unimodal_verification.png (单峰性验证)")
        self.log_file.close()

# ================== 主程序 ==================
def main():
    # 检查文件存在性
    files = {
        "beam_traced_x_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",  # 束沿X移动时测量的Y截面
        "beam_traced_y_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",  # 束沿Y移动时测量的X截面
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 80)
    print("离子束刻蚀效率优化 (行/列单峰约束版)".center(80))
    print(f"输入文件:")
    print(f" - 束沿X移动时的Y截面: {files['beam_traced_x_axis']}")
    print(f" - 束沿Y移动时的X截面: {files['beam_traced_y_axis']}")
    print(f" - 初始束流分布: {files['initial_beam']}")
    print("=" * 80)
    
    # 检查文件存在
    missing = []
    for name, path in files.items():
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")
    
    if missing:
        print("错误: 以下文件不存在:")
        for item in missing:
            print(f"  - {item}")
        print("请检查文件路径后重试!")
        sys.exit(1)
    
    # 创建优化器
    try:
        optimizer = BeamEfficiencyOptimizer(
            beam_traced_x_axis=files["beam_traced_x_axis"],
            beam_traced_y_axis=files["beam_traced_y_axis"],
            initial_guess_path=files["initial_beam"]
        )
        
        # 执行11阶段优化
        result, err_x, err_y = optimizer.run_optimization()
        
        # 可视化
        optimizer.visualize_results()
        
        # 最终报告
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("优化完成! 最终误差:".center(80))
        print(f" - 束沿X移动时Y向误差: {err_x:.1f}%")
        print(f" - 束沿Y移动时X向误差: {err_y:.1f}%")
        print(f"结果文件:")
        print(f"  - optimized_beam_distribution.csv (优化后束流分布)")
        print(f"  - beam_optimization_results.png (可视化结果)")
        print(f"  - beam_radial_distribution.png (径向分布验证)")
        print(f"  - unimodal_verification.png (单峰性验证)")
        print(f"  - beam_optimization_log.txt (详细日志)")
        print("=" * 80)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        if 'optimizer' in locals() and hasattr(optimizer, 'log_file'):
            optimizer.log_file.close()

if __name__ == "__main__":
    main()
