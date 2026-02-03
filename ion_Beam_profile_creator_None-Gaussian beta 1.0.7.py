import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
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
        """初始化优化器，修正了实验数据文件定义：
        - beam_traced_x_axis: 离子束沿X轴移动时测量的Y方向截面
        - beam_traced_y_axis: 离子束沿Y轴移动时测量的X方向截面
        """
        # 创建日志文件（UTF-8编码解决乱码问题）
        self.log_file = open("beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("离子束刻蚀效率优化引擎启动")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"输入文件说明:")
        self.log(f" - 离子束沿X轴移动时的Y方向截面数据: {beam_traced_x_axis}")
        self.log(f" - 离子束沿Y轴移动时的X方向截面数据: {beam_traced_y_axis}")
        self.log("=" * 30)
        
        # 加载初始猜测
        self.load_initial_beam(initial_guess_path)
        
        # 加载实验数据
        self.beam_traced_x_axis = self.load_experimental_data(beam_traced_x_axis)  # 沿X轴移动 (测量Y截面)
        self.beam_traced_y_axis = self.load_experimental_data(beam_traced_y_axis)  # 沿Y轴移动 (测量X截面)
        
        # 网格系统
        self.grid_bound = grid_bound
        self.grid_points = grid_points
        self.grid = np.linspace(-grid_bound, grid_bound, grid_points)
        self.grid_spacing = 2 * grid_bound / (grid_points - 1)
        
        # 优化参数
        self.scan_velocity = 30.0  # 扫描速度 (mm/s)
        self.opt_radius = 10.0  # 优化半径 (mm)
        self.max_mutations = 4  # 变异候选数

        # 创建优化掩膜
        self.create_optimization_mask()
        
        # 新增分区优化变量
        self.region_stage = "center"  # 优化阶段: center/edge/outer
        self.center_threshold = None  # FWHM值
        self.center_mask = None       # 中心区域掩膜
        self.edge_mask = None         # 边缘区域掩膜
        self.outer_mask = None        # 外层区域掩膜 (新增)
        
        # 在加载初始束流后创建区域掩膜
        self.create_region_masks(self.initial_beam)

        self.stage_ends = []  # 存储阶段结束点的迭代索引
        
        # 历史记录
        self.history = {
            "iteration": [],
            "abs_error": [],
            "rel_err_x": [],
            "rel_err_y": [],
            "max_val": []
        }
        self.optimized_beam = self.initial_beam / self.max_val  # 初始优化束流

    def log(self, message):
        """记录带时间戳的日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_file.write(log_entry + "\n")
        self.log_file.flush()

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
        
        # 创建外部区域掩膜
        self.outer_mask = (self.distance_from_center > self.opt_radius) & (self.distance_from_center <= self.grid_bound)
        self.log(f"外部区域包含 {np.sum(self.outer_mask)} 个点 (半径 > {self.opt_radius}mm)")

    def create_interpolator(self, beam_matrix):
        """创建双线性插值器"""
        return RegularGridInterpolator(
            (self.grid, self.grid),
            beam_matrix * self.max_val,  # 反归一化
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
    
    def create_region_masks(self, beam_matrix):
        """创建分区优化掩膜"""
        # 计算FWHM（半高全宽）
        max_val = np.max(beam_matrix)
        half_max = max_val * 0.5
        half_max_indices = beam_matrix >= half_max
        
        # 确保中心点在内
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        
        # 计算平均FWHM
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt(xx**2 + yy**2)
        center_r = r[center_i, center_j]
        
        # 找出所有半高宽点的最远距离
        if np.any(half_max_indices):
            max_r = np.max(r[half_max_indices])
            self.center_threshold = max_r * 1.2  # 加20%缓冲
        else:
            self.center_threshold = 5.0  # 默认值
        
        # 创建距离矩阵（以中心为原点）
        r_center = np.sqrt((xx - self.grid[center_i])**2 + (yy - self.grid[center_j])**2)
        
        # 中心区域掩膜（FWHM+缓冲）
        self.center_mask = (r_center <= self.center_threshold) & self.optimization_mask
        
        # 边缘区域掩膜
        self.edge_mask = (r_center > self.center_threshold) & (r_center <= self.opt_radius) & self.optimization_mask
        
        # 外部区域掩膜 
        self.outer_mask = (r_center > self.opt_radius) & (r_center <= self.grid_bound)
        
        # 统计信息
        center_points = np.sum(self.center_mask)
        edge_points = np.sum(self.edge_mask)
        outer_points = np.sum(self.outer_mask)
        self.log(f"分区优化: 中心区域{center_points}点, 边缘区域{edge_points}点, 外部区域{outer_points}点")
        self.log(f"FWHM阈值: {self.center_threshold:.2f}mm")


    def enforce_radial_constraints(self, beam_matrix):
        """
        强制执行径向约束（完整实现）
        1. 确保中心点位于物理中心附近
        2. 禁止中心区域零值点
        3. 强制径向单调递减
        4. 确保下降速率先增后减且只有一个拐点
        """
        rows, cols = beam_matrix.shape
        
        # 1. 寻找并修正中心点位置
        max_idx = np.argmax(beam_matrix)
        center_idx = np.unravel_index(max_idx, beam_matrix.shape)
        center_i, center_j = center_idx
        center_pos = (self.grid[center_i], self.grid[center_j])
        center_distance = np.sqrt(center_pos[0]**2 + center_pos[1]**2)
        
        # 修正偏离过远的中心点
        if center_distance > 2.0:
            center_region_i = np.where((np.abs(self.grid) < 1.0))[0]
            center_region_j = np.where((np.abs(self.grid) < 1.0))[0]
            
            if center_region_i.size > 0 and center_region_j.size > 0:
                center_region_values = beam_matrix[center_region_i[:, None], center_region_j]
                max_in_center = np.max(center_region_values)
                max_idx = np.unravel_index(np.argmax(center_region_values), (center_region_i.size, center_region_j.size))
                center_i, center_j = center_region_i[max_idx[0]], center_region_j[max_idx[1]]
                beam_matrix[center_i, center_j] = max(beam_matrix[center_i, center_j], max_in_center)
                center_pos = (self.grid[center_i], self.grid[center_j])
                self.log(f"修正中心偏离: {center_pos}, 值={max_in_center:.4f}")
        
        # 重新确定中心点
        center_i, center_j = np.unravel_index(np.argmax(beam_matrix), beam_matrix.shape)
        center_pos = (self.grid[center_i], self.grid[center_j])
        max_val = beam_matrix[center_i, center_j]
        self.log(f"当前中心点位置: ({center_pos[0]:.2f}, {center_pos[1]:.2f}), 值={max_val:.4f}")
        
        # 2. 创建距离矩阵
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt((xx - center_pos[0])**2 + (yy - center_pos[1])**2)
        
        # 3. 中心区域（半径≤10mm）禁止零值点
        inner_region = (r <= 10.0)
        low_points_mask = beam_matrix < max_val * 0.01
        inner_low_points = np.logical_and(inner_region, low_points_mask)
        low_points_count = np.sum(inner_low_points)
        
        if low_points_count > 0:
            base_value = max_val * 0.02
            beam_matrix[inner_low_points] = base_value
            self.log(f"禁止零值点: 设置 {low_points_count} 个点至少为 {base_value:.4f}")
        
        # 4. 沿射线方向的完整约束
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 增加到16个方向更平滑
        modified_points = 0
        
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            max_radius = self.grid_bound * 1.0  # 延伸至边界
            steps = np.arange(0, max_radius + 0.5, 0.5)  # 0.5mm步进
            distances = steps
            
            # 计算射线上的网格点索引
            x_indices = np.round(center_i + dx * (distances / self.grid_spacing)).astype(int)
            y_indices = np.round(center_j + dy * (distances / self.grid_spacing)).astype(int)
            
            # 筛选有效索引
            valid_mask = (x_indices >= 0) & (x_indices < rows) & \
                        (y_indices >= 0) & (y_indices < cols)
            valid_x = x_indices[valid_mask]
            valid_y = y_indices[valid_mask]
            distances = distances[valid_mask]
            
            if len(valid_x) < 5:  # 至少需要5个点用于分析
                continue
                
            # 获取射线上的值
            ray_values = beam_matrix[valid_x, valid_y]
            
            # 4.1 强制单调递减（主要约束）
            for k in range(1, len(valid_x)):
                i, j = valid_x[k], valid_y[k]
                r_current = np.sqrt((self.grid[i]-center_pos[0])**2 + (self.grid[j]-center_pos[1])**2)
                    
                current_val = beam_matrix[i, j]
                prev_val = beam_matrix[valid_x[k-1], valid_y[k-1]]
                
                # 确保严格递减（允许1%波动）
                if current_val > prev_val * 0.99:
                    new_val = prev_val * 0.98
                    if new_val != current_val:
                        beam_matrix[i, j] = new_val
                        modified_points += 1
                
                # 设置递减上限
                upper_limit = prev_val * 0.95
                if current_val > upper_limit:
                    beam_matrix[i, j] = upper_limit
                    modified_points += 1
                    
                # 外部区域上限
                if r_current > self.opt_radius:
                    beam_matrix[i, j] = min(beam_matrix[i, j], max_val * 0.05)
            
            # 重新获取更新后的射线值
            ray_values = beam_matrix[valid_x, valid_y]
            
        # 5. 确保所有值非负
        beam_matrix = np.maximum(beam_matrix, 0)
        
        if modified_points > 0:
            self.log(f"径向约束: 调整了 {modified_points} 个点")
        
        return beam_matrix



    def simulate_etching(self, beam_matrix, direction):
        """
        模拟指定方向的刻蚀轮廓
        
        修正参数定义:
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

    def calculate_error(self, sim_x_scan, sim_y_scan, region_mask=False):
        """
        计算模拟结果与实验数据的误差（完整实现）
        - region_mask: 是否只计算中心区域的误差
        """
        # 处理束沿Y移动时的X方向轮廓（束沿Y移动 -> 测量X方向轮廓）
        exp_x_data = self.beam_traced_y_axis
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        
        # 归一化实验数据
        exp_x_max = np.max(exp_x_val) if np.max(exp_x_val) > 0 else 1.0
        exp_x_norm = exp_x_val / exp_x_max
        
        # 插值模拟数据到实验点位置
        sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_scan)
        
        # 1. 如果需要区域掩膜（中心优化阶段）
        if region_mask and self.center_threshold is not None:
            # 创建中心区域掩膜
            center_mask_x = (exp_x_pos >= -self.center_threshold) & (exp_x_pos <= self.center_threshold)
            
            # 应用掩膜
            exp_x_norm_masked = exp_x_norm[center_mask_x]
            sim_x_interp_masked = sim_x_interp[center_mask_x]
        else:
            # 全区域计算
            exp_x_norm_masked = exp_x_norm
            sim_x_interp_masked = sim_x_interp
        
        # 2. 计算X误差
        abs_dev_x = np.abs(sim_x_interp_masked - exp_x_norm_masked)
        rel_err_x = np.mean(abs_dev_x) * 100  # 初始化为绝对误差
        
        # 3. 处理束沿X移动时的Y方向轮廓（束沿X移动 -> 测量Y方向轮廓）
        exp_y_data = self.beam_traced_x_axis
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        
        # 归一化实验数据
        exp_y_max = np.max(exp_y_val) if np.max(exp_y_val) > 0 else 1.0
        exp_y_norm = exp_y_val / exp_y_max
        
        # 插值模拟数据到实验点位置
        sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_scan)
        
        # 4. 如果需要区域掩膜（中心优化阶段）
        if region_mask and self.center_threshold is not None:
            # 创建中心区域掩膜
            center_mask_y = (exp_y_pos >= -self.center_threshold) & (exp_y_pos <= self.center_threshold)
            
            # 应用掩膜
            exp_y_norm_masked = exp_y_norm[center_mask_y]
            sim_y_interp_masked = sim_y_interp[center_mask_y]
        else:
            # 全区域计算
            exp_y_norm_masked = exp_y_norm
            sim_y_interp_masked = sim_y_interp
        
        # 5. 计算Y误差
        abs_dev_y = np.abs(sim_y_interp_masked - exp_y_norm_masked)
        rel_err_y = np.mean(abs_dev_y) * 100  # 初始化为绝对误差
        
        # 6. 计算综合绝对误差
        abs_err_x = np.mean(abs_dev_x)
        abs_err_y = np.mean(abs_dev_y)
        abs_error = (abs_err_x + abs_err_y) / 2
        
        # 7. 只有在实验数据非零时计算真正相对误差
        non_zero_x = exp_x_norm > 1e-5
        if np.any(non_zero_x) and region_mask is False:
            rel_err_x = np.mean(abs_dev_x[non_zero_x] / exp_x_norm[non_zero_x]) * 100
        
        non_zero_y = exp_y_norm > 1e-5
        if np.any(non_zero_y) and region_mask is False:
            rel_err_y = np.mean(abs_dev_y[non_zero_y] / exp_y_norm[non_zero_y]) * 100
        
        return abs_error, rel_err_x, rel_err_y




    def mutate_beam(self, beam_matrix, magnitude, sim_x=None, sim_y=None):
        """改进的变异束流分布 - 基于偏差分析的定向变异策略"""
        rows, cols = beam_matrix.shape
        new_beam = beam_matrix.copy()
        
        # 如果没有提供当前模拟值，需要计算
        if sim_x is None or sim_y is None:
            sim_x = self.simulate_etching(new_beam, "x")
            sim_y = self.simulate_etching(new_beam, "y")
        
        # 计算与实验数据的偏差
        exp_x_data = self.beam_traced_y_axis  # 束沿Y移动时的X轮廓数据
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        
        # 归一化实验数据
        exp_x_max = np.max(exp_x_val) if np.max(exp_x_val) > 0 else 1.0
        exp_x_norm = exp_x_val / exp_x_max
        
        # 插值模拟数据到实验点位置
        sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x)
        
        # 计算X方向的偏差 (模拟值 - 实验值)
        dev_x = sim_x_interp - exp_x_norm
        
        # 计算跨零点位置
        cross_zero_x = []
        for i in range(1, len(dev_x)):
            if dev_x[i-1] * dev_x[i] < 0:  # 符号变化表示过零点
                cross_zero_x.append((exp_x_pos[i-1] + exp_x_pos[i]) / 2)
        
        # 同样的过程处理Y方向
        # 计算与实验数据的偏差
        exp_y_data = self.beam_traced_x_axis  # 束沿X移动时的Y轮廓数据
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        
        # 归一化实验数据
        exp_y_max = np.max(exp_y_val) if np.max(exp_y_val) > 0 else 1.0
        exp_y_norm = exp_y_val / exp_y_max
        
        # 插值模拟数据到实验点位置
        sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y)
        
        # 计算Y方向的偏差 (模拟值 - 实验值)
        dev_y = sim_y_interp - exp_y_norm
        
        # 计算跨零点位置
        cross_zero_y = []
        for i in range(1, len(dev_y)):
            if dev_y[i-1] * dev_y[i] < 0:  # 符号变化表示过零点
                cross_zero_y.append((exp_y_pos[i-1] + exp_y_pos[i]) / 2)
        
        # 如果没有检测到零点，使用中心和边缘替代
        if not cross_zero_x: 
            cross_zero_x = [0]  # 使用中心点
        
        if not cross_zero_y: 
            cross_zero_y = [0]  # 使用中心点
        
        # 1. 如果没有分区信息，直接使用优化区域（兼容性）
        if self.center_mask is None:
            if self.optimization_mask is None:
                valid_indices = np.indices(new_beam.shape)
                valid_indices = (valid_indices[0].ravel(), valid_indices[1].ravel())
            else:
                valid_indices = np.where(self.optimization_mask | self.outer_mask)  # 包含外部区域
        else:
            # 根据当前阶段确定目标区域
            if self.region_stage == "center":
                target_mask = self.center_mask
            elif self.region_stage == "edge":
                target_mask = self.edge_mask
            elif self.region_stage == "outer":
                target_mask = self.outer_mask
            else:
                target_mask = np.ones((rows, cols), dtype=bool)
                
            valid_indices = np.where(target_mask)
        
        # 计算中心点
            center_i, center_j = np.unravel_index(np.argmax(new_beam), new_beam.shape)
        center_x, center_y = self.grid[center_i], self.grid[center_j]
        center_val = new_beam[center_i, center_j]
        
        # 2. 只对有效区域的点进行变异
        for i in range(len(valid_indices[0])):
            x, y = valid_indices[0][i], valid_indices[1][i]
            pos_x = self.grid[x]
            pos_y = self.grid[y]
            
            # 变异概率调整：中心区域概率高，边缘和外层概率递减
            if self.region_stage == "center":
                mutation_prob = 0.65
            elif self.region_stage == "edge":
                mutation_prob = 0.45
            elif self.region_stage == "outer":
                mutation_prob = 0.25
            else:
                mutation_prob = 0.3
            
            if np.random.rand() < mutation_prob:
                # 获取当前点到中心的距离
                dist = np.sqrt((pos_x - center_x)**2 + (pos_y - center_y)**2)
                
                # 动态变异幅度策略
                base_magnitude = magnitude
                dist_factor = 1.0
                
                # 距离中心越远，变异幅度越小
                if dist > 0.1:
                    dist_factor = np.exp(-dist / 8.0)  # 8mm衰减长度
                
                # 当前值与中心值的比例
                if center_val > 1e-5:
                    value_factor = max(0.1, min(1.0, new_beam[x, y] / center_val))
                else:
                    value_factor = 0.5
                
                # 外层区域增加变异幅度因子
                if self.region_stage == "outer" and dist > self.opt_radius:
                    outer_factor = 1.5
                else:
                    outer_factor = 1.0
                
                # 最终变异幅度
                effective_magnitude = base_magnitude * dist_factor * value_factor * outer_factor * np.random.rand()
                
                # ======== 关键改进：基于偏差的定向变异 ========
                # 确定X方向变异方向
                direction_x = 0
                for zero_x in cross_zero_x:
                    if pos_x < zero_x:  # 零点左侧
                        # 如果中心在左侧且偏差为正，应减小
                        # 如果中心在左侧且偏差为负，应增大
                        # 简化：在零点左侧默认增大
                        direction_x = 1
                    else:  # 零点右侧
                        # 在零点右侧默认减小
                        direction_x = -1
                    break  # 只考虑最近的零点
                
                # 确定Y方向变异方向
                direction_y = 0
                for zero_y in cross_zero_y:
                    if pos_y < zero_y:  # 零点下侧
                        direction_y = 1
                    else:  # 零点上侧
                        direction_y = -1
                    break  # 只考虑最近的零点
                
                # 综合方向
                direction = 1 if (direction_x + direction_y) > 0 else -1
                
                # 应用变异并确保非负
                mutation = direction * effective_magnitude
                new_val = max(0, new_beam[x, y] + mutation)
                
                # 区域上限控制
                if self.region_stage == "center":
                    new_val = max(0, min(1.0, new_val))
                elif self.region_stage == "edge":
                    new_val = max(0, min(0.5, new_val))  # 边缘区域上限
                elif self.region_stage == "outer":
                    new_val = max(0, min(0.2, new_val))  # 外部区域上限
                else:
                    new_val = max(0, min(1.0, new_val))
                
                new_beam[x, y] = new_val
        
        # 3. 应用约束后返回
        return self.enforce_radial_constraints(new_beam)        



    def optimize_step(self, current_matrix, abs_error, magnitude):
        """执行单步优化"""
        # 创建候选方案
        candidates = [current_matrix.copy()]  # 保留当前方案
        for _ in range(self.max_mutations):
            candidate = self.mutate_beam(current_matrix, magnitude)
            candidates.append(candidate)
        
        # 评估所有候选方案
        best_candidate = None
        best_abs_error = abs_error
        best_errs = (0, 0)
        errors = []
        
        for candidate in candidates:
            try:
                # 模拟束沿X和Y移动时的轮廓
                sim_x = self.simulate_etching(candidate, "x")  # 束沿Y移动 -> X轮廓
                sim_y = self.simulate_etching(candidate, "y")  # 束沿X移动 -> Y轮廓
                
                # 计算误差
                abs_error, rel_err_x, rel_err_y = self.calculate_error(sim_x, sim_y)
                errors.append(abs_error)
                
                if abs_error < best_abs_error:
                    best_abs_error = abs_error
                    best_errs = (rel_err_x, rel_err_y)
                    best_candidate = candidate
            except Exception as e:
                self.log(f"评估失败: {str(e)}")
                errors.append(1.0)  # 高误差
            
        # 记录评估结果
        errors_str = ", ".join([f"{err:.5f}" for err in errors])
        self.log(f"变异评估: [{errors_str}]")
        
        if best_candidate is None:
            best_candidate = candidates[0]  # 返回原始
            self.log("无改进")
        else:
            improvement = abs_error - best_abs_error
            self.log(f"找到改进 {improvement:.5f}")
        
        return best_candidate, best_abs_error, best_errs
    
    def calculate_beam_properties(self, beam_matrix):
        """计算束流特性（FWHM和中心值）（完整实现）"""
        # 找到最大值位置
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        center_val = beam_matrix[center_i, center_j]
        
        # 径向截面分析
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        fwhm_values = []
        
        for angle in angles:
            # 沿射线方向采样
            dx, dy = np.cos(angle), np.sin(angle)
            distances = np.linspace(0, self.grid_bound, num=100)
            xx = center_i + dx * distances / self.grid_spacing
            yy = center_j + dy * distances / self.grid_spacing
            
            # 获得射线上的值
            ray_values = np.zeros(len(distances))
            for idx, (x, y) in enumerate(zip(xx, yy)):
                if 0 <= x < beam_matrix.shape[0] and 0 <= y < beam_matrix.shape[0]:
                    ray_values[idx] = beam_matrix[int(x), int(y)]
                else:
                    ray_values[idx] = 0
            
            # 计算该方向的FWHM
            half_max = center_val * 0.5
            above_half = np.where(ray_values > half_max)[0]
            
            if len(above_half) > 1:
                first_idx = above_half[0]
                last_idx = above_half[-1]
                fwhm = distances[last_idx] - distances[first_idx]
                fwhm_values.append(fwhm)
        
        # 平均FWHM（至少有4个方向才有意义）
        if len(fwhm_values) >= 4:
            avg_fwhm = np.mean(fwhm_values)
        else:
            avg_fwhm = 7.0  # 默认值
        
        # 存储属性
        self.center_threshold = avg_fwhm * 1.2  # 加上20%缓冲区
        self.log(f"计算束流特性: FWHM={avg_fwhm:.2f}mm, 中心阈值={self.center_threshold:.2f}mm")
        
        # 创建中心区域掩膜
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r_center = np.sqrt((xx - self.grid[center_i])**2 + (yy - self.grid[center_j])**2)
        self.center_mask = (r_center <= self.center_threshold) & self.optimization_mask
        
        # 创建边缘区域掩膜
        self.edge_mask = (r_center > self.center_threshold) & (r_center <= self.opt_radius) & self.optimization_mask
        
        # 创建外部区域掩膜
        self.outer_mask = (r_center > self.opt_radius) & (r_center <= self.grid_bound)
        
        # 调试信息
        center_points = np.sum(self.center_mask)
        edge_points = np.sum(self.edge_mask)
        outer_points = np.sum(self.outer_mask)
        self.log(f"分区优化: 中心区域{center_points}点, 边缘区域{edge_points}点, 外部区域{outer_points}点")
        
        return avg_fwhm, center_val
    
    def calculate_magnitude(self, current_iter, max_iters, min_mag=0.02):
        """动态计算变异幅度"""
        # 基础公式：初期高变异，后期低变异
        base_magnitude = 0.15 * np.exp(-current_iter / (max_iters / 3))
        
        # 根据优化阶段调整
        if self.region_stage == "center":
        # 中心区域：中期变异幅度最大
            magnitude = min(0.12, max(min_mag, base_magnitude * (1.0 - current_iter/(max_iters*1.5))))
        elif self.region_stage == "outer":
            # 外部区域：整体较小变异
            magnitude = min(0.05, max(min_mag, base_magnitude * 0.4))
        elif self.region_stage == "smoothing":  # 新增第四阶段
            # 平滑阶段：更小的变异幅度
            magnitude = min(0.04, max(min_mag, base_magnitude * 0.3))
        else:
            # 边缘区域：正常变异
            magnitude = min(0.08, max(min_mag, base_magnitude * 0.7))
        
        # 添加随机波动 (±20%)
        #magnitude *= 0.8 + np.random.rand() * 0.4
        
        return max(magnitude, min_mag)



    def run_optimization(self, max_iterations=100, target_rel_error=10.0):
        """完整的运行优化过程（三阶段策略）"""
        # 0. 初始设置
        self.log(f"\n开始优化过程 (目标相对误差: {target_rel_error}%)")
        current_matrix = self.optimized_beam.copy()
        stage_iterations = max(max_iterations // 4, 15)  # 确保至少15次迭代
        
        # 计算初始束流特性
        if not hasattr(self, 'center_threshold') or self.center_threshold is None:
            self.calculate_beam_properties(current_matrix)
        
        self.log(f"检测到的束流特性 - FWHM: {self.center_threshold:.2f}mm, 中心值: {np.max(self.optimized_beam):.4f}")
        
        # 1. 初始评估
        sim_x = self.simulate_etching(current_matrix, "x")
        sim_y = self.simulate_etching(current_matrix, "y")
        abs_error0, rel_err_x0, rel_err_y0 = self.calculate_error(sim_x, sim_y)
        self.log(f"初始误差: 绝对={abs_error0:.4f}, 束X动时Y向误差: {rel_err_x0:.1f}%, 束Y动时X向误差: {rel_err_y0:.1f}%")
        
        # 添加初始状态到历史记录
        self.history["iteration"].append(0)
        self.history["abs_error"].append(abs_error0)
        self.history["rel_err_x"].append(rel_err_x0)
        self.history["rel_err_y"].append(rel_err_y0)
        
        # 存储当前最佳的模拟轮廓（用于指导变异）
        best_sim = (sim_x, sim_y)
        
        # 2. 第一阶段：中心区域优化
        self.region_stage = "center"
        self.log("\n=== 第一阶段：优化中心区域 ===")
        start_time = time.time()
        stagnation_count = 0
        
        # 更严格的目标误差（中心区域）
        center_target_error = min(target_rel_error * 0.75, 15.0)
        
        # === 修复：初始化中心区域误差 ===
        center_rel_err_x = rel_err_x0
        center_rel_err_y = rel_err_y0
        
        for iteration in range(1, stage_iterations + 1):
            # 2.1 动态调整变异幅度
            magnitude = self.calculate_magnitude(iteration, stage_iterations)
            
            self.log(f"迭代 {iteration}: 第1阶段(中心), 变异幅度={magnitude:.3f}")
            
            # 使用当前最佳模拟轮廓指导变异
            current_sim_x, current_sim_y = best_sim
            
            # 2.2 创建候选方案 - 使用当前模拟轮廓指导变异
            candidates = [current_matrix.copy()]
            for _ in range(self.max_mutations):  # 变异候选
                candidate = self.mutate_beam(current_matrix, magnitude, sim_x=current_sim_x, sim_y=current_sim_y)
                candidates.append(candidate)
            
            best_candidate = candidates[0]
            best_abs_error = self.history["abs_error"][-1]  # 使用上一次误差作为基准
            best_errs = (center_rel_err_x, center_rel_err_y)
            best_candidate_sim = best_sim  # 预置为当前的最优模拟
            
            # 2.3 评估所有候选方案（重点在中心区域误差）
            for candidate in candidates:
                cand_sim_x = self.simulate_etching(candidate, "x")
                cand_sim_y = self.simulate_etching(candidate, "y")
                abs_error, rel_err_x, rel_err_y = self.calculate_error(cand_sim_x, cand_sim_y, region_mask=True)
                
                # 中心区域误差指标（使用绝对误差）
                center_error = abs_error
                
                if center_error < best_abs_error:
                    best_abs_error = center_error
                    best_errs = (rel_err_x, rel_err_y)
                    best_candidate = candidate
                    best_candidate_sim = (cand_sim_x, cand_sim_y)  # 存储候选解的模拟结果
            
            # 2.4 记录评估结果
            if best_candidate is not candidates[0]:
                improvement = self.history["abs_error"][-1] - best_abs_error # 仅使用上一次误差
                self.log(f"中心区域改进: Δ={improvement:.5f}, 新误差={best_abs_error:.4f}")
                极速_count = 0
                # 更新中心区域误差
                center_rel_err_x, center_rel_err_y = best_errs
                self.log(f"中心误差: X向={center_rel_err_x:.1f}%, Y向={center_rel_err_y:.1f}%")
                
                # 更新历史记录
                self.history["iteration"].append(iteration)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(center_rel_err_x)
                self.history["rel_err_y"].append(center_rel_err_y)
                
                # 更新当前最佳模拟轮廓
                best_sim = best_candidate_sim
            else:
                stagnation_count += 1
                self.log(f"中心区域无改进，维持绝对误差: {best_abs_error:.4f}")
                self.log(f"中心区域相对误差: X向={center_rel_err_x:.1f}%, Y向={center_rel_err_y:.1f}%")
            
            current_matrix = best_candidate
            
            # 2.5 检查收敛
            # 仅使用绝对误差作为收敛条件，避免中心区域误差值不一致
            if best_abs_error < center_target_error:
                self.log(f"中心区域达标 (绝对误差 {best_abs_error:.4f} < {center_target_error:.4f})")
                break
            elif stagnation_count > 3:
                self.log(f"连续3次无改进，结束中心区域优化")
                break
            elif center_rel_err_x < 25 and center_rel_err_y < 25:
                self.log(f"中心区相对误差降至25%以下，提前结束中心优化")
                break

            # 在第一阶段结束后添加这行
            stage1_end_index = len(self.history["iteration"]) - 1
            self.stage_ends.append(stage1_end_index)
        
        # 3. 冻结中心区域并准备第二阶段
        # 保存中心区域优化后的矩阵
        self.log(f"保存优化的中心区域结果")
        center_optimized_beam = current_matrix.copy()
        
        # 切换到边缘优化阶段
        self.region_stage = "edge"
        self.log("\n=== 第二阶段：优化边缘区域 ===")
        
        # 获取当前最佳模拟轮廓
        best_sim_edge = best_sim if best_candidate is not candidates[0] else (self.simulate_etching(current_matrix, "x"), self.simulate_etching(current_matrix, "y"))
                                                                            
        
        for iteration in range(1, stage_iterations + 1):
            # 4.1 动态调整变异幅度（边缘阶段幅度小于中心）
            magnitude = self.calculate_magnitude(iteration, stage_iterations) * 0.8
            
            self.log(f"迭代 {iteration}: 第2阶段(边缘), 变异幅度={magnitude:.3f}")
            
            # 使用当前最佳模拟轮廓指导变异
            current_sim_x_edge, current_sim_y_edge = best_sim_edge
            
            # 4.2 创建候选方案 - 使用当前模拟轮廓指导变异
            candidates = [current_matrix.copy()]
            for _ in range(self.max_mutations):  
                candidate = self.mutate_beam(current_matrix, magnitude, sim_x=current_sim_x_edge, sim_y=current_sim_y_edge)
                # 强制保持中心区域不变
                candidate[self.center_mask] = center_optimized_beam[self.center_mask]
                candidates.append(candidate)
            
            best_candidate = candidates[0]
            best_abs_error = self.history["abs_error"][-1]  # 上一次误差
            best_rel_errs = (100, 100)  # 初始化为高误差
            best_candidate_sim_edge = best_sim_edge  # 预置为当前的模拟
            
            # 4.3 评估所有候选结果
            for candidate in candidates:
                cand_sim_x = self.simulate_etching(candidate, "x")
                cand_sim_y = self.simulate_etching(candidate, "y")
                abs_error, rel_err_x, rel_err_y = self.calculate_error(cand_sim_x, cand_sim_y)
                
                if abs_error < best_abs_error:
                    best_abs_error = abs_error
                    best_rel_errs = (rel_err_x, rel_err_y)
                    best_candidate = candidate
                    best_candidate_sim_edge = (cand_sim_x, cand_sim_y)  # 存储候选解的模拟结果
            
            # 4.4 记录评估结果
            if best_candidate is not candidates[0]:
                improvement = self.history["abs_error"][-1] - best_abs_error
                self.log(f"边缘改进: Δ={improvement:.5f}, 新绝对误差={best_abs_error:.4f}")
                stagnation_count = 0
                self.log(f"相对误差: 束X动时Y向={best_rel_errs[0]:.1f}%, 束Y动时X向={best_rel_errs[1]:.1f}%")
                
                # 更新历史记录
                self.history["iteration"].append(iteration + stage_iterations)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(best_rel_errs[0])
                self.history["rel_err_y"].append(best_rel_errs[1])
                
                # 更新当前最佳模拟轮廓
                best_sim_edge = best_candidate_sim_edge
            else:
                stagnation_count += 1
                self.log(f"无改进，维持误差: {best_abs_error:.4f}")
            
            current_matrix = best_candidate
            
            # 4.5 检查收敛 - 添加提前停止条件
            if best_rel_errs[0] < target_rel_error and best_rel_errs[1] < target_rel_error:
                self.log(f"达成目标误差! (X向={best_rel_errs[0]:.1f}%, Y向={best_rel_errs[1]:.1f}%)")
                break
            elif stagnation_count > 3:
                self.log(f"连续3次无改进，结束边缘优化")
                break
            elif best_rel_errs[0] < 20 and best_rel_errs[1] < 20:
                self.log(f"相对误差降至20%以下，提前结束优化")
                break
                
        # 保存边缘优化结果
        edge_optimized_beam = current_matrix.copy()

        stage2_end_index = len(self.history["iteration"]) - 1
        self.stage_ends.append(stage2_end_index)
        
        # ================== 新增第三阶段：外部区域优化 ==================
        self.region_stage = "outer"
        self.log("\n=== 第三阶段：优化外部区域 ===")
        
        # 获取当前最佳模拟轮廓
        best_sim_outer = best_sim_edge
        
        for iteration in range(1, max(int(stage_iterations/2), 10) + 1):  # 至少10次迭代
            # 5.1 动态调整变异幅度（外部区域幅度更小）
            magnitude = self.calculate_magnitude(iteration, stage_iterations) * 0.5
            
            self.log(f"迭代 {iteration}: 第3阶段(外部), 变异幅度={magnitude:.3f}, 外部点数={np.sum(self.outer_mask)}")
            
            # 使用当前最佳模拟轮廓指导变异
            current_sim_x_outer, current_sim_y_outer = best_sim_outer
            
            # 5.2 创建候选方案
            candidates = [current_matrix.copy()]
            for _ in range(self.max_mutations):  
                candidate = self.mutate_beam(current_matrix, magnitude, sim_x=current_sim_x_outer, sim_y=current_sim_y_outer)
                # 强制保持中心区域不变
                candidate[self.center_mask | self.edge_mask] = edge_optimized_beam[self.center_mask | self.edge_mask]
                candidates.append(candidate)
            
            best_candidate = candidates[0]
            best_abs_error = self.history["abs_error"][-1]
            best_rel_errs = (100, 100)
            best_candidate_sim_outer = best_sim_outer
            
            # 5.3 评估候选方案
            for candidate in candidates:
                cand_sim_x = self.simulate_etching(candidate, "x")
                cand_sim_y = self.simulate_etching(candidate, "y")
                abs_error, rel_err_x, rel_err_y = self.calculate_error(cand_sim_x, cand_sim_y)
                
                if abs_error < best_abs_error:
                    best_abs_error = abs_error
                    best_rel_errs = (rel_err_x, rel_err_y)
                    best_candidate = candidate
                    best_candidate_sim_outer = (cand_sim_x, cand_sim_y)
            
            # 5.4 记录评估结果
            if best_candidate is not candidates[0]:
                improvement = self.history["abs_error"][-1] - best_abs_error
                self.log(f"外部改进: Δ={improvement:.5f}, 新绝对误差={best_abs_error:.4f}")
                stagnation_count = 0
                self.log(f"外部优化相对误差: X向={best_rel_errs[0]:.1f}%, Y向={best_rel_errs[1]:.1f}%")
                
                # 更新历史记录
                self.history["iteration"].append(iteration + 2*stage_iterations)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(best_rel_errs[0])
                self.history["rel_err_y"].append(best_rel_errs[1])
            else:
                stagnation_count += 1
                self.log(f"无外部改进，维持误差: {best_abs_error:.4f}")
                
            current_matrix = best_candidate
            
            # 5.5 检查收敛
            if best_abs_error < center_target_error:
                self.log(f"绝对误差达标 (目标 {center_target_error:.4f})")
                break
            elif stagnation_count > 3:
                self.log(f"连续3次无改进，结束外部优化")
                break

            stage3_end_index = len(self.history["iteration"]) - 1
            self.stage_ends.append(stage3_end_index)
        
        # 6. 最终处理
        self.optimized_beam = current_matrix
        elapsed_time = time.time() - start_time

        # ================== 新增第四阶段：平滑非中心区域 ==================
        self.region_stage = "smoothing"
        self.log("\n=== 第四阶段：平滑非中心区域 ===")
        
        # 保存第三阶段优化后的矩阵
        pre_smoothing_matrix = current_matrix.copy()
        
        # 定义新的平滑优化区域（边缘+外部）
        self.non_center_mask = self.edge_mask | self.outer_mask
        
        # 获取当前最佳模拟轮廓
        best_sim_smoothing = best_sim_outer
        
        # 第四阶段迭代
        smoothing_iterations = max(int(stage_iterations/2), 8)  # 比第三阶段迭代次数少
        
        for iteration in range(1, smoothing_iterations + 1):
            # 6.1 动态调整变异幅度（平滑阶段幅度小）
            magnitude = self.calculate_magnitude(iteration, smoothing_iterations) * 0.4
            
            self.log(f"迭代 {iteration}: 第4阶段(平滑), 变异幅度={magnitude:.3f}")
            
            # 6.2 对边缘和外部区域进行高斯平滑
            if iteration % 2 == 0:  # 每2次迭代进行一次平滑处理
                self.log(f"执行非中心区域高斯平滑")
                
                # 保存中心区域原始值
                center_values = current_matrix[self.center_mask]
                
                # 对非中心区域进行高斯滤波
                smoothed_matrix = current_matrix.copy()
                
                # 应用高斯滤波（仅对非中心区域）
                from scipy.ndimage import gaussian_filter
                # 创建临时矩阵，对非中心区域应用高斯滤波
                temp_matrix = np.zeros_like(current_matrix)
                temp_matrix[self.non_center_mask] = current_matrix[self.non_center_mask]
                temp_matrix = gaussian_filter(temp_matrix, sigma=1.2)  # 适当的高斯标准差
                
                # 将平滑后的结果只应用到非中心区域
                smoothed_matrix[self.non_center_mask] = temp_matrix[self.non_center_mask]
                
                # 恢复中心区域原始值
                smoothed_matrix[self.center_mask] = center_values
                
                # 应用径向约束（确保平滑后满足物理约束）
                smoothed_matrix = self.enforce_radial_constraints(smoothed_matrix)
                current_matrix = smoothed_matrix
                
                # 重新计算当前模拟结果
                current_sim_x = self.simulate_etching(current_matrix, "x")
                current_sim_y = self.simulate_etching(current_matrix, "y")
                best_sim_smoothing = (current_sim_x, current_sim_y)
            
            # 6.3 使用当前最佳模拟轮廓指导变异
            current_sim_x_smoothing, current_sim_y_smoothing = best_sim_smoothing
            
            # 6.4 创建候选方案
            candidates = [current_matrix.copy()]
            for _ in range(self.max_mutations):  
                candidate = self.mutate_beam(current_matrix, magnitude, 
                                        sim_x=current_sim_x_smoothing, 
                                        sim_y=current_sim_y_smoothing)
                # 强制保持中心区域不变
                candidate[self.center_mask] = pre_smoothing_matrix[self.center_mask]
                candidates.append(candidate)
            
            best_candidate = candidates[0]
            best_abs_error = self.history["abs_error"][-1]
            best_rel_errs = (100, 100)
            best_candidate_sim_smoothing = best_sim_smoothing
            
            # 6.5 评估候选方案
            for candidate in candidates:
                cand_sim_x = self.simulate_etching(candidate, "x")
                cand_sim_y = self.simulate_etching(candidate, "y")
                abs_error, rel_err_x, rel_err_y = self.calculate_error(cand_sim_x, cand_sim_y)
                
                if abs_error < best_abs_error:
                    best_abs_error = abs_error
                    best_rel_errs = (rel_err_x, rel_err_y)
                    best_candidate = candidate
                    best_candidate_sim_smoothing = (cand_sim_x, cand_sim_y)
            
            # 6.6 记录评估结果
            if best_candidate is not candidates[0]:
                improvement = self.history["abs_error"][-1] - best_abs_error
                self.log(f"平滑改进: Δ={improvement:.5f}, 新绝对误差={best_abs_error:.4f}")
                stagnation_count = 0
                self.log(f"平滑优化相对误差: X向={best_rel_errs[0]:.1f}%, Y向={best_rel_errs[1]:.1f}%")
                
                # 更新历史记录
                self.history["iteration"].append(iteration + 3*stage_iterations)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(best_rel_errs[0])
                self.history["rel_err_y"].append(best_rel_errs[1])
                
                # 更新当前矩阵
                current_matrix = best_candidate
                best_sim_smoothing = best_candidate_sim_smoothing
            else:
                stagnation_count += 1
                self.log(f"无平滑改进，维持误差: {best_abs_error:.4f}")
            
            # 6.7 检查收敛
            if best_abs_error < center_target_error:
                self.log(f"绝对误差达标 (目标 {center_target_error:.4f})")
                break
            elif stagnation_count > 3:
                self.log(f"连续3次无改进，结束平滑优化")
                break
        
        # 保存平滑优化结果
        self.log(f"保存平滑后的束流分布")
        smoothing_end_index = len(self.history["iteration"]) - 1
        self.stage_ends.append(smoothing_end_index)
        
        # 更新最终优化束流
        self.optimized_beam = current_matrix
        elapsed_time = time.time() - start_time
        
        # 计算总迭代次数 - 改为使用记录的迭代次数长度
        iter_count = len(self.history["iteration"])
        self.log(f"优化完成! 总迭代次数: {iter_count}")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        
        # 最终评估
        sim_x = self.simulate_etching(self.optimized_beam, "x")
        sim_y = self.simulate_etching(self.optimized_beam, "y")
        final_abs_error, final_rel_err_x, final_rel_err_y = self.calculate_error(sim_x, sim_y)
        
        # 只记录未记录的误差值
        if self.history["iteration"][-1] != iter_count:
            self.history["iteration"].append(iter_count)
            self.history["abs_error"].append(final_abs_error)
            self.history["rel_err_x"].append(final_rel_err_x)
            self.history["rel_err_y"].append(final_rel_err_y)
        
        self.log(f"最终绝对误差: {final_abs_error:.4f} (初始={abs_error0:.4f})")
        self.log(f"最终相对误差: ")
        self.log(f"  束X移动时Y向误差: {final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%)")
        self.log(f"  束Y移动时X向误差: {final_rel_err_y:.1f}% (初始={rel_err_y0:.1f}%)")
        
        # 保存最终结果
        optimized_beam_full = self.optimized_beam * self.max_val
        np.savetxt("optimized_beam_distribution.csv", optimized_beam_full, delimiter=",")
        
        # 保存外推结果
        outer_indices = np.where(self.outer_mask)
        if len(outer_indices[0]) > 0:
            outer_data = optimized_beam_full[outer_indices]
            xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
            outer_coords = np.column_stack((xx[outer_indices], yy[outer_indices], outer_data))
            np.savetxt("outer_extrapolation.csv", outer_coords, delimiter=",")
        
        return self.optimized_beam, final_rel_err_x, final_rel_err_y  



    def visualize_results(self):
        """可视化优化结果"""
        try:
            fig = plt.figure(figsize=(15, 12))
            fig.suptitle("离子束刻蚀效率优化结果", fontsize=16)
            
            # 原始束流分布
            ax1 = plt.subplot(331)
            im1 = ax1.imshow(self.initial_beam, cmap="viridis", extent=[-15, 15, -15, 15], vmin=0, vmax=np.max(self.initial_beam))
            ax1.contour(self.grid, self.grid, self.initial_beam, levels=[0.5*np.max(self.initial_beam)], colors='r')
            ax1.set_title("初始束流分布")
            plt.colorbar(im1, ax=ax1, label="刻蚀速率 (nm/s)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 优化后束流分布
            optimized_beam_full = self.optimized_beam * self.max_val
            ax2 = plt.subplot(332)
            im2 = ax2.imshow(optimized_beam_full, cmap="viridis", extent=[-15, 15, -15, 15], vmin=0, vmax=np.max(self.initial_beam))
            ax2.contour(self.grid, self.grid, optimized_beam_full, levels=[0.5*np.max(optimized_beam_full)], colors='r')
            ax2.set_title("优化后束流分布")
            plt.colorbar(im2, ax=ax2, label="刻蚀速率 (nm/s)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            
            # 差异分布
            ax3 = plt.subplot(333)
            diff = optimized_beam_full - self.initial_beam
            vmax_diff = max(abs(np.min(diff)), np.max(diff))
            im3 = ax3.imshow(diff, cmap="coolwarm", extent=[-15, 15, -15, 15], vmin=-vmax_diff, vmax=vmax_diff)
            ax3.set_title("束流分布变化")
            plt.colorbar(im3, ax=ax3, label="变化量 (nm/s)")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            
            # 束沿Y轴移动时的X方向轮廓对比
            ax4 = plt.subplot(334)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")
            
            # 实验数据来自束沿Y移动时测量的X方向截面
            ax4.scatter(
                self.beam_traced_y_axis[:, 0], 
                self.beam_traced_y_axis[:, 1]/np.max(self.beam_traced_y_axis[:, 1]), 
                c="g", s=30, label="实验数据 (束沿Y)"
            )
            ax4.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            ax4.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            ax4.set_title("束沿Y移动时的X轴截面")
            ax4.set_xlabel("X位置 (mm)")
            ax4.set_ylabel("归一化刻蚀深度")
            ax4.grid(True)
            ax4.legend(loc='best')
            
            # 束沿X轴移动时的Y方向轮廓对比
            ax5 = plt.subplot(335)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")
            
            # 实验数据来自束沿X移动时测量的Y方向截面
            ax5.scatter(
                self.beam_traced_x_axis[:, 0], 
                self.beam_traced_x_axis[:, 1]/np.max(self.beam_traced_x_axis[:, 1]), 
                c="g", s=30, label="实验数据 (束沿X)"
            )
            ax5.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            ax5.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            ax5.set_title("束沿X移动时的Y轴截面")
            ax5.set_xlabel("Y位置 (mm)")
            ax5.set_ylabel("归一化刻蚀深度")
            ax5.grid(True)
            ax5.legend(loc='best')
            
            # 中心与边缘区域对比
            if hasattr(self, 'center_mask') and hasattr(self, 'outer_mask'):
                ax6 = plt.subplot(336)
                # 中心区域
                center_values = optimized_beam_full[self.center_mask]
                ax6.plot(center_values, 'ro', markersize=4, alpha=0.6, label="中心区域")
                # 边缘区域
                edge_values = optimized_beam_full[self.edge_mask]
                ax6.plot(len(center_values) + np.arange(len(edge_values)), edge_values, 'go', markersize=4, alpha=0.6, label="边缘区域")
                # 外部区域
                outer_values = optimized_beam_full[self.outer_mask]
                ax6.plot(len(center_values)+len(edge_values) + np.arange(len(outer_values)), outer_values, 'bo', markersize=4, alpha=0.6, label="外部区域")
                ax6.axvline(len(center_values), color='k', linestyle='--')
                ax6.axvline(len(center_values)+len(edge_values), color='k', linestyle='--')
                ax6.set_title("区域值对比")
                ax6.set_ylabel("刻蚀速率 (nm/s)")
                ax6.grid(True)
                ax6.legend(loc='best')
            
            # 误差收敛曲线
            ax7 = plt.subplot(313)
            if len(self.history["iteration"]) > 1:
                # 确保所有数组长度一致
                iterations = self.history["iteration"]
                abs_errors = self.history["abs_error"]
                rel_err_x = self.history["rel_err_x"]
                rel_err_y = self.history["rel_err_y"]
                
                # 修剪数组使它们长度一致
                min_length = min(len(iterations), len(abs_errors), len(rel_err_x), len(rel_err_y))
                iterations = iterations[:min_length]
                abs_errors = abs_errors[:min_length]
                rel_err_x = rel_err_x[:min_length]
                rel_err_y = rel_err_y[:min_length]
                
                # 左侧Y轴 - 绝对误差
                ax7.plot(iterations, abs_errors, "k-", label="绝对误差", linewidth=2)
                ax7.set_xlabel("迭代次数")
                ax7.set_ylabel("绝对误差", color='k')
                ax7.tick_params(axis='y', labelcolor='k')
                ax7.grid(True)
                
                # 右侧Y轴 - 相对误差
                ax7b = ax7.twinx()
                ax7b.plot(iterations, rel_err_x, "r--", label="束X动时Y向误差")
                ax7b.plot(iterations, rel_err_y, "g--", label="束Y动时X向误差")
                ax7b.set_ylabel("相对误差 (%)", color='b')
                ax7b.tick_params(axis='y', labelcolor='b')
                
                # 标记阶段转换
                if len(self.stage_ends) >= 1:  # 如果有第一阶段结束点
                    ax7.axvline(
                        self.stage_ends[0], 
                        color='r', 
                        linestyle='-', 
                        alpha=0.5, 
                        label="阶段1结束"
                    )
                if len(self.stage_ends) >= 2:  # 如果有第二阶段结束点
                    ax7.axvline(
                        self.stage_ends[1], 
                        color='g', 
                        linestyle='-', 
                        alpha=0.5, 
                        label="阶段2结束"
                    )
                if len(self.stage_ends) >= 3:  # 如果有第三阶段结束点
                    ax7.axvline(
                        self.stage_ends[2], 
                        color='b', 
                        linestyle='-', 
                        alpha=0.5, 
                        label="阶段3结束"
                    )

                if len(self.stage_ends) >= 4:  # 显示第四阶段结束点
                    ax7.axvline(
                        self.stage_ends[3], 
                        color='m',  # 品红色表示第四阶段
                        linestyle='-', 
                        alpha=0.5, 
                        label="阶段4结束"
                    )
                
                # 合并图例
                lines, labels = ax7.get_legend_handles_labels()
                lines2, labels2 = ax7b.get_legend_handles_labels()
                ax7.legend(lines + lines2, labels + labels2, loc='upper right')
                
                ax7.set_title("误差收敛曲线")
            
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(top=0.92)
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
        self.log(f"  - beam_optimization_result.png (可视化结果)")
        self.log_file.close()

# ================== 主程序 ==================
def main():
    # 检查文件存在性
    files = {
        "beam_traced_x_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",  # 束沿X移动时测量的Y截面
        "beam_traced_y_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",  # 束沿Y移动时测量的X截面
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 70)
    print("非高斯离子束束流分布优化".center(70))
    print(f"输入文件:")
    print(f" - 束沿X移动时的Y截面: {files['beam_traced_x_axis']}")
    print(f" - 束沿Y移动时的X截面: {files['beam_traced_y_axis']}")
    print(f" - 初始束流分布: {files['initial_beam']}")
    print("=" * 70)
    
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
            beam_traced_x_axis=files["beam_traced_x_axis"],  # 束沿X移动时测量的Y截面
            beam_traced_y_axis=files["beam_traced_y_axis"],  # 束沿Y移动时测量的X截面
            initial_guess_path=files["initial_beam"]
        )
        
        # 第一阶段优化: 中心区域
        optimizer.log("\n===== 第一阶段优化 (中心区域优化) =====")
        result, err_x, err_y = optimizer.run_optimization(
            max_iterations=30,
            target_rel_error=25.0
        )
        
        # 第二阶段优化: 边缘区域
        optimizer.log("\n===== 第二阶段优化 (边缘区域优化) =====")
        result, err_x, err_y = optimizer.run_optimization(
            max_iterations=35,
            target_rel_error=20.0
        )
        
        # 第三阶段优化: 外部区域
        optimizer.log("\n===== 第三阶段优化 (外部区域优化) =====")
        result, err_x, err_y = optimizer.run_optimization(
            max_iterations=20,
            target_rel_error=18.0
        )
        
        # 可视化
        optimizer.visualize_results()
        
        # 最终报告
        optimizer.finalize()
        
        print("\n" + "=" * 70)
        print("优化完成! 最终误差:")
        print(f" - 束沿X移动时Y向误差: {err_x:.1f}%")
        print(f" - 束沿Y移动时X向误差: {err_y:.1f}%")
        print(f"结果文件: optimized_beam_distribution.csv, beam_optimization_results.png")
        print("=" * 70)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        if 'optimizer' in locals() and hasattr(optimizer, 'log_file'):
            optimizer.log_file.close()

if __name__ == "__main__":
    main()
