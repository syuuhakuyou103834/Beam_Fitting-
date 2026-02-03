import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import trapezoid, simpson
from scipy.ndimage import gaussian_filter
from matplotlib import gridspec
import traceback

# ====================== 提高精度版本 ======================
# 精度参数配置
HIGH_PRECISION = 0.1  # 高精度网格步长(单位：mm)
LOW_PRECISION = 1.0   # 低精度网格步长(单位：mm)

# ====================== 修复中文字体支持 ======================
def setup_plotting():
    """配置绘图环境，解决中文字体问题"""
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'],
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 300
    })
    
setup_plotting()

class BeamEfficiencyOptimizer:
    def __init__(self, beam_traced_x_axis, beam_traced_y_axis, initial_guess_path, 
                 grid_bound=15.0, highres_points=None, lowres_points=None):
        """
        高级束流分布优化器 - 高精度版本 (0.1mm)
        """
        # 创建日志文件
        self.log_file = open("beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("高级离子束刻蚀效率优化引擎 - 高精度版本")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        
        # 计算所需网格点数（基于精度要求）
        if highres_points is None:
            highres_points = int(grid_bound * 2 / HIGH_PRECISION) + 1
        if lowres_points is None:
            lowres_points = int(grid_bound * 2 / LOW_PRECISION) + 1
            
        self.highres_points = highres_points
        self.lowres_points = lowres_points
        
        self.log(f"多精度网格系统: 优化精度={HIGH_PRECISION}mm ({highres_points}x{highres_points})")
        self.log(f"           低精度网格: {LOW_PRECISION}mm ({lowres_points}x{lowres_points})")
        self.log("输入文件:")
        self.log(f"  - X截面: {beam_traced_x_axis}")
        self.log(f"  - Y截面: {beam_traced_y_axis}")
        self.log(f"  - 初始束流: {initial_guess_path}")
        self.log("=" * 30)
        
        # 优化参数
        self.opt_radius = 15.0     # 优化半径 (mm)
        self.scan_velocity = 30.0  # 扫描速度 (mm/s)
        self.drift_sigma = 1.8     # 漂移校正标准差 (mm)
        self.smooth_sigma = 0.8    # 高斯平滑参数
        self.max_mutations = 4     # 变异候选数
        self.uni_penalty = 0.5     # 单峰性约束的惩罚因子
        
        # 网格系统参数
        self.grid_bound = grid_bound
        
        # 创建高精度网格 (HIGH_PRECISION mm间隔)
        self.highres_grid = np.linspace(-grid_bound, grid_bound, highres_points)
        self.highres_spacing = 2 * grid_bound / (highres_points - 1)
        
        # 创建原始精度网格 (LOW_PRECISION mm间隔)
        self.lowres_grid = np.linspace(-grid_bound, grid_bound, lowres_points)
        self.lowres_spacing = 2 * grid_bound / (lowres_points - 1)
        
        # 设置主网格为高精度网格
        self.grid = self.highres_grid
        self.grid_spacing = self.highres_spacing
        
        # 加载实验数据并进行平移预处理
        self.log("加载并预处理实验数据...")
        self.x_data = self.load_and_preprocess_data(beam_traced_x_axis)
        self.y_data = self.load_and_preprocess_data(beam_traced_y_axis)
        
        # 创建高精度优化掩膜 (使用已定义的opt_radius)
        self.create_optimization_mask()
        
        # 加载并校准初始猜测
        self.load_initial_beam(initial_guess_path)
        
        # 中心点位置
        self.center_i, self.center_j = self.find_center(self.initial_beam)
        self.log(f"初始中心点位置: ({self.grid[self.center_i]:.2f}, {self.grid[self.center_j]:.2f})")
        
        # 创建距离矩阵（以中心为原点）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.distance_from_center = np.sqrt(
            (xx - self.grid[self.center_i])**2 + 
            (yy - self.grid[self.center_j])**2
        )
        
        # 创建16个阶段的高精度点集掩膜（适应更高精度）
        self.num_stages = 16  # 增加阶段数以适应高精度网格
        self.stage_masks = []  # 存储阶段信息
        self.create_stage_masks()
        
        self.stage_ends = []  # 存储阶段结束点的迭代索引
        
        # 历史记录 - 新增单峰性误差记录
        self.history = {
            "iteration": [],
            "abs_error": [],
            "rel_err_x": [],
            "max_val": [(self.max_val if hasattr(self, 'max_val') else 1.0)],
            "etch_vol_error": [],   # 刻蚀体积误差
            "unimodal_error": []    # 单峰性误差
        }
        
        # 初始优化束流
        self.optimized_beam = self.initial_beam / self.max_val
        
        # 强制的单峰性约束
        self.optimized_beam = self.enforce_strict_unimodality(self.optimized_beam)
        self.log("初始单峰性约束完成")
        
        # 初始单峰性评估
        uni_error = self.validate_unimodality(self.optimized_beam)
        self.log(f"初始单峰性误差: {uni_error:.4f}")

    def log(self, message):
        """记录带时间戳的日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_file.write(log_entry + "\n")
        self.log_file.flush()
        
    def log_progress(self, current, total, stage=None):
        """显示优化进度"""
        if stage is not None:
            prefix = f"[阶段 {stage+1}/{self.num_stages}]"
        else:
            prefix = "[优化进度]"
        
        progress = int(100 * current/total)
        bar = '=' * (progress//2)
        sys.stdout.write(f"\r{prefix} |{bar:{50}}| {progress}%")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    def find_center(self, beam_matrix):
        """找到束流中心点位置"""
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        return center_i, center_j

    def load_and_preprocess_data(self, file_path):
        """加载实验数据并进行平移预处理 - 使用三次样条插值"""
        self.log(f"加载实验数据: {file_path} (使用三次样条插值)")
        
        # 加载原始数据
        try:
            data = np.loadtxt(file_path, delimiter=",")
        except Exception as e:
            self.log(f"  加载失败: {str(e)}")
            # 返回空数据
            return np.column_stack((self.grid, np.zeros_like(self.grid)))
        
        # 确保数据有位置和值两列
        if data.shape[1] != 2:
            if data.size % 2 == 0:
                data = data.reshape(-1, 2)
            else:
                self.log("  错误: 数据无法转换为两列")
                return np.column_stack((self.grid, np.zeros_like(self.grid)))
        
        # 找到峰值位置
        peak_idx = np.argmax(data[:, 1])
        peak_pos = data[peak_idx, 0]
        self.log(f"  原始峰值位置: {peak_pos:.2f}mm")
        
        # 计算平移量 (将峰值移到零位置)
        shift_value = -peak_pos
        data[:, 0] += shift_value
        
        # 记录原始数据范围
        original_min = np.min(data[:, 1])
        original_max = np.max(data[:, 1])
        original_range = original_max - original_min
        
        # 创建三次样条插值器
        try:
            interpolator = interp1d(
                data[:, 0], data[:, 1], 
                kind='cubic',  # 使用三次样条插值提高精度
                bounds_error=False, 
                fill_value=0.0
            )
        except:
            # 如果插值失败，使用线性插值
            interpolator = interp1d(
                data[:, 0], data[:, 1], 
                kind='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
        
        # 创建新网格数据 (0.1mm精度)
        new_values = interpolator(self.grid)
        
        # 应用高斯平滑消除插值噪声
        new_values = gaussian_filter(new_values, sigma=1.0)
        
        # 确保非负值
        new_values[new_values < 0] = 0
        
        # 恢复原始数据范围
        current_min = np.min(new_values)
        current_max = np.max(new_values)
        current_range = current_max - current_min
        
        if current_range > 0 and original_range > 0:
            scale_factor = original_range / current_range
            offset = original_min - np.min(new_values)
            new_values = new_values * scale_factor + offset
        
        # 创建新数据集
        new_data = np.column_stack((self.grid, new_values))
        
        # 找到新的峰值位置
        new_peak_idx = np.argmax(new_values)
        self.log(f"  平移后峰值位置: x={self.grid[new_peak_idx]:.2f}mm | 峰强度: {new_values[new_peak_idx]:.2f}")
        
        return new_data
    
    def create_optimization_mask(self):
        """创建高精度优化区域掩膜（使用已定义的opt_radius）"""
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.distance_from_center = np.sqrt(xx**2 + yy**2)
        self.optimization_mask = (self.distance_from_center <= self.opt_radius)
        self.log(f"高精度优化区域包含 {np.sum(self.optimization_mask)} 个点 (半径={self.opt_radius}mm)")

    def load_initial_beam(self, file_path):
        """加载、插值并校准初始束流分布 - 使用更高精度插值"""
        try:
            # 1. 加载原始1mm精度数据
            lowres_beam = np.genfromtxt(file_path, delimiter=",")
            if lowres_beam.shape != (self.lowres_points, self.lowres_points):
                if lowres_beam.size == self.lowres_points * self.lowres_points:
                    lowres_beam = lowres_beam.reshape((self.lowres_points, self.lowres_points))
                else:
                    self.log(f"警告: 初始束流尺寸不兼容 ({lowres_beam.shape} vs {self.lowres_points}x{self.lowres_points})")
                    raise ValueError("初始束流尺寸不兼容")
            
            # 2. 使用三次样条插值到0.1mm高精度网格
            interpolator = RegularGridInterpolator(
                (self.lowres_grid, self.lowres_grid),
                lowres_beam,
                method='cubic',  # 使用更高精度的插值方法
                bounds_error=False,
                fill_value=0.0
            )
            
            # 创建高精度网格
            xx_highres, yy_highres = np.meshgrid(
                self.highres_grid, self.highres_grid, indexing='ij'
            )
            
            # 执行插值
            points = np.stack((xx_highres, yy_highres), axis=-1)
            highres_beam = interpolator(points)
            
            # 应用高斯平滑减少插值噪声
            highres_beam = gaussian_filter(highres_beam, sigma=1.0)
            
            # 3. 应用自适应校准
            # 检测X方向实验数据的峰值
            peak_val_x = np.max(self.x_data[:, 1])
            
            # 在中心区域计算平均值 (±0.5mm)
            center_mask = (
                (np.abs(self.highres_grid) < 0.5)[:, None] & 
                (np.abs(self.highres_grid) < 0.5)[None, :]
            )
            # 只在优化区域内考虑
            center_mask = center_mask & self.optimization_mask
            
            if np.sum(center_mask) > 0:
                sim_peak_val = np.mean(highres_beam[center_mask])
            else:
                sim_peak_val = np.max(highres_beam)
            
            if sim_peak_val < 1e-6:
                self.log(f"警告: 中心强度过低 ({sim_peak_val:.2f})，跳过校准")
                self.initial_beam = highres_beam
            else:
                calibration_factor = peak_val_x / sim_peak_val
                self.log(f"校准因子: {calibration_factor:.2f} (实验峰值={peak_val_x:.2f}, 模拟峰值={sim_peak_val:.2f})")
                
                # 应用刻蚀深度校准
                self.initial_beam = highres_beam * calibration_factor
            
            # 确保所有值为非负
            self.initial_beam = np.maximum(self.initial_beam, 0)
            
            # 更新最大值
            self.max_val = np.max(self.initial_beam)
            if self.max_val < 1e-6:
                self.log("警告: 最大束流强度过低，重置为1.0")
                self.max_val = 1.0
                self.initial_beam = self.initial_beam + 1e-3  # 避免零值
            
            # 找到中心点并确保其在合理位置
            self.center_i, self.center_j = self.find_center(self.initial_beam)
            center_x = self.grid[self.center_i]
            center_y = self.grid[self.center_j]
            
            # 修正中心点位置 (如果偏移过大)
            if abs(center_x) > 1.0 or abs(center_y) > 1.0:
                # 寻找最近的点
                center_i = np.argmin(np.abs(self.grid))
                center_j = np.argmin(np.abs(self.grid))
                self.center_i, self.center_j = center_i, center_j
                self.log(f"中心点修正为 (0, 0)")
            
            self.log(f"高精度初始束流: 尺寸={self.initial_beam.shape}, 最大速率={self.max_val:.4f} nm/s")
            self.log(f"中心点位置: ({self.grid[self.center_i]:.2f}, {self.grid[self.center_j]:.2f})")
            
            # 确保中心点值是最大值
            self.initial_beam[self.center_i, self.center_j] = self.max_val
            self.log(f"确认中心点强度: {self.initial_beam[self.center_i, self.center_j]:.2f} nm/s")
            
            # 应用凸性约束确保形状合理
            self.initial_beam = self.enforce_convexity(self.initial_beam)
            
        except Exception as e:
            self.log(f"加载初始束流失败: {str(e)}")
            self.log(traceback.format_exc())
            
            # 创建默认初始束流 (高斯分布)
            xx, yy = np.meshgrid(self.highres_grid, self.highres_grid, indexing='ij')
            r = np.sqrt(xx**2 + yy**2)
            self.initial_beam = np.exp(-r**2 / (2*8.0**2))
            self.max_val = 1.0
            self.center_i, self.center_j = len(self.grid)//2, len(self.grid)//2
            self.log(f"使用高斯分布作为初始束流")

    def create_stage_masks(self):
        """创建16个阶段的高精度点集掩膜（适应0.1mm高精度）"""
        self.stage_masks = []
        total_mask = np.zeros_like(self.optimization_mask, dtype=bool)
        
        # 获取网格索引
        xx, yy = np.meshgrid(range(self.highres_points), range(self.highres_points), indexing='ij')
        
        # 第一阶段 (集合1): x=0 或 y=0 的点
        stage0_mask = ((np.abs(self.grid[xx]) < 1e-5) | (np.abs(self.grid[yy]) < 1e-5))
        self.stage_masks.append(stage0_mask)
        total_mask = total_mask | stage0_mask
        
        # 其他阶段：1-15mm环 (间隔更密)
        radius_steps = np.linspace(1, 15, 15)  # 更密的半径间隔
        self.log(f"采用 {len(radius_steps)} 个半径环进行阶段优化")
        
        for k, radius in enumerate(radius_steps):
            threshold = self.grid_spacing * 1.5  # 稍微扩大阈值以覆盖更多点
            
            # 水平/垂直线上的点
            stage_mask = (
                (np.abs(np.abs(self.grid[xx]) - radius) < threshold) | 
                (np.abs(np.abs(self.grid[yy]) - radius) < threshold)
            )
            stage_mask = stage_mask & ~total_mask
            
            # 添加斜线方向45度和135度
            angle_mask = (
                (np.abs(np.abs(self.grid[xx]) - radius*np.cos(np.pi/4)) < threshold) & 
                (np.abs(np.abs(self.grid[yy]) - radius*np.sin(np.pi/4)) < threshold)
            )
            stage_mask = stage_mask | (angle_mask & ~total_mask)
            
            self.stage_masks.append(stage_mask)
            total_mask = total_mask | stage_mask
        
        # 最后一个阶段: 剩余所有点
        stage_last_mask = ~total_mask
        self.stage_masks.append(stage_last_mask)
        
        # 日志记录每个阶段的点数
        for i, mask in enumerate(self.stage_masks):
            self.log(f"阶段{i+1}/{self.num_stages}: {np.sum(mask)}点")
        
        self.current_stage_idx = 0

    def create_interpolator(self, beam_matrix):
        """创建高精度双线性插值器"""
        return RegularGridInterpolator(
            (self.grid, self.grid),
            beam_matrix * self.max_val,  # 反归一化
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
    
    def enforce_strict_unimodality(self, beam_matrix):
        """
        强制的单峰性约束：确保束流分布是单峰的（只在中心点存在唯一极大值）
        1. 从中心点向外沿射线路径检查
        2. 强制射线方向强度单调递减
        3. 确保外层值不大于内层值
        """
        # 确定中心点
        center_i, center_j = self.center_i, self.center_j
        center_val = beam_matrix[center_i, center_j] if beam_matrix[center_i, center_j] > 0 else 1.0
        
        # 创建新束流副本
        new_beam = beam_matrix.copy()
        
        # 在所有射线方向应用严格递减约束
        num_rays = 72  # 增加射线数量以适应高精度网格
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        max_radius = self.opt_radius
        
        for angle in angles:
            # 构建射线路径 (0.1mm步长)
            distances = np.arange(0.25, max_radius, 0.25)
            ray_points = []
            
            for dist in distances:
                x_pos = dist * np.cos(angle)
                y_pos = dist * np.sin(angle)
                
                # 找到最近的网格点
                x_idx = np.argmin(np.abs(self.grid - x_pos))
                y_idx = np.argmin(np.abs(self.grid - y_pos))
                
                if 0 <= x_idx < self.highres_points and 0 <= y_idx < self.highres_points:
                    # 检查是否已在列表中
                    if not any(pt[0] == x_idx and pt[1] == y_idx for pt in ray_points):
                        ray_points.append((x_idx, y_idx, dist))
            
            # 按距离排序
            ray_points = sorted(ray_points, key=lambda x: x[2])
            
            # 递减约束应用
            if not ray_points:
                continue
                
            # 从中心点开始
            prev_x, prev_y = center_i, center_j
            prev_dist = 0.0
            prev_val = center_val
            
            for pt in ray_points:
                x_idx, y_idx, dist = pt
                current_val = new_beam[x_idx, y_idx]
                
                # 目标值应小于前一个值（带安全阈值）
                threshold = 0.01
                decay_factor = np.exp(-(dist - prev_dist) / 10.0)  # 指数衰减因子
                max_allowed = prev_val * decay_factor + threshold
                
                if current_val > max_allowed:
                    # 设置为最大允许值
                    new_beam[x_idx, y_idx] = max_allowed
                    prev_val = max_allowed
                else:
                    # 只要不大于就保持，但也要确保后续点不会太大
                    prev_val = max_allowed if current_val > prev_val else current_val
                
                prev_x, prev_y = x_idx, y_idx
                prev_dist = dist
        
        # 应用高斯平滑
        smoothed_beam = gaussian_filter(new_beam, sigma=self.smooth_sigma)
        
        # 确保中心点仍然是最大值
        smoothed_beam[center_i, center_j] = center_val
        
        # 维持非负性
        smoothed_beam = np.maximum(smoothed_beam, 0)
        
        return smoothed_beam

    def enforce_convexity(self, beam_matrix, strict=True):
        """
        强制束流分布凸性（辐射状递减）
        确保从中心向外单调递减，带高斯平滑
        """
        # 确定中心极值点位置
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        center_val = beam_matrix[center_i, center_j]
        
        # 创建新束流副本
        new_beam = beam_matrix.copy()
        
        # 在72个射线方向应用严格递减约束
        angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
        max_radius = self.opt_radius
        
        for angle in angles:
            # 构建射线路径 (0.1mm步长)
            distances = np.arange(0.25, max_radius, 0.25)
            ray_points = []
            
            for dist in distances:
                x_pos = dist * np.cos(angle)
                y_pos = dist * np.sin(angle)
                
                # 找到最近的网格点
                x_idx = np.argmin(np.abs(self.grid - x_pos))
                y_idx = np.argmin(np.abs(self.grid - y_pos))
                
                if 0 <= x_idx < self.highres_points and 0 <= y_idx < self.highres_points:
                    # 检查是否已在列表中
                    if not any(pt[0] == x_idx and pt[1] == y_idx for pt in ray_points):
                        ray_points.append((x_idx, y_idx, dist))
            
            # 按距离排序
            ray_points = sorted(ray_points, key=lambda x: x[2])
            
            # 递减约束应用
            if not ray_points:
                continue
                
            # 从中心点开始
            prev_x, prev_y = center_i, center_j
            prev_val = center_val
            prev_dist = 0.0
            
            for idx, pt in enumerate(ray_points):
                x_idx, y_idx, dist = pt
                current_val = new_beam[x_idx, y_idx]
                
                # 目标值应小于前一个值
                if idx == 0:  # 第一个点使用中心值
                    prev_val = center_val
                
                # 计算允许的最大值（考虑距离衰减）
                decay = max(0.8, 1 - dist/self.opt_radius)  # 距离中心越远衰减越大
                max_allowed = prev_val * decay
                
                if current_val > max_allowed:
                    new_beam[x_idx, y_idx] = max_allowed
                    current_val = max_allowed
                
                prev_val = current_val
        
        # 应用高斯平滑
        smoothed_beam = gaussian_filter(new_beam, sigma=0.7)
        
        # 维持非负性
        smoothed_beam = np.maximum(smoothed_beam, 0)
        
        # 确保中心点仍然是最大值
        smoothed_beam[center_i, center_j] = np.max(smoothed_beam)
        
        return smoothed_beam

    def validate_unimodality(self, beam_matrix):
        """
        验证束流分布的单峰性
        从中心点出发沿所有方向检查是否单调递减
        返回单峰性误差（违反单调性的次数和幅度）
        """
        # 确定中心极值点位置
        center_i, center_j = np.unravel_index(np.argmax(beam_matrix), beam_matrix.shape)
        
        # 初始化误差
        error_count = 0
        error_magnitude = 0.0
        
        # 确定检查距离范围
        step_size = min(0.5, self.highres_spacing*2)
        distances = np.arange(step_size, self.opt_radius, step_size)
        num_directions = 48  # 检查方向数量
        
        for angle in np.linspace(0, 2*np.pi, num_directions, endpoint=False):
            prev_val = beam_matrix[center_i, center_j]
            
            for dist in distances:
                # 计算当前方向的网格坐标
                x_coord = self.grid[center_j] + dist * np.cos(angle)
                y_coord = self.grid[center_i] + dist * np.sin(angle)
                
                # 找到最近的网格索引
                i = np.argmin(np.abs(self.grid - x_coord))
                j = np.argmin(np.abs(self.grid - y_coord))
                
                if (0 <= i < self.highres_points and 
                    0 <= j < self.highres_points and
                    self.distance_from_center[i, j] <= self.opt_radius):
                    
                    current_val = beam_matrix[i, j]
                    
                    # 检查是否违反递减规律
                    if current_val > prev_val:
                        error_count += 1
                        violation = (current_val - prev_val) / prev_val
                        error_magnitude += violation
                    
                    prev_val = current_val
        
        # 返回综合误差（违反次数与幅度的乘积）
        if error_count > 0:
            return (1 + error_count) * error_magnitude
        return 0.0

    def simulate_etching(self, beam_matrix, axis='x'):
        """
        模拟束沿垂直轴移动时的轮廓
        axis='x': 束沿Y移动，产生X方向轮廓
        axis='y': 束沿X移动，产生Y方向轮廓
        """
        profile = np.zeros_like(self.grid)
        grid_size = len(self.grid)
        
        # 计算漂移卷积核
        kernel_radius = min(100, int(3 * self.drift_sigma / self.grid_spacing))
        kernel_points = 2 * kernel_radius + 1
        kernel_range = np.linspace(-kernel_radius, kernel_radius, kernel_points) * self.grid_spacing
        kernel = np.exp(-0.5 * (kernel_range/self.drift_sigma)**2)
        kernel /= np.sum(kernel)  # 归一化
        
        if axis == 'x':  # 扫描Y方向，测量X截面
            # 对于每个X位置
            for i in range(grid_size):
                # 提取当前X位置的整条Y线
                rates = beam_matrix[i, :] * self.max_val
                
                try:
                    # 添加漂移扩散效应
                    if self.drift_sigma > 0 and len(rates) > len(kernel):
                        convolved_rates = np.convolve(rates, kernel, mode='same')
                    else:
                        convolved_rates = rates
                    
                    # 使用辛普森积分
                    if len(convolved_rates) >= 3:
                        profile[i] = simpson(convolved_rates, self.grid) / self.scan_velocity
                    else:
                        profile[i] = trapezoid(convolved_rates, self.grid) / self.scan_velocity
                except:
                    profile[i] = trapezoid(convolved_rates, self.grid) / self.scan_velocity
                    
        elif axis == 'y':  # 扫描X方向，测量Y截面
            # 对于每个Y位置
            for j in range(grid_size):
                # 提取当前Y位置的整条X线
                rates = beam_matrix[:, j] * self.max_val
                
                try:
                    # 添加漂移扩散效应
                    if self.drift_sigma > 0 and len(rates) > len(kernel):
                        convolved_rates = np.convolve(rates, kernel, mode='same')
                    else:
                        convolved_rates = rates
                    
                    # 使用辛普森积分
                    if len(convolved_rates) >= 3:
                        profile[j] = simpson(convolved_rates, self.grid) / self.scan_velocity
                    else:
                        profile[j] = trapezoid(convolved_rates, self.grid) / self.scan_velocity
                except:
                    profile[j] = trapezoid(convolved_rates, self.grid) / self.scan_velocity
        
        # 归一化
        profile_max = np.max(profile) if np.max(profile) > 0 else 1.0
        return profile / profile_max

    def calculate_error(self, sim_x, sim_y):
        """计算高精度模拟结果与实验数据的误差"""
        # 初始化默认值
        abs_error_x = abs_error_y = 1.0
        rel_err_x = rel_err_y = 100.0
        exp_peak_val_x = exp_peak_val_y = 1.0
        sim_x_max = sim_y_max = 1.0
        
        try:
            # X截面误差
            if hasattr(self, 'x_data') and len(self.x_data) > 0:
                # 实验数据
                exp_x_pos = self.x_data[:, 0]
                exp_x_val = self.x_data[:, 1]
                
                # 避免除以零
                exp_peak_val_x = np.max(exp_x_val) if np.max(exp_x_val) > 0 else 1.0
                exp_x_norm = exp_x_val / exp_peak_val_x
                
                # 模拟数据
                sim_x_max = np.max(sim_x) if np.max(sim_x) > 0 else exp_peak_val_x
                sim_x_norm = sim_x / sim_x_max
                
                # 插值到相同网格
                sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_norm)
                
                # 计算绝对误差和相对误差
                abs_error_x = np.mean(np.abs(sim_x_interp - exp_x_norm))
                
                # 只考虑峰值5%以上的数据点
                valid_mask = exp_x_norm > np.max(exp_x_norm)*0.05
                if np.any(valid_mask):
                    rel_err_x = np.mean(np.abs(sim_x_interp[valid_mask] - exp_x_norm[valid_mask]) / 
                                      (exp_x_norm[valid_mask] + 1e-10)) * 100
                else:
                    rel_err_x = np.mean(np.abs(sim_x_interp)) * 100
        except Exception as e:
            self.log(f"X方向误差计算失败: {str(e)}")
        
        try:
            # Y截面误差
            if hasattr(self, 'y_data') and len(self.y_data) > 0:
                exp_y_pos = self.y_data[:, 0]
                exp_y_val = self.y_data[:, 1]
                
                exp_peak_val_y = np.max(exp_y_val) if np.max(exp_y_val) > 0 else 1.0
                exp_y_norm = exp_y_val / exp_peak_val_y
                
                sim_y_max = np.max(sim_y) if np.max(sim_y) > 0 else exp_peak_val_y
                sim_y_norm = sim_y / sim_y_max
                
                sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_norm)
                
                abs_error_y = np.mean(np.abs(sim_y_interp - exp_y_norm))
                
                # 只考虑峰值5%以上的数据点
                valid_mask = exp_y_norm > np.max(exp_y_norm)*0.05
                if np.any(valid_mask):
                    rel_err_y = np.mean(np.abs(sim_y_interp[valid_mask] - exp_y_norm[valid_mask]) / 
                                      (exp_y_norm[valid_mask] + 1e-10)) * 100
                else:
                    rel_err_y = np.mean(np.abs(sim_y_interp)) * 100
        except Exception as e:
            self.log(f"Y方向误差计算失败: {str(e)}")
        
        # 综合绝对误差 (X方向60%权重)
        weights = {'x': 0.6, 'y': 0.4}
        combined_abs_error = weights['x'] * abs_error_x + weights['y'] * abs_error_y
        
        # 计算实验与模拟峰值偏差
        x_peak_ratio = np.abs(sim_x_max - exp_peak_val_x) / exp_peak_val_x if exp_peak_val_x > 0 else 0
        y_peak_ratio = np.abs(sim_y_max - exp_peak_val_y) / exp_peak_val_y if exp_peak_val_y > 0 else 0
        
        # 峰值偏差惩罚（避免改变峰值位置）
        peak_penalty = (weights['x'] * x_peak_ratio**2 + weights['y'] * y_peak_ratio**2) * 0.1
        
        # 总误差加入峰值惩罚
        return combined_abs_error + peak_penalty, rel_err_x

    def mutate_beam(self, beam_matrix, magnitude):
        """基于误差分析的定向变异策略 - 带凸性约束和局部限制"""
        new_beam = beam_matrix.copy()
        if not hasattr(self, 'current_stage_idx') or self.current_stage_idx >= len(self.stage_masks):
            return new_beam
        
        current_mask = self.stage_masks[self.current_stage_idx]
        indices = np.where(current_mask)
        
        # 控制变异密度 (阶段依赖)
        mutation_prob = 0.7 if self.current_stage_idx < 5 else 0.5
        
        for idx in range(len(indices[0])):
            i, j = indices[0][idx], indices[1][idx]
            
            # 随机跳过部分点
            if np.random.rand() > mutation_prob:
                continue
            
            # 变异幅度基于距离衰减
            if hasattr(self, 'distance_from_center') and hasattr(self, 'center_i'):
                dist_to_center = self.distance_from_center[i, j]
                dist_factor = max(0.3, np.exp(-dist_to_center / 10.0))
            else:
                dist_factor = 1.0  # 默认值
            
            # 基础变异幅度 (随阶段递减)
            effective_mag = magnitude * dist_factor * (0.8 + 0.4 * np.random.rand())
            
            # 随机变异方向 (增加/减少) - 避免过大变异
            if beam_matrix[i, j] > 0.2:  # 在较高强度区域倾向减少
                direction = np.random.choice([-1, 0.5], p=[0.7, 0.3])
            else:  # 在较低强度区域增加多样性
                direction = np.random.choice([-0.5, 1], p=[0.3, 0.7])
                
            mutation = direction * effective_mag
            
            # 应用变异 (带边界检查)
            new_val = beam_matrix[i, j] + mutation
            new_val = max(0, min(1.0, new_val))  # 边界约束
            new_beam[i, j] = new_val
        
        # 应用凸性约束
        return self.enforce_convexity(new_beam, strict=False)

    def validate_etch_depth(self, beam_matrix):
        """验证高精度积分总量一致性 (简化)"""
        try:
            # 从实验X方向数据估算总刻蚀体积
            if not hasattr(self, 'x_data') or len(self.x_data) < 2:
                return 0.0
            
            exp_x_val = self.x_data[:, 1]
            exp_x_pos = self.x_data[:, 0]
            
            # 插值到统一网格
            exp_interp = np.interp(self.grid, exp_x_pos, exp_x_val)
            
            # 计算X方向积分
            x_span = max(exp_x_pos) - min(exp_x_pos)
            x_integral = simpson(exp_interp, self.grid)  
            
            # Y方向范围 - 假设与X相同
            exp_total = x_integral * x_span
            
            # 模拟总量 - 基于当前束流分布
            xx, yy = np.meshgrid(self.grid, self.grid, indexing='ij')
            beam_full = beam_matrix * self.max_val
            cell_area = (np.diff(self.grid)[0] ** 2)  # 每个网格单元的面积
            
            # 只计算优化区域内的体積
            mask = (self.distance_from_center <= self.opt_radius)
            valid_beam = beam_full * mask
            sim_total = np.sum(valid_beam) * cell_area
            
            # 相对误差
            if exp_total > 1e-6:
                return (sim_total - exp_total) / exp_total
            else:
                return sim_total  # 避免除以零
        except:
            return 0.0

    def optimize_stage(self, current_matrix, stage_idx, max_iterations):
        """执行单个高精度阶段优化 - 添加单峰性约束"""
        self.current_stage_idx = stage_idx
        stage_name = f"阶段{stage_idx+1}/{self.num_stages}"
        self.log(f"\n=== 优化 {stage_name} === (点数:{np.sum(self.stage_masks[stage_idx]):d})")
        
        # 获取当前最佳轮廓
        try:
            sim_x = self.simulate_etching(current_matrix, axis='x')
            sim_y = self.simulate_etching(current_matrix, axis='y')
        except Exception as e:
            self.log(f"模拟刻蚀失败: {str(e)}")
            sim_x = np.zeros_like(self.grid)
            sim_y = np.zeros_like(self.grid)
        
        # 计算误差
        abs_error, rel_err_x = self.calculate_error(sim_x, sim_y)
        etch_vol_err = self.validate_etch_depth(current_matrix)
        uni_error = self.validate_unimodality(current_matrix)
        
        # 带单峰性约束的综合误差
        total_error = abs_error + self.uni_penalty * uni_error
        
        # 记录当前状态
        self.history["abs_error"].append(abs_error)
        self.history["rel_err_x"].append(rel_err_x)
        self.history["etch_vol_error"].append(etch_vol_err)
        self.history["unimodal_error"].append(uni_error)
        
        # 初始化最佳状态
        best_total_error = total_error
        best_matrix = current_matrix.copy()
        stagnation_count = 0
        
        # 冻结前阶段区域
        if stage_idx > 0:
            frozen_mask = np.zeros_like(self.optimization_mask, dtype=bool)
            for i in range(stage_idx):
                if i < len(self.stage_masks):
                    frozen_mask = frozen_mask | self.stage_masks[i]
                
            # 创建初始矩阵冻结区域
            current_matrix_frozen = best_matrix.copy()
        else:
            current_matrix_frozen = best_matrix
            frozen_mask = None
        
        # 迭代优化
        for iteration in range(1, max_iterations + 1):
            # 显示进度条
            self.log_progress(iteration, max_iterations, stage_idx)
            
            # 动态变异幅度 (随迭代递减)
            mag_start = 0.15 if stage_idx < 5 else 0.08
            magnitude = max(0.01, mag_start * (1.0 - iteration/(max_iterations+1)))
            
            # 创建变异候选 (创建多个候选方案)
            candidates = []
            for i in range(self.max_mutations):
                candidate = self.mutate_beam(current_matrix_frozen, magnitude)
                
                # 保护已优化区域
                if frozen_mask is not None:
                    candidate[frozen_mask] = best_matrix[frozen_mask]
                
                candidates.append(candidate)
            
            # 评估所有候选方案
            best_candidate_error = float('inf')
            best_candidate = None
            
            for candidate in candidates:
                try:
                    cand_sim_x = self.simulate_etching(candidate, axis='x')
                    cand_sim_y = self.simulate_etching(candidate, axis='y')
                    cand_abs_error, cand_rel_err_x = self.calculate_error(cand_sim_x, cand_sim_y)
                    cand_uni_error = self.validate_unimodality(candidate)
                    cand_total_error = cand_abs_error + self.uni_penalty * cand_uni_error
                    
                    # 记录最佳候选方案
                    if cand_total_error < best_candidate_error:
                        best_candidate_error = cand_total_error
                        best_candidate = candidate
                except:
                    continue
                
            # 检查改进情况
            improvement = best_total_error - best_candidate_error
            
            if best_candidate_error < best_total_error:
                improvement_rate = -100 * improvement / best_total_error
                self.log(f"[{stage_name} I{iteration}] 改进: Δ={-improvement:.4f} ({improvement_rate:.1f}%)")
                self.log(f"      新误差={best_candidate_error:.6f} (刻蚀={cand_abs_error:.4f}, 单峰={cand_uni_error:.4f})")
                
                best_total_error = best_candidate_error
                best_matrix = best_candidate.copy()
                stagnation_count = 0
                
                # 更新历史记录
                if "iteration" in self.history:
                    iter_num = self.history["iteration"][-1] + 1
                else:
                    self.history["iteration"] = [0]
                    iter_num = 1
                
                self.history["iteration"].append(iter_num)
                self.optimized_beam = best_matrix
            else:
                stagnation_count += 1
                if iteration % 3 == 0:  # 减少日志输出频率
                    self.log(f"[{stage_name} I{iteration}] 无改进 (当前:{best_total_error:.6f}, 候选:{best_candidate_error:.6f})")
            
            # 提前终止条件
            if stagnation_count >= 5 or best_total_error < 0.005:
                if stagnation_count >= 5:
                    self.log(f"提前终止 {stage_name} (停滞次数:{stagnation_count})")
                elif best_total_error < 0.005:
                    self.log(f"提前终止 {stage_name} (已达到目标误差<0.005)")
                break
        
        return best_matrix

    def run_optimization(self):
        """运行高精度16阶段优化过程"""
        # 初始评估
        try:
            sim_x_init = self.simulate_etching(self.optimized_beam, 'x')
            sim_y_init = self.simulate_etching(self.optimized_beam, 'y')
            abs_error0, rel_err_x0 = self.calculate_error(sim_x_init, sim_y_init)
            uni_error0 = self.validate_unimodality(self.optimized_beam)
        except Exception as e:
            self.log(f"初始评估失败: {str(e)}")
            abs_error0, rel_err_x0, uni_error0 = 0.5, 50.0, 1.0
            
        etch_vol_err0 = self.validate_etch_depth(self.optimized_beam)
        
        # 历史记录初始化
        self.history = {
            "iteration": [0],
            "abs_error": [abs_error0],
            "rel_err_x": [rel_err_x0],
            "max_val": [self.max_val],
            "etch_vol_error": [etch_vol_err0],
            "unimodal_error": [uni_error0]
        }
        self.log(f"初始误差: 绝对误差={abs_error0:.6f}, 单峰性误差={uni_error0:.6f}")
        
        start_time = time.time()
        
        # 阶段迭代设置 (增加早期阶段迭代次数)
        stage_iterations = [10, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 3, 3, 3]
        
        # 执行16阶段优化
        for stage_idx in range(self.num_stages):
            if stage_idx < len(stage_iterations):
                max_iter = stage_iterations[stage_idx] 
            else:
                max_iter = 3
                
            self.log(f"\n开始优化阶段 {stage_idx+1}/{self.num_stages} (最大迭代: {max_iter})")
            self.optimized_beam = self.optimize_stage(
                self.optimized_beam, 
                stage_idx, 
                max_iter
            )
            self.stage_ends.append(len(self.history["iteration"]) - 1)
        
        # 最终优化处理
        self.log("\n=== 最终优化处理 ===")
        
        # 最终单峰性约束
        self.optimized_beam = self.enforce_strict_unimodality(self.optimized_beam)
        self.log("最终单峰性约束完成")
        
        # 最终高斯平滑
        self.optimized_beam = gaussian_filter(self.optimized_beam, sigma=0.6)
        self.log("高斯平滑 (σ=0.6mm)完成")
        
        # 确保中心点仍然是最大值
        center_i, center_j = self.find_center(self.optimized_beam)
        center_val = np.max(self.optimized_beam)
        self.optimized_beam[center_i, center_j] = center_val
        
        # 降采样到原始1mm精度
        highres_beam = self.optimized_beam * self.max_val
        optimized_beam_lowres = self.downsample_to_lowres(highres_beam)
        
        # 结果保存
        np.savetxt("optimized_beam_highres.csv", highres_beam, delimiter=",")
        np.savetxt("optimized_beam_distribution.csv", optimized_beam_lowres, delimiter=",")
        self.log("优化结果已保存: optimized_beam_highres.csv 和 optimized_beam_distribution.csv")
        
        # 最终评估
        final_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        final_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        
        try:
            final_abs_error, final_rel_err_x = self.calculate_error(final_sim_x, final_sim_y)
            final_etch_error = self.validate_etch_depth(self.optimized_beam)
            final_uni_error = self.validate_unimodality(self.optimized_beam)
        except Exception as e:
            self.log(f"最终评估失败: {str(e)}")
            final_abs_error, final_rel_err_x, final_etch_error, final_uni_error = 1.0, 100.0, 1.0, 1.0
        
        # 性能统计
        elapsed_time = time.time() - start_time
        iter_count = self.history["iteration"][-1]
        
        self.log(f"\n优化完成! 总迭代次数: {iter_count}")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        self.log(f"最终绝对误差: {final_abs_error:.6f} (初始={abs_error0:.6f})")
        self.log(f"最终单峰性误差: {final_uni_error:.6f} (初始={uni_error0:.6f})")
        self.log(f"X截面相对误差: {final_rel_err_x:.2f}%")
        self.log(f"最终刻蚀体积误差: {final_etch_error*100:.2f}%")
        
        # 记录最终单峰性误差
        self.history["unimodal_error"].append(final_uni_error)
        
        # 结果可视化
        try:
            self.visualize_results()
        except Exception as e:
            self.log(f"结果可视化失败: {str(e)}")
        
        return optimized_beam_lowres, final_abs_error

    def downsample_to_lowres(self, highres_beam):
        """将高精度优化结果降采样到1mm精度"""
        interpolator = RegularGridInterpolator(
            (self.highres_grid, self.highres_grid),
            highres_beam,
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
        
        # 生成目标网格点
        xx_low, yy_low = np.meshgrid(self.lowres_grid, self.lowres_grid, indexing="ij")
        points_lowres = np.column_stack((xx_low.ravel(), yy_low.ravel()))
        
        # 插值采样
        lowres_values = interpolator(points_lowres)
        lowres_beam = lowres_values.reshape(self.lowres_points, self.lowres_points)
        
        return lowres_beam

    def visualize_results(self):
        """直观显示优化结果（X/Y分离显示）- 添加单峰性误差显示"""
        try:
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle("离子束刻蚀效率优化结果 (高精度版本)", fontsize=18, y=0.98)
            
            # 创建网格布局 (3行2列)
            gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
            
            # 1. 初始束流分布
            ax1 = plt.subplot(gs[0, 0])
            if hasattr(self, 'initial_beam'):
                init_norm = self.initial_beam / self.max_val
                im1 = ax1.imshow(init_norm, cmap='viridis', 
                                extent=[min(self.grid), max(self.grid), min(self.grid), max(self.grid)],
                                aspect='auto', origin='lower')
                ax1.set_title("初始束流分布", fontsize=14)
                ax1.set_xlabel("X (mm)")
                ax1.set_ylabel("Y (mm)")
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # 2. 优化后束流分布
            ax2 = plt.subplot(gs[0, 1])
            if hasattr(self, 'optimized_beam'):
                im2 = ax2.imshow(self.optimized_beam, cmap='viridis', 
                                extent=[min(self.grid), max(self.grid), min(self.grid), max(self.grid)],
                                aspect='auto', origin='lower')
                ax2.set_title("优化后束流分布", fontsize=14)
                ax2.set_xlabel("X (mm)")
                ax2.set_ylabel("Y (mm)")
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # 3. 误差历史曲线
            ax3 = plt.subplot(gs[1, 0])
            if "iteration" in self.history and "abs_error" in self.history:
                iterations = self.history["iteration"]
                # 创建第二个Y轴
                ax3b = ax3.twinx()
                
                # 绝对误差曲线
                ax3.plot(iterations, self.history["abs_error"][:len(iterations)], 
                        'b-', linewidth=2, label='绝对误差')
                
                # 单峰性误差曲线（对数坐标）
                uni_errors = self.history["unimodal_error"][:len(iterations)]
                ax3b.semilogy(iterations, uni_errors, 'r--', linewidth=2, label='单峰性误差')
                
                # 标记阶段结束点
                if self.stage_ends:
                    for i, stage_end in enumerate(self.stage_ends):
                        if stage_end < len(iterations):
                            ax3.axvline(x=iterations[stage_end], color='gray', linestyle='--', alpha=0.7)
                            ax3.text(iterations[stage_end], np.max(self.history["abs_error"][:len(iterations)])*0.95, 
                                    f'阶段{i+1}', rotation=90, fontsize=8, alpha=0.7)
                
                ax3.set_xlabel("迭代次数")
                ax3.set_ylabel("绝对误差", color='b')
                ax3b.set_ylabel("单峰性误差 (log)", color='r')
                ax3.set_title("优化过程误差变化", fontsize=14)
                ax3.grid(True, linestyle='--', alpha=0.6)
                
                # 合并图例
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3b.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # 4. X方向截面拟合结果
            ax4 = plt.subplot(gs[1, 1])
            if hasattr(self, 'x_data'):
                try:
                    sim_x = self.simulate_etching(self.optimized_beam, 'x')
                    exp_x_max = np.max(self.x_data[:, 1])
                    exp_x_norm = self.x_data[:, 1] / exp_x_max
                    
                    # 找出有效数据范围
                    valid_mask = (self.x_data[:, 0] >= min(self.grid)) & (self.x_data[:, 0] <= max(self.grid))
                    
                    # 绘制X方向数据
                    ax4.plot(self.x_data[valid_mask, 0], exp_x_norm[valid_mask], 
                            'bo', alpha=0.5, markersize=5, label='实验数据')
                    ax4.plot(self.grid, sim_x / np.max(sim_x), 'r-', linewidth=2, label='优化拟合')
                    
                    # 添加误差统计信息
                    _, rel_err_x = self.calculate_error(sim_x, np.zeros_like(sim_x))
                    ax4.text(0.05, 0.9, f'相对误差: {rel_err_x:.2f}%', 
                            transform=ax4.transAxes, fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.8))
                    
                    ax4.set_xlabel("X位置 (mm)")
                    ax4.set_ylabel("归一化刻蚀深度")
                    ax4.set_title("X轴方向截面拟合 (束流Y向扫描)", fontsize=14)
                    ax4.grid(True, linestyle='--', alpha=0.6)
                    ax4.legend(loc='best', frameon=True, shadow=True)
                except Exception as e:
                    self.log(f"X方向数据绘图失败: {str(e)}")
            
            # 5. Y方向截面拟合结果
            ax5 = plt.subplot(gs[2, 0])
            if hasattr(self, 'y_data'):
                try:
                    sim_y = self.simulate_etching(self.optimized_beam, 'y')
                    exp_y_max = np.max(self.y_data[:, 1])
                    exp_y_norm = self.y_data[:, 1] / exp_y_max
                    
                    # 找出有效数据范围
                    valid_mask = (self.y_data[:, 0] >= min(self.grid)) & (self.y_data[:, 0] <= max(self.grid))
                    
                    ax5.plot(self.y_data[valid_mask, 0], exp_y_norm[valid_mask], 
                            'g^', alpha=0.5, markersize=5, label='实验数据')
                    ax5.plot(self.grid, sim_y / np.max(sim_y), 'm-', linewidth=2, label='优化拟合')
                    
                    ax5.set_xlabel("Y位置 (mm)")
                    ax5.set_ylabel("归一化刻蚀深度")
                    ax5.set_title("Y轴方向截面拟合 (束流X向扫描)", fontsize=14)
                    ax5.grid(True, linestyle='--', alpha=0.6)
                    ax5.legend(loc='best', frameon=True, shadow=True)
                except Exception as e:
                    self.log(f"Y方向数据绘图失败: {str(e)}")
            
            # 6. 误差与体积统计
            ax6 = plt.subplot(gs[2, 1])
            if self.history and "abs_error" in self.history and "etch_vol_error" in self.history:
                # 刻蚀体积误差
                vol_errors = np.array(self.history["etch_vol_error"][:len(self.history["iteration"])]) * 100
                
                # 只绘制有意义的点
                valid_ids = []
                for i, val in enumerate(vol_errors):
                    if not np.isnan(val) and abs(val) < 1e6:
                        valid_ids.append(i)
                
                if valid_ids:
                    valid_iter = [self.history["iteration"][i] for i in valid_ids]
                    valid_vol = [vol_errors[i] for i in valid_ids]
                    
                    ax6.plot(valid_iter, valid_vol, 'go-', linewidth=2, label='刻蚀体积误差 (%)')
                
                ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax6.set_xlabel("迭代次数")
                ax6.set_ylabel("刻蚀体积误差 (%)")
                ax6.set_title("刻蚀体积一致性", fontsize=14)
                ax6.grid(True, linestyle='--', alpha=0.6)
                ax6.legend(loc='best')
            
            # 总体统计信息
            if self.history and "abs_error" in self.history and "unimodal_error" in self.history:
                error_info = (
                    f"最终绝对误差: {self.history['abs_error'][-1]:.6f}\n"
                    f"单峰性误差: {self.history['unimodal_error'][-1]:.6f}\n"
                    f"刻蚀体积误差: {self.history['etch_vol_error'][-1]*100:.2f}%"
                )
                plt.figtext(0.85, 0.05, error_info, 
                            fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(hspace=0.3, wspace=0.25)
            
            # 保存结果
            plt.savefig("beam_optimization_highres_results.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.log("结果可视化已保存: beam_optimization_highres_results.png")
        except Exception as e:
            self.log(f"可视化失败: {str(e)}")
            self.log(traceback.format_exc())

    def finalize(self):
        """结束优化并关闭日志"""
        self.log("\n优化完成!")
        self.log("结果文件:")
        self.log(f"  - optimized_beam_highres.csv (高精度优化结果)")
        self.log(f"  - optimized_beam_distribution.csv (标准分辨率输出)")
        self.log(f"  - beam_optimization_highres_results.png (高精度可视化报告)")
        self.log(f"  - beam_optimization_log.txt (详细日志)")
        
        # 添加单峰性误差总结
        if self.history["unimodal_error"]:
            initial_uni = self.history["unimodal_error"][0]
            final_uni = self.history["unimodal_error"][-1]
            if initial_uni > 1e-6:
                uni_improvement = (initial_uni - final_uni) / initial_uni * 100
            else:
                uni_improvement = -100 * final_uni if final_uni > 0 else 0
            self.log(f"单峰性误差改善: {initial_uni:.4f} → {final_uni:.4f} (提升{uni_improvement:.1f}%)")
        
        if self.log_file:
            self.log_file.close()

# ================== 主程序 ==================
def main():
    # 输入文件路径 - 修正为实际文件名
    input_files = {
        "beam_traced_x_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",
        "beam_traced_y_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 80)
    print("高级离子束刻蚀效率优化引擎 - 高精度版本 (0.1mm网格)".center(80))
    print("改进点: 三次样条插值 | 辛普森积分 | 16阶段优化 | 射线约束".center(80))
    print("=" * 30)
    print("输入文件:")
    for name, path in input_files.items():
        print(f"  - {name:25}: {path}")
    print("=" * 80)
    
    # 文件检查
    missing_files = [name for name, path in input_files.items() if not os.path.exists(path)]
    
    if missing_files:
        print("警告: 以下文件不存在:")
        for name in missing_files:
            print(f"  - {name}")
        
        # 询问是否继续
        proceed = input("部分文件缺失，是否继续? (y/n): ").strip().lower()
        if proceed != 'y':
            print("程序终止")
            sys.exit(1)
    
    try:
        # 使用新精度参数初始化
        optimizer = BeamEfficiencyOptimizer(
            beam_traced_x_axis=input_files["beam_traced_x_axis"],
            beam_traced_y_axis=input_files["beam_traced_y_axis"],
            initial_guess_path=input_files["initial_beam"],
            grid_bound=15.0
        )
        
        # 执行优化
        optimized_beam, error = optimizer.run_optimization()
        
        # 生成报告
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("高精度优化成功完成!".center(80))
        print(f"最终绝对误差: {error:.6f}")
        
        # 输出单峰性误差结果
        if hasattr(optimizer, 'history') and "unimodal_error" in optimizer.history:
            uni_error = optimizer.history["unimodal_error"][-1]
            print(f"单峰性误差: {uni_error:.6f}")
        
        # 输出刻蚀体积误差
        if hasattr(optimizer, 'history') and "etch_vol_error" in optimizer.history:
            vol_err = optimizer.history["etch_vol_error"][-1] * 100
            print(f"刻蚀体积误差: {vol_err:.2f}%")
            
        print("结果文件:")
        print(f"  - optimized_beam_highres.csv")
        print(f"  - optimized_beam_distribution.csv")
        print(f"  - beam_optimization_highres_results.png")
        print(f"  - beam_optimization_log.txt")
        print("=" * 80)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
