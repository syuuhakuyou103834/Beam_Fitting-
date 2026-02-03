import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from matplotlib import gridspec
import traceback

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
                 grid_bound=15.0, highres_points=121, lowres_points=31):
        """
        高级束流分布优化器 - 修复初始化顺序问题
        """
        # 创建日志文件
        self.log_file = open("beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("高级离子束刻蚀效率优化引擎")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"多精度网格系统: 优化精度=0.25mm ({highres_points}x{highres_points})")
        self.log("输入文件:")
        self.log(f"  - X截面: {beam_traced_x_axis}")
        self.log(f"  - Y截面: {beam_traced_y_axis}")
        self.log(f"  - 初始束流: {initial_guess_path}")
        self.log("=" * 30)
        
        # 优化参数（在网格创建之前定义）
        self.opt_radius = 15.0     # 优化半径 (mm)
        self.scan_velocity = 30.0  # 扫描速度 (mm/s)
        self.drift_sigma = 1.8     # 漂移校正标准差 (mm)
        self.smooth_sigma = 0.8    # 高斯平滑参数
        self.max_mutations = 4     # 变异候选数
        self.uni_penalty = 0.5     # 单峰性约束的惩罚因子
        
        # 网格系统参数
        self.grid_bound = grid_bound
        self.highres_points = highres_points
        self.lowres_points = lowres_points
        
        # 创建高精度网格 (0.25mm间隔)
        self.highres_grid = np.linspace(-grid_bound, grid_bound, highres_points)
        self.highres_spacing = 2 * grid_bound / (highres_points - 1)
        
        # 创建原始精度网格 (1mm间隔)
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
        
        # 创建11个阶段的高精度点集掩膜
        self.num_stages = 11
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

    def find_center(self, beam_matrix):
        """找到束流中心点位置"""
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        return center_i, center_j

    def load_and_preprocess_data(self, file_path):
        """加载实验数据并进行平移预处理"""
        self.log(f"加载实验数据: {file_path}")
        
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
        
        # 创建插值器
        try:
            interpolator = interp1d(
                data[:, 0], data[:, 1], 
                kind='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
        except:
            # 如果插值失败，使用最近点
            interpolator = interp1d(
                data[:, 0], data[:, 1], 
                kind='nearest', 
                bounds_error=False, 
                fill_value=0.0
            )
        
        # 创建新网格数据 (0.25mm精度)
        new_values = interpolator(self.grid)
        
        # 确保非负值
        new_values[new_values < 0] = 0
        
        # 创建新数据集
        new_data = np.column_stack((self.grid, new_values))
        
        # 找到新的峰值位置
        new_peak_idx = np.argmax(new_values)
        self.log(f"  平移后峰值位置: x={self.grid[new_peak_idx]:.2f}mm (差值: {abs(self.grid[new_peak_idx]):.4f}mm)")
        
        return new_data
    
    def create_optimization_mask(self):
        """创建高精度优化区域掩膜（使用已定义的opt_radius）"""
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.distance_from_center = np.sqrt(xx**2 + yy**2)
        self.optimization_mask = (self.distance_from_center <= self.opt_radius)
        self.log(f"高精度优化区域包含 {np.sum(self.optimization_mask)} 个点 (半径={self.opt_radius}mm)")

    def load_initial_beam(self, file_path):
        """加载、插值并校准初始束流分布"""
        try:
            # 1. 加载原始1mm精度数据
            lowres_beam = np.genfromtxt(file_path, delimiter=",")
            if lowres_beam.shape != (self.lowres_points, self.lowres_points):
                if lowres_beam.size == self.lowres_points * self.lowres_points:
                    lowres_beam = lowres_beam.reshape((self.lowres_points, self.lowres_points))
                else:
                    self.log(f"警告: 初始束流尺寸不兼容 ({lowres_beam.shape} vs {self.lowres_points}x{self.lowres_points})")
                    raise ValueError("初始束流尺寸不兼容")
            
            # 2. 插值到0.25mm高精度网格
            interpolator = RegularGridInterpolator(
                (self.lowres_grid, self.lowres_grid),
                lowres_beam,
                method='cubic',
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
        """创建11个阶段的高精度点集掩膜"""
        self.stage_masks = []
        total_mask = np.zeros_like(self.optimization_mask, dtype=bool)
        
        # 获取网格索引
        xx, yy = np.meshgrid(range(self.highres_points), range(self.highres_points), indexing='ij')
        
        # 第一阶段 (集合1): x=0 或 y=0 的点
        stage0_mask = ((np.abs(self.grid[xx]) < 1e-5) | (np.abs(self.grid[yy]) < 1e-5))
        self.stage_masks.append(stage0_mask)
        total_mask = total_mask | stage0_mask
        
        # 其他阶段 (1-9mm环)
        for k in range(1, 10):
            k_value = float(k)
            threshold = self.grid_spacing
            stage_mask = (
                (np.abs(np.abs(self.grid[xx]) - k_value) < threshold) | 
                (np.abs(np.abs(self.grid[yy]) - k_value) < threshold)
            )
            stage_mask = stage_mask & ~total_mask
            self.stage_masks.append(stage_mask)
            total_mask = total_mask | stage_mask
        
        # 第十一阶段 (集合11): 剩余所有点
        stage10_mask = ~total_mask
        self.stage_masks.append(stage10_mask)
        
        # 日志记录每个阶段的点数
        for i, mask in enumerate(self.stage_masks):
            self.log(f"集合{i+1} (阶段{i}): {np.sum(mask)}点")
        
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
        num_rays = 36
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        max_radius = self.opt_radius
        
        for angle in angles:
            # 构建射线路径 (0.25mm步长)
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
        # 确定中心点
        center_i, center_j = self.center_i, self.center_j
        center_val = beam_matrix[center_i, center_j] if beam_matrix[center_i, center_j] > 0 else 0.1
        
        # 创建新束流副本
        new_beam = beam_matrix.copy()
        
        # 在36个射线方向应用严格递减约束
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        max_radius = self.opt_radius
        
        for angle in angles:
            # 构建射线路径 (0.25mm步长)
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
                
                # 目标值应小于前一个值
                threshold = 0.01  # 添加小阈值避免数值问题
                decay_factor = np.exp(-(dist - prev_dist) / 5.0)
                max_allowed = prev_val * decay_factor + threshold
                
                if current_val > max_allowed:
                    new_beam[x_idx, y_idx] = max_allowed
                
                prev_val = new_beam[x_idx, y_idx]
                prev_dist = dist
        
        # 应用高斯平滑
        smoothed_beam = gaussian_filter(new_beam, sigma=self.smooth_sigma)
        
        # 确保中心点仍然是最大值
        if strict:
            smoothed_beam[center_i, center_j] = center_val
        
        # 维持非负性
        smoothed_beam = np.maximum(smoothed_beam, 0)
        
        return smoothed_beam

    def validate_unimodality(self, beam_matrix):
        """
        验证束流分布的单峰性
        从中心点出发沿所有方向检查是否单调递减
        返回单峰性误差（违反单调性的次数和幅度）
        """
        center_i, center_j = self.center_i, self.center_j
        center_val = beam_matrix[center_i, center_j]
        error_magnitude = 0.0
        
        # 在所有射线方向检查
        num_rays = 24
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        max_radius = self.opt_radius
        
        for angle in angles:
            # 构建射线路径 (0.25mm步长)
            distances = np.arange(0.25, max_radius, 0.25)
            prev_val = center_val
            prev_point = (center_i, center_j)
            
            for dist in distances:
                x_pos = dist * np.cos(angle)
                y_pos = dist * np.sin(angle)
                
                # 找到最近的网格点
                x_idx = np.argmin(np.abs(self.grid - x_pos))
                y_idx = np.argmin(np.abs(self.grid - y_pos))
                
                if not (0 <= x_idx < self.highres_points and 0 <= y_idx < self.highres_points):
                    continue
                    
                current_val = beam_matrix[x_idx, y_idx]
                
                # 当前点值应小于前一外层点的值
                violation = max(0, current_val - prev_val)
                error_magnitude += violation
                
                prev_val = current_val
                prev_point = (x_idx, y_idx)
        
        return error_magnitude

    def simulate_etching(self, beam_matrix, axis='x'):
        """
        模拟束沿垂直轴移动时的轮廓
        axis='x': 束沿Y移动，产生X方向轮廓
        axis='y': 束沿X移动，产生Y方向轮廓
        """
        # 修复：直接使用束流矩阵计算，避免积分错误
        
        # 选择路径方式
        if axis == 'x':  # 扫描Y方向，测量X截面
            # 沿Y方向扫描
            profile = np.zeros_like(self.grid)
            
            # 计算漂移卷积核
            kernel_radius = int(3 * self.drift_sigma / self.grid_spacing)
            kernel_range = np.linspace(-kernel_radius, kernel_radius, num=2*kernel_radius+1) * self.grid_spacing
            kernel = np.exp(-0.5 * (kernel_range/self.drift_sigma)**2)
            kernel /= np.sum(kernel)  # 归一化
            
            # 逐点计算
            for i in range(len(self.grid)):
                xi = self.grid[i]
                
                # 沿Y方向的位置
                y_positions = self.grid
                
                # 计算沿路径的蚀刻速率
                rates = np.zeros_like(y_positions)
                for j, yj in enumerate(y_positions):
                    rates[j] = beam_matrix[i, j] * self.max_val
                
                # 添加漂移扩散效应
                if self.drift_sigma > 0 and len(rates) > len(kernel):
                    try:
                        convolved_rates = np.convolve(rates, kernel, mode='same')
                    except Exception as e:
                        self.log(f"卷积失败: {str(e)}")
                        convolved_rates = rates
                else:
                    convolved_rates = rates
                
                # 积分得到深度 (考虑扫描速度)
                profile[i] = trapezoid(convolved_rates, y_positions)
            
        elif axis == 'y':  # 扫描X方向，测量Y截面
            # 沿X方向扫描
            profile = np.zeros_like(self.grid)
            
            # 计算漂移卷积核
            kernel_radius = int(3 * self.drift_sigma / self.grid_spacing)
            kernel_range = np.linspace(-kernel_radius, kernel_radius, num=2*kernel_radius+1) * self.grid_spacing
            kernel = np.exp(-0.5 * (kernel_range/self.drift_sigma)**2)
            kernel /= np.sum(kernel)  # 归一化
            
            # 逐点计算
            for j in range(len(self.grid)):
                yj = self.grid[j]
                
                # 沿X方向的位置
                x_positions = self.grid
                
                # 计算沿路径的蚀刻速率
                rates = np.zeros_like(x_positions)
                for i, xi in enumerate(x_positions):
                    rates[i] = beam_matrix[i, j] * self.max_val
                
                # 添加漂移扩散效应
                if self.drift_sigma > 0 and len(rates) > len(kernel):
                    try:
                        convolved_rates = np.convolve(rates, kernel, mode='same')
                    except Exception as e:
                        self.log(f"卷积失败: {str(e)}")
                        convolved_rates = rates
                else:
                    convolved_rates = rates
                
                # 积分得到深度 (考虑扫描速度)
                profile[j] = trapezoid(convolved_rates, x_positions)
                
        else:
            profile = np.zeros_like(self.grid)
        
        # 归一化
        max_profile = np.max(profile)
        if max_profile < 1e-10:
            self.log("警告: 模拟刻蚀轮廓全零!")
            max_profile = 1.0
        
        return profile / max_profile if max_profile > 0 else profile

    def calculate_error(self, sim_x, sim_y):
        """计算高精度模拟结果与实验数据的误差"""
        # X截面误差
        try:
            exp_x = self.x_data
            exp_x_pos = exp_x[:, 0]
            exp_x_val = exp_x[:, 1]
            
            # 避免除以零
            exp_x_max = np.max(exp_x_val) if np.max(exp_x_val) > 0 else 1.0
            exp_peak_val_x = exp_x_max  # 记录峰值强度
            
            # 归一化
            exp_x_norm = exp_x_val / exp_x_max if exp_x_max > 0 else exp_x_val
            sim_x_norm = sim_x / np.max(sim_x) if np.max(sim_x) > 0 else sim_x
            
            # 插值到相同网格
            sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_norm)
            
            # 计算绝对误差和相对误差
            abs_error_x = np.mean(np.abs(sim_x_interp - exp_x_norm))
            non_zero_x = exp_x_norm > np.max(exp_x_norm)*0.05
            if np.any(non_zero_x):
                rel_err_x = np.mean(np.abs(sim_x_interp[non_zero_x] - exp_x_norm[non_zero_x]) / 
                                   (exp_x_norm[non_zero_x] + 1e-10)) * 100
            else:
                rel_err_x = np.mean(np.abs(sim_x_interp)) * 100
        except Exception as e:
            self.log(f"X方向误差计算失败: {str(e)}")
            abs_error_x = 1.0
            rel_err_x = 100.0
            exp_peak_val_x = 1.0
        
        # Y截面误差
        try:
            exp_y = self.y_data
            exp_y_pos = exp_y[:, 0]
            exp_y_val = exp_y[:, 1]
            
            exp_y_max = np.max(exp_y_val) if np.max(exp_y_val) > 0 else 1.0
            exp_peak_val_y = exp_y_max  # 记录峰值强度
            
            exp_y_norm = exp_y_val / exp_y_max if exp_y_max > 0 else exp_y_val
            sim_y_norm = sim_y / np.max(sim_y) if np.max(sim_y) > 0 else sim_y
            
            sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_norm)
            
            abs_error_y = np.mean(np.abs(sim_y_interp - exp_y_norm))
            non_zero_y = exp_y_norm > np.max(exp_y_norm)*0.05
            if np.any(non_zero_y):
                rel_err_y = np.mean(np.abs(sim_y_interp[non_zero_y] - exp_y_norm[non_zero_y]) / 
                                   (exp_y_norm[non_zero_y] + 1e-10)) * 100
            else:
                rel_err_y = np.mean(np.abs(sim_y_interp)) * 100
        except Exception as e:
            self.log(f"Y方向误差计算失败: {str(e)}")
            abs_error_y = 1.0
            rel_err_y = 100.0
            exp_peak_val_y = 1.0
        
        # 综合绝对误差 (X方向60%权重)
        weights = {'x': 0.6, 'y': 0.4}
        combined_abs_error = weights['x'] * abs_error_x + weights['y'] * abs_error_y
        
        # 计算实验与模拟峰值比
        sim_x_max = np.max(sim_x) if np.max(sim_x) > 1e-10 else np.max(self.x_data[:, 1])
        x_peak_ratio = np.abs(sim_x_max - exp_peak_val_x) / exp_peak_val_x if exp_peak_val_x > 0 else 0
        y_peak_ratio = np.abs(np.max(sim_y) - exp_peak_val_y) / exp_peak_val_y if exp_peak_val_y > 0 else 0
        
        # 峰值比例惩罚项 (避免改变峰值位置)
        peak_penalty = (weights['x'] * x_peak_ratio**2 + weights['y'] * y_peak_ratio**2) * 0.1
        
        # 总误差加入峰值惩罚
        combined_abs_error += peak_penalty
        
        return combined_abs_error, rel_err_x

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
            if hasattr(self, 'distance_from_center'):
                dist_to_center = self.distance_from_center[i, j]
                dist_factor = max(0.3, np.exp(-dist_to_center / 10.0))
            else:
                dist_factor = 1.0  # 默认值
            
            # 基础变异幅度 (随阶段递减)
            effective_mag = magnitude * dist_factor * (0.8 + 0.4 * np.random.rand())
            
            # 随机变异方向 (增加/减少) - 避免过大变异
            direction = np.random.choice([-1, 1])
            mutation = direction * effective_mag
            
            # 应用变异 (带边界检查)
            new_val = beam_matrix[i, j] + mutation
            new_val = max(0, min(1.0, new_val))  # 边界约束
            new_beam[i, j] = new_val
        
        # 应用凸性约束
        if hasattr(self, 'enforce_convexity'):
            return self.enforce_convexity(new_beam, strict=False)
        
        return new_beam

    def validate_etch_depth(self, beam_matrix):
        """验证高精度积分总量一致性 (简化)"""
        try:
            # 从实验数据估算总刻蚀体积
            exp_x_val = self.x_data[:, 1]
            exp_x_pos = self.x_data[:, 0]
            
            # 插值到统一网格
            exp_interp = np.interp(self.grid, exp_x_pos, exp_x_val)
            
            # 计算X方向积分
            x_integral = trapezoid(exp_interp, self.grid)
            
            # Y方向范围 (假设为30mm)
            y_range = 30.0
            exp_total = x_integral * y_range
            
            # 模拟总量 - 基于当前束流分布
            grid_area = self.grid_spacing ** 2
            beam_full = beam_matrix * self.max_val
            sim_total = np.sum(beam_full) * grid_area
            
            # 相对误差
            if exp_total > 0:
                rel_error = (sim_total - exp_total) / exp_total
            else:
                rel_error = 0.0
                
            return rel_error
        except:
            return 0.0

    def optimize_stage(self, current_matrix, stage_idx, max_iterations):
        """执行单个高精度阶段优化 - 添加单峰性约束"""
        self.current_stage_idx = stage_idx
        stage_name = f"阶段{stage_idx+1}/11"
        self.log(f"\n=== 优化 {stage_name} === (点数:{np.sum(self.stage_masks[stage_idx]):d})")
        
        # 获取当前最佳轮廓
        sim_x = self.simulate_etching(current_matrix, axis='x')
        sim_y = self.simulate_etching(current_matrix, axis='y')
        
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
            # 动态变异幅度 (随迭代递减)
            mag_start = 0.15 if stage_idx < 5 else 0.08
            magnitude = max(0.01, mag_start * (1.0 - iteration/(max_iterations+1)))
            
            # 创建变异候选
            candidate = self.mutate_beam(current_matrix_frozen, magnitude)
            
            # 保护已优化区域
            if frozen_mask is not None:
                candidate[frozen_mask] = best_matrix[frozen_mask]
            
            # 评估候选
            cand_sim_x = self.simulate_etching(candidate, axis='x')
            cand_sim_y = self.simulate_etching(candidate, axis='y')
            cand_abs_error, cand_rel_err_x = self.calculate_error(cand_sim_x, cand_sim_y)
            cand_uni_error = self.validate_unimodality(candidate)
            cand_total_error = cand_abs_error + self.uni_penalty * cand_uni_error
            
            # 检查改进情况
            improvement = best_total_error - cand_total_error
            
            if cand_total_error < best_total_error:
                improvement_rate = -100 * improvement / best_total_error
                self.log(f"[{stage_name} I{iteration}] 改进: Δ={-improvement:.4f} ({improvement_rate:.1f}%)")
                self.log(f"      新误差={cand_total_error:.6f} (刻蚀={cand_abs_error:.4f}, 单峰={cand_uni_error:.4f})")
                
                best_total_error = cand_total_error
                best_matrix = candidate.copy()
                sim_x, sim_y = cand_sim_x, cand_sim_y
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
                self.log(f"[{stage_name} I{iteration}] 无改进 (当前:{best_total_error:.6f}, 候选:{cand_total_error:.6f})")
            
            # 收敛检查
            if stagnation_count >= 3 or best_total_error < 0.005:
                self.log(f"提前终止 {stage_name} (停滞次数:{stagnation_count})")
                break
        
        self.log(f"阶段完成: 最终误差={best_total_error:.6f} (刻蚀={cand_abs_error:.4f}, 单峰={cand_uni_error:.4f})")
        return best_matrix

    def run_optimization(self):
        """运行高精度11阶段优化过程"""
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
        self.log(f"初始绝对误差: {abs_error0:.6f}, 单峰性误差: {uni_error0:.6f}")
        
        start_time = time.time()
        
        # 阶段迭代设置 (增加早期阶段迭代次数)
        stage_iterations = [6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 4]
        
        # 执行11阶段优化
        for stage_idx in range(self.num_stages):
            max_iter = stage_iterations[stage_idx] if stage_idx < len(stage_iterations) else 3
            self.optimized_beam = self.optimize_stage(
                self.optimized_beam, 
                stage_idx, 
                max_iter
            )
            self.stage_ends.append(len(self.history["iteration"]) - 1)
        
        # 最终单峰性约束
        self.optimized_beam = self.enforce_strict_unimodality(self.optimized_beam)
        
        # 最终高斯平滑
        self.optimized_beam = gaussian_filter(self.optimized_beam, sigma=0.7)
        
        # 确保中心点仍然是最大值
        if hasattr(self, 'center_i') and hasattr(self, 'center_j'):
            center_val = np.max(self.optimized_beam)
            self.optimized_beam[self.center_i, self.center_j] = center_val
        
        # 降采样到原始1mm精度
        highres_beam = self.optimized_beam * self.max_val
        optimized_beam_lowres = self.downsample_to_lowres(highres_beam)
        
        # 结果保存
        np.savetxt("optimized_beam_highres.csv", highres_beam, delimiter=",")
        np.savetxt("optimized_beam_distribution.csv", optimized_beam_lowres, delimiter=",")
        
        # 最终评估
        final_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        final_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        final_abs_error, final_rel_err_x = self.calculate_error(final_sim_x, final_sim_y)
        final_etch_error = self.validate_etch_depth(self.optimized_beam)
        final_uni_error = self.validate_unimodality(self.optimized_beam)
        
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
        self.visualize_results()
        
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
            # 创建更大的画布 (20x10英寸)
            fig = plt.figure(figsize=(20, 10))
            fig.suptitle("离子束刻蚀效率优化结果", fontsize=18)
            
            # 创建网格布局 (2行3列)
            gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])
            
            # 1. 初始束流分布
            ax1 = plt.subplot(gs[0, 0])
            init_norm = self.initial_beam / self.max_val
            im1 = ax1.imshow(init_norm, cmap='viridis', 
                            extent=[min(self.grid), max(self.grid), min(self.grid), max(self.grid)])
            ax1.set_title("初始束流分布", fontsize=14)
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # 2. 优化后束流分布
            ax2 = plt.subplot(gs[0, 1])
            im2 = ax2.imshow(self.optimized_beam, cmap='viridis', 
                            extent=[min(self.grid), max(self.grid), min(self.grid), max(self.grid)])
            ax2.set_title("优化后束流分布", fontsize=14)
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # 3. 截面拟合结果 - X轴方向
            ax3 = plt.subplot(gs[0, 2])
            # 4. 截面拟合结果 - Y轴方向
            ax4 = plt.subplot(gs[1, :])  # 底部占整行
            
            try:
                # 获取当前模拟结果
                sim_x = self.simulate_etching(self.optimized_beam, 'x')
                sim_y = self.simulate_etching(self.optimized_beam, 'y')
                
                # X方向拟合结果 (顶部右侧)
                ax3.set_title("X轴方向截面拟合(束流沿Y方向移动)", fontsize=14)
                exp_x_max = np.max(self.x_data[:, 1])
                sim_x_max = np.max(sim_x) if np.max(sim_x) > 0 else 1.0
                
                # 绘制X方向数据
                ax3.plot(self.x_data[:, 0], self.x_data[:, 1]/exp_x_max, 
                        'bo', alpha=0.7, markersize=6, label='实验数据')
                ax3.plot(self.grid, sim_x, 'r-', linewidth=2.5, label='优化拟合')
                
                # 添加误差统计信息
                abs_error, rel_err_x = self.calculate_error(sim_x, sim_y)
                ax3.text(0.05, 0.9, f'相对误差: {rel_err_x:.2f}%', 
                        transform=ax3.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
                
                ax3.set_xlabel("X位置 (mm)")
                ax3.set_ylabel("归一化刻蚀深度")
                ax3.grid(True, linestyle='--', alpha=0.6)
                ax3.legend(loc='best', frameon=True, shadow=True)
                
                # 设置X轴范围
                x_min, x_max = np.percentile(self.grid, [2, 98])
                ax3.set_xlim(x_min, x_max)
                
                # Y方向拟合结果 (底部整行)
                ax4.set_title("Y轴方向截面拟合(束流沿X方向移动)", fontsize=14)
                exp_y_max = np.max(self.y_data[:, 1])
                
                ax4.plot(self.y_data[:, 0], self.y_data[:, 1]/exp_y_max, 
                        'g^', alpha=0.7, markersize=6, label='实验数据')
                ax4.plot(self.grid, sim_y, 'm-', linewidth=2.5, label='优化拟合')
                
                # 添加辅助误差线 (虚线表示零误差)
                ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                # 添加误差统计信息
                ax4.text(0.05, 0.9, f'绝对误差: {abs_error:.4f}', 
                        transform=ax4.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))
                
                ax4.set_xlabel("Y位置 (mm)")
                ax4.set_ylabel("归一化刻蚀深度")
                ax4.grid(True, linestyle='--', alpha=0.6)
                ax4.legend(loc='best', frameon=True, shadow=True)
                
                # 设置Y轴范围
                y_min, y_max = np.percentile(self.grid, [2, 98])
                ax4.set_xlim(y_min, y_max)
                
                # 添加单峰性误差显示
                if self.history["unimodal_error"]:
                    final_uni_error = self.history["unimodal_error"][-1]
                    fig.text(0.75, 0.05, f'最终单峰性误差: {final_uni_error:.6f}', 
                            fontsize=12, ha='center',
                            bbox=dict(facecolor='yellow', alpha=0.4))
                    
                # 添加刻蚀体积误差
                if self.history["etch_vol_error"]:
                    vol_err = self.history["etch_vol_error"][-1] * 100
                    fig.text(0.25, 0.05, f'刻蚀体积误差: {vol_err:.2f}%', 
                            fontsize=12, ha='center',
                            bbox=dict(facecolor='lightblue', alpha=0.4))
                
            except Exception as e:
                self.log(f"截面拟合可视化失败: {str(e)}")
                self.log(traceback.format_exc())
            
            # 调整布局并保存
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig("beam_optimization_results.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.log("结果可视化已保存")
        except Exception as e:
            self.log(f"可视化完全失败: {str(e)}")
            self.log(traceback.format_exc())

    def finalize(self):
        """结束优化并关闭日志"""
        self.log("\n优化完成!")
        self.log("结果文件:")
        self.log(f"  - optimized_beam_highres.csv (高精度优化结果)")
        self.log(f"  - optimized_beam_distribution.csv (标准分辨率输出)")
        self.log(f"  - beam_optimization_results.png (可视化报告)")
        
        # 添加单峰性误差总结
        if self.history["unimodal_error"]:
            initial_uni = self.history["unimodal_error"][0]
            final_uni = self.history["unimodal_error"][-1]
            uni_improvement = (initial_uni - final_uni) / initial_uni * 100
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
    print("高级离子束刻蚀效率优化引擎 - 单峰性约束版".center(80))
    print("解决拟合曲线多波峰问题".center(80))
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
        optimizer = BeamEfficiencyOptimizer(
            beam_traced_x_axis=input_files["beam_traced_x_axis"],
            beam_traced_y_axis=input_files["beam_traced_y_axis"],
            initial_guess_path=input_files["initial_beam"],
            highres_points=121,
            lowres_points=31
        )
        
        # 执行优化
        optimized_beam, error = optimizer.run_optimization()
        
        # 生成报告
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("优化成功完成!".center(80))
        print(f"最终误差: {error:.6f}")
        
        # 输出单峰性误差结果
        if optimizer.history["unimodal_error"]:
            uni_error = optimizer.history["unimodal_error"][-1]
            print(f"单峰性误差: {uni_error:.6f}")
        
        print("结果文件:")
        print(f"  - optimized_beam_highres.csv")
        print(f"  - optimized_beam_distribution.csv")
        print(f"  - beam_optimization_results.png")
        print(f"  - beam_optimization_log.txt")
        print("=" * 80)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
