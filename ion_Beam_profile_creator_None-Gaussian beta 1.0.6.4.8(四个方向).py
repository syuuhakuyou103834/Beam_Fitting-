import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter, maximum_filter
from matplotlib import gridspec
import traceback

# ============== 严格单峰性优化版本 ==============
# 精度参数
HIGH_PRECISION = 0.1  # 高精度网格步长(单位：mm)
GRID_BOUND = 15.0     # 网格边界 (mm)
RING_STEP = 0.1       # 环状区域步长 (0.1mm)

# ============== 方向衰减参数配置 ==============
# 四个方向的衰减因子（0-1之间，值越小衰减越快）
DIRECTION_DECAY_FACTORS = {
    'x+': 0.75,  # x正方向 (从0.65提高到0.85)
    'x-': 0.75,  # x负方向 (从0.65提高到0.85)
    'y+': 0.75,  # y正方向 (从0.70提高到0.90)
    'y-': 0.75   # y负方向 (从0.70提高到0.90)
}

MIN_RADIAL_SLOPE = 0.002  # 最小径向斜率约束 (从0.008降到0.002)
SMALL_OFFSET = 0.001  # 防止除以零的小偏移量

# ============== 修复中文字体支持 ==============
def setup_plotting():
    """配置绘图环境，解决中文字体问题"""
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'],
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 300
    })
    
setup_plotting()

class DirectionalBeamOptimizer:
    def __init__(self, beam_traced_x_axis, beam_traced_y_axis, initial_guess_path):
        """
        方向性束流优化器 - 支持四个方向独立衰减控制
        """
        # 创建日志文件
        self.log_file = open("directional_beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("方向性束流优化引擎 (0.1mm 环状优化) - 修复版本")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"网格精度: {HIGH_PRECISION:.2f}mm, 环状步长: {RING_STEP:.2f}mm")
        self.log("方向衰减因子:")
        self.log(f"  - X+方向: {DIRECTION_DECAY_FACTORS['x+']:.3f}")
        self.log(f"  - X-方向: {DIRECTION_DECAY_FACTORS['x-']:.3f}")
        self.log(f"  - Y+方向: {DIRECTION_DECAY_FACTORS['y+']:.3f}")
        self.log(f"  - Y-方向: {DIRECTION_DECAY_FACTORS['y-']:.3f}")
        self.log("输入文件:")
        self.log(f"  - X截面: {beam_traced_x_axis}")
        self.log(f"  - Y截面: {beam_traced_y_axis}")
        self.log(f"  - 初始束流: {initial_guess_path}")
        self.log("=" * 60)
        
        # 计算网格点数
        self.grid_points = int(GRID_BOUND * 2 / HIGH_PRECISION) + 1
        self.grid = np.linspace(-GRID_BOUND, GRID_BOUND, self.grid_points)
        self.grid_spacing = 2 * GRID_BOUND / (self.grid_points - 1)
        
        # 优化参数
        self.opt_radius = GRID_BOUND - 1.0  # 优化半径 (mm)
        self.scan_velocity = 30.0   # 扫描速度 (mm/s)
        self.drift_sigma = 1.8      # 漂移校正标准差 (mm)
        self.smooth_sigma = 0.5     # 高斯平滑参数
        
        # 创建距离矩阵（以原点为中心）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_matrix = np.sqrt(xx**2 + yy**2)  # 距离原点距离
        self.x_matrix = xx
        self.y_matrix = yy
        
        # 加载和处理数据
        self.log("加载并预处理实验数据...")
        self.x_data = self.load_and_preprocess_data(beam_traced_x_axis)
        self.y_data = self.load_and_preprocess_data(beam_traced_y_axis)
        
        # 加载初始束流（强制中心点峰值）
        self.load_initial_beam(initial_guess_path)
        
        # 初始化优化束流（归一化）
        self.optimized_beam = self.initial_beam / (self.max_val + SMALL_OFFSET)  # 防止除以零
        self.center_idx = self.grid_points // 2
        self.center_x = self.grid[self.center_idx]
        self.center_y = self.grid[self.center_idx]
        
        # 历史记录
        self.history = {
            "iteration": [0],
            "abs_error": [],
            "unimodal_error": [],
            "peak_count": []
        }

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
            prefix = f"[阶段 {stage+1}]"
        else:
            prefix = "[优化进度]"
        
        progress = int(100 * current/total)
        bar = '=' * (progress//2)
        sys.stdout.write(f"\r{prefix} |{bar:{50}}| {progress}%")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    def load_and_preprocess_data(self, file_path):
        """加载实验数据并进行平移预处理"""
        self.log(f"加载实验数据: {file_path}")
        
        try:
            # 加载原始数据
            data = np.loadtxt(file_path, delimiter=",")
            
            # 确保数据有两列
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("数据格式不正确，应为N行2列")
                
            # 找出峰值位置并进行平移
            peak_idx = np.argmax(data[:, 1])
            peak_pos = data[peak_idx, 0]
            shift_value = -peak_pos
            
            # 应用平移
            data[:, 0] += shift_value
            
            # 使用三次样条插值到标准网格
            interpolator = interp1d(
                data[:, 0], data[:, 1], 
                kind='cubic', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            new_values = interpolator(self.grid)
            # 平滑处理
            new_values = gaussian_filter(new_values, sigma=0.8)
            # 非负约束
            new_values = np.maximum(new_values, 0)
            
            # 创建新数据集
            new_data = np.column_stack((self.grid, new_values))
            new_peak_idx = np.argmax(new_values)
            self.log(f"  平移后峰值位置: {self.grid[new_peak_idx]:.2f}mm | 强度: {new_values[new_peak_idx]:.2f}")
            
            return new_data
            
        except Exception as e:
            self.log(f"加载数据失败: {str(e)}")
            # 返回空数据
            return np.column_stack((self.grid, np.zeros_like(self.grid)))

    def load_initial_beam(self, file_path):
        """加载初始束流分布，强制中心点峰值"""
        try:
            # 加载初始束流
            initial_data = np.genfromtxt(file_path, delimiter=",")
            
            # 确保尺寸正确
            target_points = int(GRID_BOUND * 2 / 1.0) + 1  # 1mm精度的点数量
            if initial_data.shape[0] != target_points or initial_data.shape[1] != target_points:
                self.log(f"调整初始束流尺寸: {initial_data.shape} -> ({target_points},{target_points})")
                new_data = np.zeros((target_points, target_points))
                min_size = min(initial_data.shape[0], target_points)
                new_data[:min_size, :min_size] = initial_data[:min_size, :min_size]
                initial_data = new_data
            
            # 创建插值器 (1mm精度)
            lowres_grid = np.linspace(-GRID_BOUND, GRID_BOUND, target_points)
            interpolator = RegularGridInterpolator(
                (lowres_grid, lowres_grid),
                initial_data,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            # 插值到高精度网格 (0.1mm)
            xx, yy = np.meshgrid(self.grid, self.grid, indexing='ij')
            points = np.stack((xx, yy), axis=-1)
            highres_beam = interpolator(points)
            
            # 峰值校准
            exp_peak_val_x = np.max(self.x_data[:, 1])
            grid_center = self.grid_points // 2
            
            # 取中心3x3区域的平均值
            center_val = np.mean(highres_beam[grid_center-1:grid_center+2, grid_center-1:grid_center+2])
            
            if center_val > 1e-6:
                calibration_factor = (exp_peak_val_x) / (center_val + SMALL_OFFSET)  # 防止除以零
                highres_beam *= calibration_factor
            
            # 强制中心为最大值
            center_val = np.max(highres_beam) + SMALL_OFFSET
            highres_beam[grid_center, grid_center] = center_val
            
            # 应用高斯平滑
            highres_beam = gaussian_filter(highres_beam, sigma=0.5)
            
            self.initial_beam = highres_beam
            self.max_val = center_val  # 使用中心点最大值
            self.center_idx = grid_center
            
            self.log(f"加载成功: 中心值={self.max_val:.2f} nm/s, 尺寸={self.initial_beam.shape}")
            
        except Exception as e:
            self.log(f"加载初始束流失败: {str(e)}")
            # 创建高斯分布作为默认
            r = np.sqrt(self.grid[:, None]**2 + self.grid[None, :]**2)
            self.initial_beam = 100 * np.exp(-r**2 / (2*5.0**2))
            self.max_val = np.max(self.initial_beam)
            self.center_idx = self.grid_points // 2
    
    def enforce_strict_unimodality(self, beam_matrix):
        """
        强制的单峰性约束：支持四个方向独立衰减的严格单峰性
        0.1mm环状优化步骤 - 修复的约束逻辑
        """
        # 计算辐向距离矩阵
        r = self.r_matrix
        
        # 生成径向距离分段 (0.1mm步长)
        radial_bins = np.arange(0, np.max(r), RING_STEP)
        beam_new = beam_matrix.copy()
        
        # 设置中心点约束
        center_val = beam_matrix[self.center_idx, self.center_idx]
        if center_val < np.max(beam_matrix):
            beam_new[self.center_idx, self.center_idx] = np.max(beam_matrix)
        
        # 按距离环从内向外处理 (0.1mm步长)
        for r_bin in radial_bins[1:]:
            # 计算内层参考值 (内边界)
            inner_mask = (r >= r_bin - RING_STEP) & (r < r_bin - RING_STEP/2)
            inner_values = beam_matrix[inner_mask]  # 使用原始值，而非修改后的值
            
            if len(inner_values) == 0:
                continue
                
            # 获取当前0.1mm环的掩膜
            mask = (r >= r_bin - RING_STEP/2) & (r < r_bin + RING_STEP/2)
            indices = np.where(mask)
            
            # 处理环上的每个点
            for idx in range(len(indices[0])):
                i, j = indices[0][idx], indices[1][idx]
                x = self.x_matrix[i, j]
                y = self.y_matrix[i, j]
                
                # 根据方向选择衰减因子
                if abs(x) > abs(y):
                    # x方向主导
                    decay_factor = DIRECTION_DECAY_FACTORS['x+'] if x > 0 else DIRECTION_DECAY_FACTORS['x-']
                else:
                    # y方向主导
                    decay_factor = DIRECTION_DECAY_FACTORS['y+'] if y > 0 else DIRECTION_DECAY_FACTORS['y-']
                
                # 确定参考值（内层值的最大值乘以方向衰减因子）
                ref_val = np.max(inner_values) * decay_factor
                
                # 柔性约束：混合原始值和约束值
                current_val = beam_matrix[i, j]  # 原始值
                if current_val > ref_val:
                    # 使用加权平均：50%原始值 + 50%约束值
                    beam_new[i, j] = 0.5 * current_val + 0.5 * ref_val
                
                # 应用最小径向梯度约束 (修复：使用最大梯度而非最小梯度)
                radial_gate = decay_factor * (np.max(inner_values) - MIN_RADIAL_SLOPE * self.r_matrix[i, j])
                if beam_new[i, j] < radial_gate:
                    beam_new[i, j] = radial_gate
        
        # 确保中心点仍为最大值
        current_max = np.max(beam_new)
        beam_new[self.center_idx, self.center_idx] = current_max + SMALL_OFFSET
        
        # 轻微高斯平滑
        beam_new = gaussian_filter(beam_new, sigma=0.3)
        
        return beam_new

    def validate_unimodality(self, beam_matrix):
        """
        严格验证束流分布的单峰性
        返回: 1. 是否单峰 (布尔值)
              2. 峰的数量
              3. 最大非中心峰值比例
        """
        # 检查中心是否为全局最大值
        center_val = beam_matrix[self.center_idx, self.center_idx]
        center_mask = (np.abs(self.x_matrix) < 0.5) & (np.abs(self.y_matrix) < 0.5)
        
        # 找到除中心区域外的最大值
        if np.max(beam_matrix[~center_mask]) + SMALL_OFFSET > center_val:
            # 如果中心区域外有更高的点
            return False, 0, 1.0
        
        # 找出除中心外的局部极大值
        max_filtered = maximum_filter(beam_matrix, size=3)
        local_maxima = (beam_matrix == max_filtered) & (beam_matrix > 0.01 * center_val)
        local_maxima[center_mask] = False
        
        # 统计局部极大值数量
        peak_count = np.sum(local_maxima)
        
        # 计算最大非中心峰值比例
        if peak_count > 0:
            max_non_center = np.max(beam_matrix[local_maxima])
            max_ratio = max_non_center / center_val
        else:
            max_ratio = 0
        
        # 判断是否满足单峰性 (仅中心一个峰)
        is_unimodal = (peak_count == 0)
        
        return is_unimodal, peak_count, max_ratio

    def simulate_etching(self, beam_matrix, axis='x'):
        """
        模拟束流扫描产生的刻蚀轮廓
        axis='x': 束流沿Y扫描，测量X方向轮廓
        axis='y': 束流沿X扫描，测量Y方向轮廓
        """
        # 计算漂移卷积核
        kernel_radius = min(100, int(3 * self.drift_sigma / self.grid_spacing))
        kernel_points = 2 * kernel_radius + 1
        kernel_range = np.linspace(-kernel_radius, kernel_radius, kernel_points) * self.grid_spacing
        kernel = np.exp(-0.5 * (kernel_range/self.drift_sigma)**2)
        kernel /= np.sum(kernel) + SMALL_OFFSET  # 归一化，防止除以零
        
        # 创建轮廓数组
        profile = np.zeros(len(self.grid))
        actual_beam = beam_matrix * self.max_val  # 实际刻蚀速率
        
        # 预筛选有效区域
        active_mask = (self.r_matrix < self.opt_radius) & (actual_beam > 0.01 * np.max(actual_beam))
        
        if axis == 'x':  # 沿Y方向扫描
            # 使用向量化方法提高效率
            for i in range(len(self.grid)):
                if np.any(active_mask[i, :]):
                    line = actual_beam[i, :]
                    if len(line) >= 5:
                        profile[i] = simpson(line, self.grid) / self.scan_velocity
                
        elif axis == 'y':  # 沿X方向扫描
            for j in range(len(self.grid)):
                if np.any(active_mask[:, j]):
                    line = actual_beam[:, j]
                    if len(line) >= 5:
                        profile[j] = simpson(line, self.grid) / self.scan_velocity
        
        # 归一化
        max_val = np.max(profile) if np.max(profile) > 0 else 1.0
        return profile / (max_val + SMALL_OFFSET)

    def calculate_error(self, sim_x, sim_y):
        """计算模拟轮廓与实验数据的误差"""
        # X方向误差
        exp_x = self.x_data
        exp_x_pos = exp_x[:, 0]
        exp_x_val = exp_x[:, 1]
        
        try:
            # 实验数据归一化
            exp_x_max = np.max(exp_x_val)
            exp_x_norm = exp_x_val / (exp_x_max + SMALL_OFFSET)
            
            # 模拟数据归一化
            sim_x_norm = sim_x / (np.max(sim_x) + SMALL_OFFSET)
            
            # 插值到相同网格
            sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_norm)
            
            # 计算均方根误差 (只考虑峰值50%以上的数据点)
            valid_mask = exp_x_val > 0.5 * exp_x_max
            rmse_x = np.sqrt(np.mean((sim_x_interp[valid_mask] - exp_x_norm[valid_mask])**2))
            
            # 检查单峰性违反
            is_unimodal, peak_count, max_ratio = self.validate_unimodality_profile(sim_x)
            if not is_unimodal:
                # 针对额外峰值的惩罚 (大幅降低惩罚程度)
                penalty = peak_count * 0.01 + max_ratio * 0.05
                rmse_x += penalty
        except Exception as e:
            self.log(f"X方向误差计算失败: {str(e)}")
            rmse_x = 0.1  # 默认值
        
        # Y方向误差
        exp_y = self.y_data
        exp_y_pos = exp_y[:, 0]
        exp_y_val = exp_y[:, 1]
        
        try:
            exp_y_max = np.max(exp_y_val)
            exp_y_norm = exp_y_val / (exp_y_max + SMALL_OFFSET)
            sim_y_norm = sim_y / (np.max(sim_y) + SMALL_OFFSET)
            sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_norm)
            
            # 计算均方根误差 (只考虑峰值50%以上的数据点)
            valid_mask = exp_y_val > 0.5 * exp_y_max
            rmse_y = np.sqrt(np.mean((sim_y_interp[valid_mask] - exp_y_norm[valid_mask])**2))
            
            is_unimodal, peak_count, max_ratio = self.validate_unimodality_profile(sim_y)
            if not is_unimodal:
                penalty = peak_count * 0.01 + max_ratio * 0.05
                rmse_y += penalty
        except Exception as e:
            self.log(f"Y方向误差计算失败: {str(e)}")
            rmse_y = 0.1  # 默认值
        
        # 综合误差 (X方向权重60%)
        combined_error = 0.6 * rmse_x + 0.4 * rmse_y
        return combined_error

    def validate_unimodality_profile(self, profile):
        """
        验证轮廓曲线的单峰性
        返回: (is_unimodal, peak_count, max_non_center_ratio)
        """
        # 找出局部极大值
        diffs = np.diff(profile)
        peaks = np.where((diffs[:-1] > 0) & (diffs[1:] < 0))[0] + 1
        
        # 中心点索引
        center_idx = np.argmin(np.abs(self.grid))
        center_pos = self.grid[center_idx]
        
        # 中心点是否为全局最大值 (考虑小偏移)
        center_val = profile[center_idx]
        if center_val + SMALL_OFFSET < np.max(profile):
            return False, len(peaks), 1.0
        
        # 统计非中心峰 (排除中心±1mm区域)
        non_center_peaks = []
        for peak_idx in peaks:
            peak_pos = self.grid[peak_idx]
            if abs(peak_pos - center_pos) > 1.0:
                non_center_peaks.append(peak_idx)
        
        # 计算最大非中心峰值比例
        max_non_center = max(profile[non_center_peaks]) if non_center_peaks else 0
        max_ratio = max_non_center / (center_val + SMALL_OFFSET)
        
        return (len(non_center_peaks) == 0), len(non_center_peaks), max_ratio

    def run_optimization(self, iterations=90):
        """运行优化过程 (0.1mm环状优化) - 修复的优化逻辑"""
        self.log(f"开始优化 (0.1mm环状步长)...")
        self.log(f"方向衰减参数:")
        self.log(f"  X正向: {DIRECTION_DECAY_FACTORS['x+']:.3f}, X负向: {DIRECTION_DECAY_FACTORS['x-']:.3f}")
        self.log(f"  Y正向: {DIRECTION_DECAY_FACTORS['y+']:.3f}, Y负向: {DIRECTION_DECAY_FACTORS['y-']:.3f}")
        
        start_time = time.time()
        error_history = []
        unimodal_history = []
        improvements = []
        
        # 初始错误计算
        init_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        init_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        init_error = self.calculate_error(init_sim_x, init_sim_y)
        best_error = init_error
        best_beam = self.optimized_beam.copy()
        error_history.append(init_error)
        
        # 检查初始单峰性状态
        is_unimodal, peak_count, max_ratio = self.validate_unimodality(self.optimized_beam)
        unimodal_history.append(is_unimodal)
        self.log(f"\n初始状态: 绝对误差={init_error:.6f} 单峰性={is_unimodal} (峰值={peak_count})")
        
        # 减少变异幅度并增加探索性
        max_mutation_mag = 0.05  # 最大变异幅度
        min_mutation_mag = 0.001  # 最小变异幅度
        
        for i in range(iterations):
            self.log_progress(i+1, iterations)
            
            # 变异因子（随迭代次数减少）
            mutation_factor = max_mutation_mag * (1 - i/iterations)
            mutation_factor = max(mutation_factor, min_mutation_mag)
            
            # 生成多个变异候选
            num_candidates = 5
            candidates = []
            for _ in range(num_candidates):
                candidate = best_beam.copy()  # 从当前最佳开始变异
                
                # 变异比例基于距离中心距离
                dist_factor = (self.r_matrix / self.opt_radius) ** 2  # 离中心越远，变异概率越高
                mutation_prob = 0.5 * dist_factor
                
                # 在优化区域内变异
                mutation_mask = (self.r_matrix < self.opt_radius) & (np.random.random(best_beam.shape) < mutation_prob)
                
                # 应用变异 (使用高斯噪声)
                mutation = mutation_factor * np.random.randn(*best_beam.shape)
                candidate[mutation_mask] += mutation[mutation_mask]
                
                # 边界约束
                candidate = np.maximum(candidate, 0)
                candidate = np.minimum(candidate, 1.0)
                
                # 应用0.1mm环状的单峰性约束
                candidate = self.enforce_strict_unimodality(candidate)
                
                candidates.append(candidate)
            
            # 评估所有候选方案并选择最佳
            best_candidate = None
            best_candidate_error = float('inf')
            
            for idx, candidate in enumerate(candidates):
                try:
                    cand_sim_x = self.simulate_etching(candidate, 'x')
                    cand_sim_y = self.simulate_etching(candidate, 'y')
                    cand_error = self.calculate_error(cand_sim_x, cand_sim_y)
                    
                    # 检查单峰性 (允许轻微违反)
                    is_unimodal, _, max_ratio = self.validate_unimodality(candidate)
                    non_center_penalty = max_ratio * 0.01 if max_ratio > 0.1 else 0
                    
                    # 如果满足单峰性或惩罚较小
                    if is_unimodal or non_center_penalty < 0.02:
                        if cand_error < best_candidate_error:
                            best_candidate_error = cand_error
                            best_candidate = candidate
                except:
                    continue
                
            # 检查改进情况
            improvement = best_error - best_candidate_error
            relative_improv = (improvement / (best_error + SMALL_OFFSET)) * 100
            
            # 接受策略：误差降低或轻微增加但满足形状约束
            if best_candidate is not None and (improvement > 0 or relative_improv > -5):
                # 保持记录
                improvements.append(improvement)
                
                # 更新最佳状态
                if best_candidate_error < best_error:
                    best_error = best_candidate_error
                    best_beam = best_candidate
                    self.optimized_beam = best_candidate
                    
                    # 每5次迭代记录日志
                    if i % 5 == 0:
                        self.log(f"Iter {i}: 误差={best_error:.6f} ({relative_improv:.1f}%)")
            else:
                # 如果没有改进
                if i % 5 == 0:
                    self.log(f"Iter {i}: 无改进 (当前:{best_error:.6f}, 候选:{best_candidate_error:.6f})")
            
            # 记录历史
            error_history.append(best_candidate_error if best_candidate is not None else best_error)
            is_unimodal, _, _ = self.validate_unimodality(self.optimized_beam)
            unimodal_history.append(is_unimodal)
        
        # 结束优化
        sys.stdout.write("\n")
        elapsed_time = time.time() - start_time
        
        # 最终评估
        final_sim_x = self.simulate_etching(best_beam, 'x')
        final_sim_y = self.simulate_etching(best_beam, 'y')
        final_error = self.calculate_error(final_sim_x, final_sim_y)
        
        # 保存输出
        output_beam = best_beam * self.max_val
        np.savetxt("directional_optimized_beam.csv", output_beam, delimiter=",")
        
        # 保存方向衰减参数
        with open("decay_parameters.txt", "w") as f:
            f.write(f"# 束流优化方向衰减参数\n")
            f.write(f"# X+方向衰减因子: {DIRECTION_DECAY_FACTORS['x+']}\n")
            f.write(f"# X-方向衰减因子: {DIRECTION_DECAY_FACTORS['x-']}\n")
            f.write(f"# Y+方向衰减因子: {DIRECTION_DECAY_FACTORS['y+']}\n")
            f.write(f"# Y-方向衰减因子: {DIRECTION_DECAY_FACTORS['y-']}\n")
            f.write(f"# 环形步长: {RING_STEP} mm\n")
            f.write(f"# 初始误差: {init_error:.6f}\n")
            f.write(f"# 最终误差: {final_error:.6f}\n")
        
        self.log(f"\n优化完成! 耗时: {elapsed_time:.1f}秒")
        self.log(f"初始误差: {init_error:.6f} → 最终误差: {final_error:.6f}")
        self.log(f"优化改进: {((init_error - final_error)/init_error*100):.1f}%")
        
        # 可视化结果
        self.visualize_results(
            init_sim_x, init_sim_y, 
            final_sim_x, final_sim_y, 
            error_history, unimodal_history
        )
        
        return output_beam

    def visualize_results(self, init_sim_x, init_sim_y, final_sim_x, final_sim_y, errors, unimodal):
        """创建可视化报告"""
        try:
            fig = plt.figure(figsize=(16, 18))
            fig.suptitle("方向性束流优化结果 (0.1mm环状优化)", fontsize=20, y=0.98)
            
            # 网格布局 (3行2列)
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
            
            # 1. 初始束流分布
            ax1 = plt.subplot(gs[0, 0])
            init_norm = self.initial_beam / self.max_val
            im = ax1.imshow(init_norm, cmap='viridis', extent=[-15, 15, -15, 15], 
                           origin='lower', aspect='auto', interpolation='bicubic')
            plt.colorbar(im, ax=ax1)
            ax1.set_title("初始束流分布", fontsize=14)
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 标记中心点
            ax1.plot(0, 0, 'r+', markersize=12, markeredgewidth=2)
            
            # 2. 优化后束流分布
            ax2 = plt.subplot(gs[0, 1])
            if hasattr(self, 'optimized_beam'):
                im = ax2.imshow(self.optimized_beam, cmap='viridis', extent=[-15, 15, -15, 15], 
                               origin='lower', aspect='auto', interpolation='bicubic')
                plt.colorbar(im, ax=ax2)
                
                # 绘制方向标识
                ax2.arrow(2, 0, 3, 0, width=0.2, head_width=0.8, head_length=0.8, 
                          fc='w', ec='w', linewidth=2)
                ax2.text(5, 0.5, f'x+衰减:{DIRECTION_DECAY_FACTORS["x+"]:.3f}', 
                         color='w', fontsize=10, ha='center')
                
                ax2.arrow(-2, 0, -3, 0, width=0.2, head_width=0.8, head_length=0.8, 
                          fc='w', ec='w', linewidth=2)
                ax2.text(-5, 0.5, f'x-衰减:{DIRECTION_DECAY_FACTORS["x-"]:.3f}', 
                         color='w', fontsize=10, ha='center')
                
                ax2.arrow(0, 2, 0, 3, width=0.2, head_width=0.8, head_length=0.8, 
                          fc='w', ec='w', linewidth=2)
                ax2.text(0.5, 5, f'y+衰减:{DIRECTION_DECAY_FACTORS["y+"]:.3f}', 
                         color='w', fontsize=10, ha='center')
                
                ax2.arrow(0, -2, 0, -3, width=0.2, head_width=0.8, head_length=0.8, 
                          fc='w', ec='w', linewidth=2)
                ax2.text(0.5, -5, f'y-衰减:{DIRECTION_DECAY_FACTORS["y-"]:.3f}', 
                         color='w', fontsize=10, ha='center')
                
                ax2.set_title("优化后束流分布 (方向性衰减)", fontsize=14)
                ax2.set_xlabel("X (mm)")
                ax2.set_ylabel("Y (mm)")
                
                # 检查单峰性
                is_unimodal, peak_count, max_ratio = self.validate_unimodality(self.optimized_beam)
                status = "通过" if is_unimodal else f"违反 ({peak_count}个峰)"
                ax2.text(0.5, -0.12, f"单峰性检查: {status} | 环状步长={RING_STEP:.2f}mm", 
                        transform=ax2.transAxes, fontsize=11, ha='center',
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            
            # 3. X方向初始拟合
            ax3 = plt.subplot(gs[1, 0])
            ax3.set_title("初始X方向截面拟合", fontsize=14)
            # 实验数据
            ax3.plot(self.x_data[:, 0], self.x_data[:, 1] / np.max(self.x_data[:, 1]), 
                    'bo', alpha=0.5, markersize=5, label='实验数据')
            # 模拟数据
            ax3.plot(self.grid, init_sim_x, 'r-', linewidth=2, label='初始拟合')
            ax3.legend(loc='best')
            ax3.set_xlabel("X位置 (mm)")
            ax3.set_ylabel("归一化刻蚀深度")
            ax3.grid(True, linestyle='--', alpha=0.6)
            
            # 标记峰值位置
            peaks = np.where(np.diff(np.sign(np.diff(init_sim_x))) < 0)[0] + 1
            center_idx = np.argmin(np.abs(self.grid))
            ax3.plot(self.grid[peaks], init_sim_x[peaks], 'go', markersize=8, 
                    fillstyle='none', label='峰值点')
            if peaks.size > 0:
                ax3.legend(loc='best')
            
            # 4. X方向优化后拟合
            ax4 = plt.subplot(gs[1, 1])
            ax4.set_title("优化后X方向截面拟合", fontsize=14)
            # 实验数据
            ax4.plot(self.x_data[:, 0], self.x_data[:, 1] / np.max(self.x_data[:, 1]), 
                    'bo', alpha=0.5, markersize=5, label='实验数据')
            # 模拟数据
            ax4.plot(self.grid, final_sim_x, 'r-', linewidth=2, label='优化拟合')
            ax4.set_xlabel("X位置 (mm)")
            ax4.set_ylabel("归一化刻蚀深度")
            ax4.grid(True, linestyle='--', alpha=0.6)
            
            # 验证单峰性
            is_single_peak, peak_count, max_ratio = self.validate_unimodality_profile(final_sim_x)
            status = "单峰" if is_single_peak else "多峰"
            ax4.text(0.8, 0.9, f"拟合模式: {status}", 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            
            # 5. 误差变化曲线
            ax5 = plt.subplot(gs[2, 0])
            ax5.plot(range(len(errors)), errors, 'b-', linewidth=2, label='绝对误差')
            ax5.set_xlabel("迭代次数")
            ax5.set_ylabel("优化误差", color='b')
            ax5.grid(True, linestyle='--', alpha=0.6)
            ax5.set_title("优化误差变化", fontsize=14)
            
            # 添加单峰性记录
            ax5b = ax5.twinx()
            unimodal_int = np.array(unimodal, dtype=int)
            ax5b.step(range(len(unimodal)), unimodal_int, 'm--', where='mid')
            ax5b.set_ylim(-0.1, 1.5)
            ax5b.set_yticks([0, 1])
            ax5b.set_yticklabels(['违反', '通过'])
            ax5b.set_ylabel("单峰性检查", color='m')
            
            # 6. Y方向拟合结果
            ax6 = plt.subplot(gs[2, 1])
            ax6.set_title("Y方向截面拟合对比", fontsize=14)
            # 实验数据
            ax6.plot(self.y_data[:, 0], self.y_data[:, 1] / np.max(self.y_data[:, 1]), 
                    'g^', alpha=0.5, markersize=5, label='实验数据')
            # 初始拟合
            ax6.plot(self.grid, init_sim_y, 'm--', linewidth=1.5, label='初始拟合')
            # 优化拟合
            ax6.plot(self.grid, final_sim_y, 'g-', linewidth=2, label='优化拟合')
            ax6.set_xlabel("Y位置 (mm)")
            ax6.set_ylabel("归一化刻蚀深度")
            ax6.grid(True, linestyle='--', alpha=0.6)
            ax6.legend(loc='best')
            
            # 标记方向差异
            if len(init_sim_y) == len(final_sim_y):
                y_diff = np.mean(np.abs(init_sim_y - final_sim_y))
                ax6.text(0.1, 0.9, f"Y方向平均改进: {y_diff:.4f}", 
                        transform=ax6.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            
            # 方向衰减因子注释
            fig.text(0.82, 0.12, 
                    f"方向衰减因子:\nX+: {DIRECTION_DECAY_FACTORS['x+']:.3f}\nX-: {DIRECTION_DECAY_FACTORS['x-']:.3f}\nY+: {DIRECTION_DECAY_FACTORS['y+']:.3f}\nY-: {DIRECTION_DECAY_FACTORS['y-']:.3f}",
                    fontsize=11, bbox=dict(facecolor='lightyellow', alpha=0.7))
            
            # 环状步长注释
            fig.text(0.82, 0.05, f"环状步长: {RING_STEP}mm", fontsize=11)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            # 保存结果
            plt.savefig("directional_beam_optimization_result.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.log("结果可视化已保存")
            
        except Exception as e:
            self.log(f"可视化失败: {str(e)}")
            self.log(traceback.format_exc())
    
    def finalize(self):
        """结束优化并关闭日志"""
        self.log("优化完成! 结果已保存")
        self.log("  - directional_optimized_beam.csv (优化后的束流分布)")
        self.log("  - decay_parameters.txt (方向衰减参数)")
        self.log("  - directional_beam_optimization_result.png (可视化报告)")
        
        if self.log_file:
            self.log_file.close()

# ================== 主程序 ==================
def main():
    # 输入文件路径
    input_files = {
        "beam_traced_x_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",
        "beam_traced_y_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 80)
    print("方向性束流优化引擎 - 修复版本".center(80))
    print(f"支持四方向独立衰减控制 (环形步长:{RING_STEP}mm)".center(80))
    print("=" * 80)
    print("方向衰减因子配置:")
    print(f"  - X+方向: {DIRECTION_DECAY_FACTORS['x+']:.3f}")
    print(f"  - X-方向: {DIRECTION_DECAY_FACTORS['x-']:.3f}")
    print(f"  - Y+方向: {DIRECTION_DECAY_FACTORS['y+']:.3f}")
    print(f"  - Y-方向: {DIRECTION_DECAY_FACTORS['y-']:.3f}")
    print("=" * 30)
    print("输入文件:")
    for name, path in input_files.items():
        print(f"  - {name:55}: {path}")
    print("=" * 80)
    
    # 文件检查
    missing_files = [f"  - {name}: {path}" for name, path in input_files.items() if not os.path.exists(path)]
    
    if missing_files:
        print("警告: 以下文件不存在:")
        for msg in missing_files:
            print(msg)
        
        # 询问是否继续
        proceed = input("部分文件缺失，是否继续? (y/n): ").strip().lower()
        if proceed != 'y':
            print("程序终止")
            return
    
    try:
        # 初始化优化器
        optimizer = DirectionalBeamOptimizer(
            beam_traced_x_axis=input_files["beam_traced_x_axis"],
            beam_traced_y_axis=input_files["beam_traced_y_axis"],
            initial_guess_path=input_files["initial_beam"]
        )
        
        # 运行优化
        optimized_beam = optimizer.run_optimization(iterations=100)
        
        # 生成报告
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("方向性优化成功完成!".center(80))
        print("结果文件:")
        print("  - directional_optimized_beam.csv (优化后束流分布)")
        print("  - decay_parameters.txt (方向衰减参数)")
        print("  - directional_beam_optimization_result.png (可视化报告)")
        print("=" * 80)
        
    except Exception as e:
        print(f"优化出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
