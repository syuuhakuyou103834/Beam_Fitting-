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

def setup_plotting():
    """配置绘图环境，解决中文字体问题"""
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'],
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 300
    })
    
setup_plotting()

def down_sample_beam(beam_matrix, src_resolution=0.1, target_resolution=1.0):
    """
    将束流分布从高分辨率降采样到低分辨率
    """
    grid_bound = GRID_BOUND
    target_points = int(2 * grid_bound / target_resolution) + 1
    target_grid = np.linspace(-grid_bound, grid_bound, target_points)
    
    src_grid = np.linspace(-grid_bound, grid_bound, beam_matrix.shape[0])
    
    interp_fn = RegularGridInterpolator(
        (src_grid, src_grid), 
        beam_matrix,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )
    
    xx, yy = np.meshgrid(target_grid, target_grid, indexing='ij')
    points = np.stack((xx, yy), axis=-1)
    
    sampled_beam = interp_fn(points)
    sampled_beam = np.maximum(sampled_beam, 0)
    
    return sampled_beam

# 参数配置
HIGH_PRECISION = 0.1
GRID_BOUND = 15.0
RING_STEP = 0.1
EDGE_REGION = 2.0  # 边缘区域定义（半高峰宽之外）

# 方向衰减参数 - 为边缘区域添加额外控制
DIRECTION_DECAY_FACTORS = {
    'x+': {'core': 0.75, 'edge': 0.55},
    'x-': {'core': 0.75, 'edge': 0.55},
    'y+': {'core': 0.75, 'edge': 0.55},
    'y-': {'core': 0.75, 'edge': 0.55}
}

MIN_RADIAL_SLOPE = 0.002
SMALL_OFFSET = 1e-6  # 更小的偏移量避免除零错误

class DirectionalBeamOptimizer:
    def __init__(self, beam_traced_x_axis, beam_traced_y_axis, initial_guess_path):
        """增强的边缘区域优化的束流优化器"""
        # 创建日志文件
        self.log_file = open("enhanced_edge_beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("增强版束流优化引擎 - 针对边缘区域优化")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"网格精度: {HIGH_PRECISION:.2f}mm, 环状步长: {RING_STEP:.2f}mm")
        self.log(f"边缘区域定义: FWHM之外 ±{EDGE_REGION}mm")
        self.log("方向衰减因子(核心/边缘):")
        for dir, factors in DIRECTION_DECAY_FACTORS.items():
            self.log(f"  - {dir}方向: {factors['core']:.3f}/{factors['edge']:.3f}")
        
        # 计算网格点数 (修复初始值问题)
        self.grid_points = int(GRID_BOUND * 2 / HIGH_PRECISION) + 1
        self.grid = np.linspace(-GRID_BOUND, GRID_BOUND, self.grid_points)
        self.grid_spacing = HIGH_PRECISION
        
        # 创建距离矩阵（以原点为中心）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_matrix = np.sqrt(xx**2 + yy**2)  # 距离原点距离
        self.x_matrix = xx
        self.y_matrix = yy
        self.angle_matrix = np.arctan2(yy, xx)
        
        # 优化参数
        self.opt_radius = GRID_BOUND - 1.0
        self.scan_velocity = 30.0
        self.drift_sigma = 1.8
        self.smooth_sigma = 0.5
        self.edge_weight_factor = 1.5  # 边缘数据的额外权重
        
        # 定义边缘阈值默认值（解决依赖问题）
        self.edge_threshold_x = EDGE_REGION
        self.edge_threshold_y = EDGE_REGION
        
        # 加载和处理数据
        self.log("加载并预处理实验数据...")
        self.x_data = self.load_and_preprocess_data(beam_traced_x_axis, 'x')
        self.y_data = self.load_and_preprocess_data(beam_traced_y_axis, 'y')
        
        # 确定边缘区域阈值
        self.edge_threshold_x = self.find_edge_threshold(self.x_data)
        self.edge_threshold_y = self.find_edge_threshold(self.y_data)
        self.log(f"基于实验数据的边缘区域阈值: X={self.edge_threshold_x:.2f}mm, Y={self.edge_threshold_y:.2f}mm")
        
        # 创建边缘掩膜（必须放在数据加载之后）
        self.edge_mask = self.create_edge_mask()
        
        # 加载初始束流
        self.load_initial_beam(initial_guess_path)
        
        # 初始化优化束流
        self.optimized_beam = self.initial_beam / (self.max_val + SMALL_OFFSET)
        self.center_idx = self.grid_points // 2
        self.center_x = self.grid[self.center_idx]
        self.center_y = self.grid[self.center_idx]
        
        # 历史记录
        self.history = {
            "iteration": [0],
            "abs_error": [],
            "edge_error": [],
            "peak_count": []
        }

    def find_edge_threshold(self, data):
        """确定边缘区域的阈值位置（半高峰宽）"""
        values = data[:, 1]
        max_val = np.max(values)
        
        if max_val < SMALL_OFFSET:
            return EDGE_REGION  # 默认值
        
        half_max = max_val * 0.5
        
        # 寻找50%峰值的位置
        above_half = np.where(values > half_max)[0]
        if len(above_half) < 2:
            return EDGE_REGION  # 默认值
            
        fwhm_start = np.min(data[above_half, 0])
        fwhm_end = np.max(data[above_half, 0])
        
        # 边缘区域在FWHM之外
        threshold = max(abs(fwhm_start), abs(fwhm_end)) + EDGE_REGION
        return min(threshold, GRID_BOUND - 0.5)  # 不能超过网格边界

    def create_edge_mask(self):
        """创建边缘区域掩模"""
        x_mask = (np.abs(self.x_matrix) > self.edge_threshold_x)
        y_mask = (np.abs(self.y_matrix) > self.edge_threshold_y)
        return np.logical_or(x_mask, y_mask)

    def log(self, message):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_file.write(log_entry + "\n")
        self.log_file.flush()
        
    def log_progress(self, current, total, stage=None):
        """显示优化进度"""
        if stage is not None:
            prefix = f"[阶段 {stage}]"
        else:
            prefix = "[优化进度]"
        
        progress = int(100 * current/total)
        bar = '=' * (progress//2)
        sys.stdout.write(f"\r{prefix} |{bar:{50}}| {progress}%")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    def load_and_preprocess_data(self, file_path, axis):
        """加载实验数据并进行预处理 - 特别处理边缘区域"""
        try:
            # 加载原始数据
            data = np.loadtxt(file_path, delimiter=",")
            
            # 确保数据有两列
            if data.ndim != 2 or data.shape[1] != 2:
                self.log(f"警告: {file_path} 数据格式不正确，应为N行2列")
                # 返回默认数据（中心峰）
                r = np.abs(self.grid)
                return np.column_stack((self.grid, np.exp(-(r**2)/(2*5.0**2))))
                
            # 找出峰值位置并进行平移
            peak_idx = np.argmax(data[:, 1])
            peak_pos = data[peak_idx, 0]
            shift_value = -peak_pos
            
            # 应用平移
            data[:, 0] += shift_value
            
            # 使用三次样条插值到标准网格
            valid_mask = np.isfinite(data[:, 1]) & (data[:, 1] > 0)
            if np.sum(valid_mask) < 5:
                self.log(f"错误: {file_path} 有效数据点不足")
                # 返回默认数据
                r = np.abs(self.grid)
                return np.column_stack(self.grid, np.exp(-(r**2)/(2*5.0**2)))
            
            interpolator = interp1d(
                data[valid_mask, 0], 
                data[valid_mask, 1], 
                kind='linear',  # 使用线性插值更稳定
                bounds_error=False, 
                fill_value=0.0
            )
            
            new_values = interpolator(self.grid)
            
            # 平滑处理（中心区域平滑更强）
            new_values = gaussian_filter(new_values, sigma=0.8)
            
            # 非负约束
            new_values = np.maximum(new_values, 0)
            
            # 创建新数据集
            new_data = np.column_stack((self.grid, new_values))
            new_peak_idx = np.argmax(new_values)
            self.log(f"  {axis}数据加载成功 | 峰值位置: {self.grid[new_peak_idx]:.2f}mm | 强度: {new_values[new_peak_idx]:.2f}")
            
            return new_data
            
        except Exception as e:
            self.log(f"加载数据失败: {str(e)}")
            # 返回默认数据（高斯分布）
            r = np.abs(self.grid)
            return np.column_stack(self.grid, np.exp(-(r**2)/(2*5.0**2)))

    def load_initial_beam(self, file_path):
        """加载初始束流分布，强制中心点峰值"""
        # 默认高斯分布
        r = np.sqrt(self.x_matrix**2 + self.y_matrix**2)
        default_beam = 100 * np.exp(-r**2 / (2*5.0**2))
        
        try:
            # 加载初始束流
            initial_data = np.genfromtxt(file_path, delimiter=",")
            
            if initial_data.size < 4:  # 确保有足够的数据点
                self.log("警告: 初始束流数据不足，使用默认高斯分布")
                highres_beam = default_beam
            else:
                # 确保尺寸正确
                source_points = initial_data.shape[0]
                if len(initial_data.shape) < 2:
                    self.log("警告: 初始束流应为二维矩阵")
                    highres_beam = default_beam
                else:
                    # 创建插值器
                    source_grid = np.linspace(-GRID_BOUND, GRID_BOUND, source_points)
                    interpolator = RegularGridInterpolator(
                        (source_grid, source_grid),
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
            exp_peak_val_x = np.max(self.x_data[:, 1]) if np.any(self.x_data[:, 1] > 0) else np.max(highres_beam)
            grid_center = self.grid_points // 2
            
            # 取中心3x3区域的平均值
            center_val = np.mean(highres_beam[grid_center-1:grid_center+2, grid_center-1:grid_center+2])
            
            if center_val > SMALL_OFFSET:
                calibration_factor = exp_peak_val_x / center_val
                highres_beam *= calibration_factor
            
            # 确保中心为最大值
            max_val = np.max(highres_beam)
            highres_beam[grid_center, grid_center] = max_val
            
            # 应用高斯平滑
            highres_beam = gaussian_filter(highres_beam, sigma=0.5)
            
            self.initial_beam = highres_beam
            self.max_val = max_val
            
            self.log(f"加载初始束流成功: 中心值={self.max_val:.2f} nm/s")
            
        except Exception as e:
            self.log(f"加载初始束流失败: {str(e)}，使用默认高斯分布")
            self.initial_beam = default_beam
            self.max_val = np.max(default_beam)

    def enforce_strict_unimodality(self, beam_matrix):
        """
        改进的单峰性约束 - 为边缘区域设计
        """
        r = self.r_matrix
        radial_bins = np.arange(0, np.max(r), RING_STEP)
        beam_new = beam_matrix.copy()
        
        center_val = beam_matrix[self.center_idx, self.center_idx]
        if center_val < np.max(beam_matrix):
            beam_new[self.center_idx, self.center_idx] = np.max(beam_matrix)
        
        # 从中心向外逐环处理
        for r_bin in radial_bins[1:]:
            # 内层参考值
            inner_mask = (r >= r_bin - RING_STEP) & (r < r_bin - RING_STEP/2)
            inner_values = beam_matrix[inner_mask]
            
            if len(inner_values) == 0:
                continue
                
            # 获取最大内层值（去除NaN和无穷值）
            max_inner_val = np.nanmax(inner_values)
            if np.isnan(max_inner_val) or np.isinf(max_inner_val) or max_inner_val < SMALL_OFFSET:
                continue
                
            mask = (r >= r_bin - RING_STEP/2) & (r < r_bin + RING_STEP/2)
            indices = np.where(mask)
            
            for idx in range(len(indices[0])):
                i, j = indices[0][idx], indices[1][idx]
                x, y = self.x_matrix[i, j], self.y_matrix[i, j]
                
                # 是否为边缘区域
                is_edge = self.edge_mask[i, j] if hasattr(self, 'edge_mask') else False
                
                # 方向选择
                if abs(x) > abs(y):
                    factor_type = 'edge' if is_edge else 'core'
                    decay_factor = DIRECTION_DECAY_FACTORS['x+'][factor_type] if x > 0 else DIRECTION_DECAY_FACTORS['x-'][factor_type]
                else:
                    factor_type = 'edge' if is_edge else 'core'
                    decay_factor = DIRECTION_DECAY_FACTORS['y+'][factor_type] if y > 0 else DIRECTION_DECAY_FACTORS['y-'][factor_type]
                
                ref_val = max_inner_val * decay_factor
                
                # 当前值
                current_val = beam_matrix[i, j]
                
                # 柔性约束
                if current_val > ref_val:
                    beam_new[i, j] = 0.5 * current_val + 0.5 * ref_val
                elif current_val < ref_val * 0.7:
                    beam_new[i, j] = ref_val * 0.7
                
                # 应用最小径向梯度约束
                radial_gate = decay_factor * (max_inner_val - MIN_RADIAL_SLOPE * self.r_matrix[i, j])
                if radial_gate > 0 and beam_new[i, j] < radial_gate:
                    beam_new[i, j] = radial_gate
        
        return beam_new

    def simulate_etching(self, beam_matrix, axis='x'):
        """模拟束流扫描产生的刻蚀轮廓"""
        # 计算漂移卷积核
        kernel_radius = min(100, int(3 * self.drift_sigma / self.grid_spacing))
        kernel_points = 2  * kernel_radius + 1
        kernel_range = np.linspace(-kernel_radius, kernel_radius, kernel_points) * self.grid_spacing
        kernel = np.exp(-0.5 * (kernel_range/self.drift_sigma)**2)
        kernel /= np.sum(kernel) + SMALL_OFFSET  # 归一化
        
        # 创建轮廓数组
        profile = np.zeros(len(self.grid))
        actual_beam = beam_matrix * self.max_val  # 实际刻蚀速率
        
        # 预筛选有效区域
        active_mask = (self.r_matrix < self.opt_radius) & (actual_beam > 0.01 * np.max(actual_beam))
        
        if axis == 'x':  # 沿Y方向扫描
            for i in range(len(self.grid)):
                if np.any(active_mask[i, :]):
                    line = actual_beam[i, :]
                    profile[i] = simpson(line, self.grid) / self.scan_velocity
        elif axis == 'y':  # 沿X方向扫描
            for j in range(len(self.grid)):
                if np.any(active_mask[:, j]):
                    line = actual_beam[:, j]
                    profile[j] = simpson(line, self.grid) / self.scan_velocity
        
        # 应用漂移效果
        profile = np.convolve(profile, kernel, mode='same')
        
        # 归一化 (确保有正值)
        max_val = np.max(profile)
        if max_val < SMALL_OFFSET:
            return np.zeros_like(profile)  # 防止除零错误
            
        return profile / max_val

    def calculate_error(self, sim_x, sim_y):
        """计算模拟轮廓与实验数据的误差 - 稳健版本"""
        # 检查输入有效性
        if not hasattr(self, 'x_data') or not hasattr(self, 'y_data'):
            self.log("错误: 缺少实验数据，无法计算误差")
            return 1.0  # 返回较高误差
            
        # X方向误差
        exp_x = self.x_data
        exp_x_pos = exp_x[:, 0]
        exp_x_val = exp_x[:, 1]
        
        try:
            # 处理可能的零值问题
            exp_x_max = np.max(exp_x_val)
            if exp_x_max < SMALL_OFFSET:
                exp_x_max = 1.0  # 避免除零错误
            exp_x_norm = exp_x_val / exp_x_max
            
            # 处理模拟数据问题
            sim_x_max = np.max(sim_x)
            if sim_x_max < SMALL_OFFSET:
                sim_x = np.zeros_like(exp_x_norm)
            else:
                sim_x_norm = sim_x / sim_x_max
            
            # 插值到相同网格
            use_sim_x = sim_x if sim_x_max > SMALL_OFFSET else np.zeros(len(self.grid))
            sim_x_interp = np.interp(exp_x_pos, self.grid, use_sim_x)
            
            # 只考虑正值数据计算误差
            valid_mask = exp_x_val > 0.2 * exp_x_max
            if np.sum(valid_mask) < 5:
                valid_mask = exp_x_val > 0  # 仅排除零值
                
            rmse_x = np.sqrt(np.mean((sim_x_interp[valid_mask] - exp_x_norm[valid_mask])**2))
            
        except Exception as e:
            self.log(f"X方向误差计算失败: {str(e)}")
            rmse_x = 1.0  # 较高默认值
        
        # Y方向误差 (类似处理)
        exp_y = self.y_data
        exp_y_pos = exp_y[:, 0]
        exp_y_val = exp_y[:, 1]
        
        try:
            exp_y_max = np.max(exp_y_val)
            if exp_y_max < SMALL_OFFSET:
                exp_y_max = 1.0
            exp_y_norm = exp_y_val / exp_y_max
            
            sim_y_max = np.max(sim_y)
            if sim_y_max < SMALL_OFFSET:
                sim_y = np.zeros_like(exp_y_norm)
            else:
                sim_y_norm = sim_y / sim_y_max
            
            use_sim_y = sim_y if sim_y_max > SMALL_OFFSET else np.zeros(len(self.grid))
            sim_y_interp = np.interp(exp_y_pos, self.grid, use_sim_y)
            
            valid_mask = exp_y_val > 0.2 * exp_y_max
            if np.sum(valid_mask) < 5:
                valid_mask = exp_y_val > 0
                
            rmse_y = np.sqrt(np.mean((sim_y_interp[valid_mask] - exp_y_norm[valid_mask])**2))
            
        except Exception as e:
            self.log(f"Y方向误差计算失败: {str(e)}")
            rmse_y = 1.0
        
        # 综合误差 (X方向权重60%)
        combined_error = 0.6 * rmse_x + 0.4 * rmse_y
        
        return combined_error

    def run_optimization(self, iterations=50, local_steps=10):
        """运行优化的两阶段优化过程"""
        self.log("开始优化过程...")
        
        start_time = time.time()
        error_history = []
        unimodal_history = []
        
        # 初始计算
        init_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        init_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        init_error = self.calculate_error(init_sim_x, init_sim_y)
        best_error = init_error
        best_beam = self.optimized_beam.copy()
        error_history.append(init_error)
        
        # 初始单峰性
        is_unimodal, peak_count, max_ratio = True, 0, 0.0
        unimodal_history.append(is_unimodal)
        self.log(f"\n初始状态: 误差={init_error:.6f}")
        
        # === 全局优化 ===
        self.log("\n=== 开始全局优化 ===")
        
        for i in range(iterations):
            self.log_progress(i+1, iterations)
            
            mutation_factor = 0.05 * (1 - i/iterations)
            num_candidates = 3 # 减少候选数以加速
            
            candidates = []
            for _ in range(num_candidates):
                candidate = best_beam.copy()
                
                # 应用随机变异
                mutation = mutation_factor * np.random.randn(*best_beam.shape)
                mutation_mask = np.random.random(best_beam.shape) < 0.4
                candidate[mutation_mask] += mutation[mutation_mask]
                
                # 约束值范围
                candidate = np.maximum(candidate, 0)
                candidate = np.minimum(candidate, 1.0)
                
                # 应用单峰性约束
                candidate = self.enforce_strict_unimodality(candidate)
                candidates.append(candidate)
            
            # 评估所有候选方案
            best_candidate = None
            best_candidate_error = float('inf')
            
            for candidate in candidates:
                try:
                    cand_sim_x = self.simulate_etching(candidate, 'x')
                    cand_sim_y = self.simulate_etching(candidate, 'y')
                    cand_error = self.calculate_error(cand_sim_x, cand_sim_y)
                    
                    if cand_error < best_candidate_error:
                        best_candidate_error = cand_error
                        best_candidate = candidate
                except:
                    continue
                
            # 检查改进情况
            if best_candidate is not None and (best_candidate_error < best_error or i < 5):
                best_error = best_candidate_error
                best_beam = best_candidate
                self.optimized_beam = best_beam
                
                if i % 5 == 0:
                    self.log(f"Iter {i+1}: 误差={best_error:.6f}")
            
            # 记录历史
            error_history.append(best_error)
        
        # 保存最终结果
        sys.stdout.write("\n")
        elapsed_time = time.time() - start_time
        
        # 最终评估
        final_sim_x = self.simulate_etching(best_beam, 'x')
        final_sim_y = self.simulate_etching(best_beam, 'y')
        final_error = self.calculate_error(final_sim_x, final_sim_y)
        
        output_beam_highres = best_beam * self.max_val
        np.savetxt("optimized_beam_highres.csv", output_beam_highres, delimiter=",")
        
        # 保存低精度版本
        output_beam_lowres = down_sample_beam(output_beam_highres, HIGH_PRECISION, 1.0)
        np.savetxt("optimized_beam_lowres.csv", output_beam_lowres, delimiter=",")
        
        self.log(f"\n优化完成! 耗时: {elapsed_time:.1f}秒")
        self.log(f"初始误差: {init_error:.6f} → 最终误差: {final_error:.6f}")
        self.log(f"优化改进: {((init_error - final_error)/init_error*100):.1f}%")
        self.log(f"结果文件:")
        self.log(f"  - optimized_beam_highres.csv (高精度束流分布)")
        self.log(f"  - optimized_beam_lowres.csv (低精度束流分布)")
        
        # 可视化结果
        self.visualize_results(
            init_sim_x, init_sim_y, 
            final_sim_x, final_sim_y, 
            error_history,
            output_beam_highres
        )
        
        return output_beam_highres

    def visualize_results(self, init_sim_x, init_sim_y, final_sim_x, final_sim_y, errors, highres_beam):
        """创建可视化报告 - 稳健版本"""
        try:
            plt.figure(figsize=(14, 12))
            
            # 1. 初始束流分布
            plt.subplot(2, 2, 1)
            init_norm = self.initial_beam / (np.max(self.initial_beam) + SMALL_OFFSET)
            plt.imshow(init_norm, cmap='viridis', extent=[-15, 15, -15, 15], 
                      origin='lower', aspect='auto', interpolation='bicubic')
            plt.colorbar(label='束流强度 (归一化)')
            plt.title("初始束流分布")
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            
            # 2. 优化后束流分布
            plt.subplot(2, 2, 2)
            if hasattr(self, 'optimized_beam'):
                opt_beam = self.optimized_beam / (np.max(self.optimized_beam) + SMALL_OFFSET)
                plt.imshow(opt_beam, cmap='viridis', extent=[-15, 15, -15, 15], 
                          origin='lower', aspect='auto', interpolation='bicubic')
                plt.colorbar(label='束流强度 (归一化)')
                plt.title("优化后束流分布")
                plt.xlabel("X (mm)")
                plt.ylabel("Y (mm)")
            
            # 3. X方向拟合对比
            plt.subplot(2, 2, 3)
            # 实验数据
            plt.plot(self.x_data[:, 0], self.x_data[:, 1]/np.max(self.x_data[:, 1]), 
                    'bo', alpha=0.5, markersize=3, label='实验数据(X)')
            # 模拟数据
            plt.plot(self.grid, init_sim_x, 'g-', alpha=0.7, label='初始拟合(X)')
            plt.plot(self.grid, final_sim_x, 'r-', label='优化拟合(X)')
            plt.legend()
            plt.xlabel("位置 (mm)")
            plt.ylabel("归一化深度")
            plt.title("X方向截面拟合")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # 4. Y方向拟合对比
            plt.subplot(2, 2, 4)
            # 实验数据
            plt.plot(self.y_data[:, 0], self.y_data[:, 1]/np.max(self.y_data[:, 1]), 
                    'go', alpha=0.5, markersize=3, label='实验数据(Y)')
            # 模拟数据
            plt.plot(self.grid, init_sim_y, 'b-', alpha=0.7, label='初始拟合(Y)')
            plt.plot(self.grid, final_sim_y, 'm-', label='优化拟合(Y)')
            plt.legend()
            plt.xlabel("位置 (mm)")
            plt.ylabel("归一化深度")
            plt.title("Y方向截面拟合")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # 添加误差曲线
            plt.tight_layout()
            plt.savefig("beam_optimization_result.png", dpi=200, bbox_inches='tight')
            plt.close()
            
            # 单独保存误差曲线
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(errors)), errors, 'b-', linewidth=2)
            plt.xlabel("迭代次数")
            plt.ylabel("综合误差")
            plt.title("优化误差变化")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig("optimization_error.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log("结果可视化已保存")
            
        except Exception as e:
            self.log(f"可视化失败: {str(e)}")
            self.log(traceback.format_exc())

    def finalize(self):
        """完成优化"""
        self.log("优化完成! 结果已保存")
        self.log("  - optimized_beam_highres.csv (优化后的高精度束流分布)")
        self.log("  - optimized_beam_lowres.csv (优化后的低精度束流分布)")
        self.log("  - beam_optimization_result.png (可视化报告)")
        self.log("  - optimization_error.png (误差变化曲线)")
        
        if self.log_file:
            self.log_file.close()

# ========主程序 ==================
def main():
    input_files = {
        "beam_traced_x_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",
        "beam_traced_y_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 80)
    print("增强边缘束流优化引擎 (ROBUST版本)".center(80))
    print("专注于半高峰宽外区域的精度提升".center(80))
    print("=" * 80)
    print("方向衰减因子(核心/边缘):")
    for dir, factors in DIRECTION_DECAY_FACTORS.items():
        print(f"  - {dir}方向: {factors['core']:.3f}/{factors['edge']:.3f}")
    print("输入文件:")
    for name, path in input_files.items():
        print(f"  - {name:55}: {path}")
    print("=" * 80)
    
    missing_files = [f"  - {name}: {path}" for name, path in input_files.items() if not os.path.exists(path)]
    
    if missing_files:
        print("警告: 以下文件不存在:")
        for msg in missing_files:
            print(msg)
        
        proceed = input("部分文件缺失，是否继续? (y/n): ").strip().lower()
        if proceed != 'y':
            print("程序终止")
            return
    
    try:
        optimizer = DirectionalBeamOptimizer(
            beam_traced_x_axis=input_files["beam_traced_x_axis"],
            beam_traced_y_axis=input_files["beam_traced_y_axis"],
            initial_guess_path=input_files["initial_beam"]
        )
        
        optimized_beam = optimizer.run_optimization(iterations=50)
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("增强边缘优化成功完成!".center(80))
        print("结果文件:")
        print("  - optimized_beam_highres.csv (高精度束流分布)")
        print("  - optimized_beam_lowres.csv (低精度束流分布)")
        print("  - beam_optimization_result.png (可视化报告)")
        print("  - optimization_error.png (误差变化曲线)")
        print("=" * 80)
        
    except Exception as e:
        print(f"优化出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()