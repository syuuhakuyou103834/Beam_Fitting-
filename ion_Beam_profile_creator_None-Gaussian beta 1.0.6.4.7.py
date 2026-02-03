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

# ============== 严格单峰性约束参数 ==============
RADIATIVE_DECAY = 0.4   # 辐射衰减因子 (值越小衰减越快)
MIN_RADIAL_SLOPE = 0.01 # 最小径向斜率约束

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

class StrictUnimodalBeamOptimizer:
    def __init__(self, beam_traced_x_axis, beam_traced_y_axis, initial_guess_path):
        """
        严格单峰性束流优化器 - 保证中心唯一极大值
        """
        # 创建日志文件
        self.log_file = open("unimodal_beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("严格单峰性束流优化引擎")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log("单峰性约束参数: RADIATIVE_DECAY=%.2f, MIN_RADIAL_SLOPE=%.3f" % 
                (RADIATIVE_DECAY, MIN_RADIAL_SLOPE))
        
        # 计算网格点数
        self.grid_points = int(GRID_BOUND * 2 / HIGH_PRECISION) + 1
        self.grid = np.linspace(-GRID_BOUND, GRID_BOUND, self.grid_points)
        self.grid_spacing = 2 * GRID_BOUND / (self.grid_points - 1)
        
        self.log(f"网格精度: {HIGH_PRECISION:.2f}mm, 点数: {self.grid_points}x{self.grid_points}")
        self.log("输入文件:")
        self.log(f"  - X截面: {beam_traced_x_axis}")
        self.log(f"  - Y截面: {beam_traced_y_axis}")
        self.log(f"  - 初始束流: {initial_guess_path}")
        self.log("=" * 30)
        
        # 优化参数
        self.opt_radius = GRID_BOUND - 1.0  # 优化半径 (mm)
        self.scan_velocity = 30.0   # 扫描速度 (mm/s)
        self.drift_sigma = 1.8      # 漂移校正标准差 (mm)
        self.smooth_sigma = 0.5     # 高斯平滑参数
        
        # 创建距离矩阵（以原点为中心）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_matrix = np.sqrt(xx**2 + yy**2)  # 距离原点距离
        
        # 加载和处理数据
        self.log("加载并预处理实验数据...")
        self.x_data = self.load_and_preprocess_data(beam_traced_x_axis)
        self.y_data = self.load_and_preprocess_data(beam_traced_y_axis)
        
        # 加载初始束流（强制中心点峰值）
        self.load_initial_beam(initial_guess_path)
        
        # 初始化优化束流（归一化）
        self.optimized_beam = self.initial_beam / self.max_val
        
        # 初始单峰性约束
        self.optimized_beam = self.enforce_strict_unimodality(self.optimized_beam)
        
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
                calibration_factor = exp_peak_val_x / center_val
                highres_beam *= calibration_factor
            
            # 强制中心为最大值
            highres_beam[grid_center, grid_center] = np.max(highres_beam)
            
            # 应用高斯平滑
            highres_beam = gaussian_filter(highres_beam, sigma=0.5)
            
            self.initial_beam = highres_beam
            self.max_val = highres_beam[grid_center, grid_center]
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
        强制的单峰性约束：确保束流分布是严格单峰的（仅在中心存在极大值）
        使用迭代辐射约束算法：
        1. 计算径向距离矩阵
        2. 按照距离环从内向外处理
        3. 每个点值都强制小于内环的值乘以衰减因子
        """
        # 计算辐向距离矩阵
        r = self.r_matrix
        
        # 生成径向距离分段
        radial_bins = np.arange(0, np.max(r), HIGH_PRECISION)
        beam_new = beam_matrix.copy()
        
        # 设置中心点约束
        center_val = beam_matrix[self.center_idx, self.center_idx]
        if center_val < np.max(beam_matrix):
            beam_new[self.center_idx, self.center_idx] = np.max(beam_matrix)
        
        # 按距离环从内向外处理
        for r_bin in radial_bins[1:]:
            # 获取当前距离环的掩膜
            mask = (r >= r_bin - HIGH_PRECISION/2) & (r < r_bin + HIGH_PRECISION/2)
            
            # 找到内层点 (当前环的内边界)
            inner_mask = (r >= r_bin - HIGH_PRECISION) & (r < r_bin - HIGH_PRECISION/2)
            inner_values = beam_matrix[inner_mask]
            
            if len(inner_values) == 0:
                continue
                
            # 确定参考值 (内层值的最大值乘以衰减因子)
            ref_val = np.max(inner_values) * RADIATIVE_DECAY
            
            # 确保当前环的点值不超过内层参考值
            beam_new[mask] = np.minimum(beam_new[mask], ref_val)
            
            # 添加最小径向梯度约束
            min_val = ref_val - MIN_RADIAL_SLOPE * HIGH_PRECISION
            beam_new[mask] = np.maximum(beam_new[mask], min_val)
        
        # 确保中心点仍为最大值
        beam_new[self.center_idx, self.center_idx] = np.max(beam_new)
        
        # 高斯平滑
        beam_new = gaussian_filter(beam_new, sigma=self.smooth_sigma)
        
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
        if center_val < np.max(beam_matrix):
            return False, 0, 1.0
        
        # 找出除中心外的局部极大值
        max_filtered = maximum_filter(beam_matrix, size=3)
        local_maxima = (beam_matrix == max_filtered) & (beam_matrix > 0.01 * center_val)
        
        # 排除中心点
        center_mask = (np.abs(self.grid) < 0.5)[:, None] & (np.abs(self.grid) < 0.5)[None, :]
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
        kernel /= np.sum(kernel)  # 归一化
        
        # 创建轮廓数组
        profile = np.zeros_like(self.grid)
        actual_beam = beam_matrix * self.max_val  # 实际刻蚀速率
        
        if axis == 'x':  # 沿Y方向扫描
            for i in range(len(self.grid)):
                # 提取当前X位置的Y线
                line = actual_beam[i, :]
                
                # 应用漂移卷积
                if len(line) >= 5:  # 确保足够长度进行卷积
                    convolved = np.convolve(line, kernel, mode='same')
                else:
                    convolved = line
                
                # 沿Y方向积分
                profile[i] = simpson(convolved, self.grid) / self.scan_velocity
                
        elif axis == 'y':  # 沿X方向扫描
            for j in range(len(self.grid)):
                # 提取当前Y位置的X线
                line = actual_beam[:, j]
                
                # 应用漂移卷积
                if len(line) >= 5:
                    convolved = np.convolve(line, kernel, mode='same')
                else:
                    convolved = line
                
                # 沿X方向积分
                profile[j] = simpson(convolved, self.grid) / self.scan_velocity
        
        # 归一化
        max_val = np.max(profile) if np.max(profile) > 0 else 1.0
        return profile / max_val

    def calculate_error(self, sim_x, sim_y):
        """计算模拟轮廓与实验数据的误差"""
        # X方向误差
        exp_x = self.x_data
        exp_x_pos = exp_x[:, 0]
        exp_x_val = exp_x[:, 1]
        
        try:
            # 实验数据归一化
            exp_x_max = np.max(exp_x_val)
            exp_x_norm = exp_x_val / exp_x_max
            
            # 模拟数据归一化
            sim_x_norm = sim_x / np.max(sim_x) if np.max(sim_x) > 0 else sim_x
            
            # 插值到相同网格
            sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_norm)
            
            # 计算均方根误差
            rmse_x = np.sqrt(np.mean((sim_x_interp - exp_x_norm)**2))
            
            # 检查单峰性违反
            is_unimodal, peak_count, max_ratio = self.validate_unimodality_profile(sim_x)
            if not is_unimodal:
                # 针对额外峰值的惩罚
                penalty = peak_count * 0.1 + max_ratio
                rmse_x += penalty
        except:
            rmse_x = 1.0
        
        # Y方向误差
        exp_y = self.y_data
        exp_y_pos = exp_y[:, 0]
        exp_y_val = exp_y[:, 1]
        
        try:
            exp_y_max = np.max(exp_y_val)
            exp_y_norm = exp_y_val / exp_y_max
            sim_y_norm = sim_y / np.max(sim_y) if np.max(sim_y) > 0 else sim_y
            sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_norm)
            rmse_y = np.sqrt(np.mean((sim_y_interp - exp_y_norm)**2))
            
            is_unimodal, peak_count, max_ratio = self.validate_unimodality_profile(sim_y)
            if not is_unimodal:
                penalty = peak_count * 0.1 + max_ratio
                rmse_y += penalty
        except:
            rmse_y = 1.0
        
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
        center_idx = np.where(np.abs(self.grid) < 0.1)[0]
        if len(center_idx) == 0:
            center_idx = self.grid_points // 2
        else:
            center_idx = center_idx[0]
        
        # 中心点是否为全局最大值
        center_val = profile[center_idx]
        if center_val < np.max(profile):
            return False, len(peaks), 1.0
        
        # 统计非中心峰
        non_center_peaks = []
        for peak_idx in peaks:
            if abs(peak_idx - center_idx) > 3:  # 排除中心附近点
                non_center_peaks.append(peak_idx)
        
        # 计算最大非中心峰值比例
        max_non_center = max(profile[non_center_peaks]) if non_center_peaks else 0
        max_ratio = max_non_center / center_val
        
        return (len(non_center_peaks) == 0), len(non_center_peaks), max_ratio

    def run_optimization(self, iterations=80):
        """运行严格单峰性优化过程"""
        start_time = time.time()
        error_history = []
        unimodal_history = []
        best_error = float('inf')
        best_beam = None
        
        # 初始错误计算
        init_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        init_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        init_error = self.calculate_error(init_sim_x, init_sim_y)
        error_history.append(init_error)
        
        # 检查初始单峰性状态
        is_unimodal, peak_count, max_ratio = self.validate_unimodality(self.optimized_beam)
        unimodal_history.append(is_unimodal)
        self.log(f"\n开始优化: 初始绝对误差={init_error:.6f} 单峰性={is_unimodal} (峰值={peak_count})")
        
        for i in range(iterations):
            self.log_progress(i+1, iterations)
            
            # 突变因子（随迭代次数减少）
            mutation_factor = 0.15 * (1 - i/iterations)
            
            # 生成变异候选
            candidate = self.optimized_beam.copy()
            mutation_mask = self.r_matrix < self.opt_radius
            
            # 在优化区域内随机突变
            mutate_points = mutation_mask & (np.random.random(self.optimized_beam.shape) < 0.5)
            mutate_values = mutation_factor * (np.random.random(np.sum(mutate_points)) - 0.5)
            candidate[mutate_points] += mutate_values
            
            # 边界约束
            candidate = np.maximum(candidate, 0)
            candidate = np.minimum(candidate, 1.0)
            
            # 应用严格单峰性约束
            candidate = self.enforce_strict_unimodality(candidate)
            
            # 评估候选
            cand_sim_x = self.simulate_etching(candidate, 'x')
            cand_sim_y = self.simulate_etching(candidate, 'y')
            cand_error = self.calculate_error(cand_sim_x, cand_sim_y)
            
            # 检查单峰性
            is_unimodal, peak_count, max_ratio = self.validate_unimodality(candidate)
            
            # 记录历史
            error_history.append(cand_error)
            unimodal_history.append(is_unimodal)
            
            # 检查改进（单峰性是硬性要求）
            improvement = best_error - cand_error
            if (cand_error < best_error) and is_unimodal:
                best_error = cand_error
                best_beam = candidate
                self.optimized_beam = candidate
                
                # 每10次迭代记录日志
                if i % 10 == 0:
                    self.log(f"Iter {i}: 误差={best_error:.6f} 单峰性={is_unimodal} (峰值={peak_count})")
        
        # 结束优化
        sys.stdout.write("\n")
        elapsed_time = time.time() - start_time
        
        # 最终评估
        final_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        final_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        final_error = self.calculate_error(final_sim_x, final_sim_y)
        
        # 保存输出
        np.savetxt("optimized_highres_beam.csv", self.optimized_beam * self.max_val, delimiter=",")
        self.log(f"\n优化完成! 耗时: {elapsed_time:.1f}秒")
        self.log(f"最终绝对误差: {final_error:.6f} (初始={init_error:.6f})")
        
        # 可视化结果
        self.visualize_results(
            init_sim_x, init_sim_y, 
            final_sim_x, final_sim_y, 
            error_history, unimodal_history
        )
        
        return self.optimized_beam * self.max_val

    def visualize_results(self, init_sim_x, init_sim_y, final_sim_x, final_sim_y, errors, unimodal):
        """创建可视化报告"""
        try:
            fig = plt.figure(figsize=(16, 18))
            fig.suptitle("严格单峰性束流优化结果", fontsize=20, y=0.98)
            
            # 网格布局 (3行2列)
            gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8], width_ratios=[1, 1])
            
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
                ax2.set_title("优化后束流分布 (强制单峰)", fontsize=14)
                ax2.set_xlabel("X (mm)")
                ax2.set_ylabel("Y (mm)")
                
                # 检查单峰性
                is_unimodal, peak_count, max_ratio = self.validate_unimodality(self.optimized_beam)
                status = "通过" if is_unimodal else f"违反 ({peak_count}个峰)"
                ax2.text(0.5, -0.15, f"单峰性检查: {status}", 
                        transform=ax2.transAxes, fontsize=12, ha='center',
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
            
            # 标记峰值位置
            peaks = np.where(np.diff(np.sign(np.diff(final_sim_x))) < 0)[0] + 1
            if peaks.size > 0:
                ax4.plot(self.grid[peaks], final_sim_x[peaks], 'go', markersize=8, 
                        fillstyle='none', label='峰值点')
            
            # 验证单峰性
            is_single_peak, peak_count, max_ratio = self.validate_unimodality_profile(final_sim_x)
            status = "单峰" if is_single_peak else "多峰"
            ax4.text(0.95, 0.95, f"单峰性: {status}", 
                    transform=ax4.transAxes, fontsize=12, ha='right',
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
            
            # 标记中心点
            ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            # 保存结果
            plt.savefig("strict_unimodal_beam_optimization.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.log("结果可视化已保存")
            
        except Exception as e:
            self.log(f"可视化失败: {str(e)}")
    
    def finalize(self):
        """结束优化并关闭日志"""
        self.log("优化完成! 结果已保存")
        self.log("  - optimized_highres_beam.csv")
        self.log("  - strict_unimodal_beam_optimization.png")
        self.log("  - unimodal_beam_optimization_log.txt")
        
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
    print("严格单峰性束流优化引擎".center(80))
    print("优化目标: 确保束流卷积后只有一个中心峰值".center(80))
    print("=" * 80)
    print("关键特性:")
    print("  1. 严格单峰性约束（径向衰减强制）")
    print("  2. 峰值位置固定于坐标原点")
    print("  3. 自动检测并惩罚额外峰值")
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
        optimizer = StrictUnimodalBeamOptimizer(
            beam_traced_x_axis=input_files["beam_traced_x_axis"],
            beam_traced_y_axis=input_files["beam_traced_y_axis"],
            initial_guess_path=input_files["initial_beam"]
        )
        
        # 运行优化
        optimized_beam = optimizer.run_optimization(iterations=120)
        
        # 生成报告
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("优化成功完成!".center(80))
        print("结果文件:")
        print("  - optimized_highres_beam.csv (0.1mm精度束流分布)")
        print("  - strict_unimodal_beam_optimization.png (可视化报告)")
        print("  - unimodal_beam_optimization_log.txt (详细日志)")
        print("=" * 80)
        
    except Exception as e:
        print(f"优化出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
