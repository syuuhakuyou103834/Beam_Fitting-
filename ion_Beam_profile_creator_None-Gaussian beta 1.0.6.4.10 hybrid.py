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
    """将束流分布从高分辨率降采样到低分辨率"""
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
    
    center_idx = target_points // 2
    peak_val = np.max(beam_matrix)
    if peak_val > 0:
        center_val = np.mean(sampled_beam[center_idx-1:center_idx+2, center_idx-1:center_idx+2])
        if center_val < 0.5 * peak_val:
            center_val = np.max(sampled_beam)
        if center_val > 1e-6:
            scaling = peak_val / center_val
            sampled_beam *= scaling
    
    if target_resolution > src_resolution:
        smooth_sigma = 0.5
        sampled_beam = gaussian_filter(sampled_beam, sigma=smooth_sigma)
    
    return sampled_beam

# ============== 组合优化参数 ==============
HIGH_PRECISION = 0.1
GRID_BOUND = 15.0
RING_STEP = 0.1
INNER_RADIUS = 2.5  # 内外环分界半径

# 方向衰减参数 - 内环（0-10mm）
INNER_DECAY_FACTORS = {
    'x+': 0.80, 'x-': 0.80, 
    'y+': 0.80, 'y-': 0.80
}

# 方向衰减参数 - 外环（>10mm）
OUTER_DECAY_FACTORS = {
    'x+': 0.80, 'x-': 0.80, 
    'y+': 0.80, 'y-': 0.80
}

MIN_RADIAL_SLOPE_INNER = 0.003  # 内环最小径向斜率
MIN_RADIAL_SLOPE_OUTER = 0.0008  # 外环最小径向斜率（减小约束）
SMALL_OFFSET = 0.001

class HybridBeamOptimizer:
    def __init__(self, beam_traced_x_axis, beam_traced_y_axis, initial_guess_path):
        self.log_file = open("hybrid_beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("混合束流优化引擎（内环方向性衰减 + 外环自由优化）")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"网格精度: {HIGH_PRECISION:.2f}mm, 环状步长: {RING_STEP:.2f}mm")
        self.log(f"内环半径: {INNER_RADIUS}mm")
        self.log("输入文件:")
        self.log(f"  - X截面: {beam_traced_x_axis}")
        self.log(f"  - Y截面: {beam_traced_y_axis}")
        self.log(f"  - 初始束流: {initial_guess_path}")
        self.log("=" * 60)
        
        self.grid_points = int(GRID_BOUND * 2 / HIGH_PRECISION) + 1
        self.grid = np.linspace(-GRID_BOUND, GRID_BOUND, self.grid_points)
        self.grid_spacing = 2 * GRID_BOUND / (self.grid_points - 1)
        
        self.opt_radius = GRID_BOUND - 1.0
        self.scan_velocity = 30.0
        self.drift_sigma = 1.8
        self.smooth_sigma = 0.5
        
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_matrix = np.sqrt(xx**2 + yy**2)
        self.x_matrix = xx
        self.y_matrix = yy
        
        self.log("加载并预处理实验数据...")
        self.x_data = self.load_and_preprocess_data(beam_traced_x_axis)
        self.y_data = self.load_and_preprocess_data(beam_traced_y_axis)
        
        self.load_initial_beam(initial_guess_path)
        
        self.optimized_beam = self.initial_beam / (self.max_val + SMALL_OFFSET)
        self.center_idx = self.grid_points // 2
        self.center_x = self.grid[self.center_idx]
        self.center_y = self.grid[self.center_idx]
        
        self.history = {
            "iteration": [0],
            "abs_error": [],
            "center_error": [],
            "edge_error": []
        }
        
        # 创建内外环掩码（用于不同的优化策略）
        self.inner_mask = self.r_matrix <= INNER_RADIUS
        self.outer_mask = (self.r_matrix > INNER_RADIUS) & (self.r_matrix <= GRID_BOUND)
        self.log(f"内环点数: {np.sum(self.inner_mask)}, 外环点数: {np.sum(self.outer_mask)}")

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_file.write(log_entry + "\n")
        self.log_file.flush()
        
    def log_progress(self, current, total):
        progress = int(100 * current/total)
        bar = '=' * (progress//2)
        sys.stdout.write(f"\r[优化进度] |{bar:{50}}| {progress}%")
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    def load_and_preprocess_data(self, file_path):
        self.log(f"加载实验数据: {file_path}")
        
        try:
            data = np.loadtxt(file_path, delimiter=",")
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("数据格式不正确，应为N行2列")
                
            peak_idx = np.argmax(data[:, 1])
            peak_pos = data[peak_idx, 0]
            shift_value = -peak_pos
            data[:, 0] += shift_value
            
            interpolator = interp1d(
                data[:, 0], data[:, 1], 
                kind='cubic', 
                bounds_error=False, 
                fill_value=0.0
            )
            
            new_values = interpolator(self.grid)
            new_values = gaussian_filter(new_values, sigma=0.8)
            new_values = np.maximum(new_values, 0)
            
            new_data = np.column_stack((self.grid, new_values))
            new_peak_idx = np.argmax(new_values)
            self.log(f"  平移后峰值位置: {self.grid[new_peak_idx]:.2f}mm | 强度: {new_values[new_peak_idx]:.2f}")
            
            return new_data
            
        except Exception as e:
            self.log(f"加载数据失败: {str(e)}")
            return np.column_stack((self.grid, np.zeros_like(self.grid)))

    def load_initial_beam(self, file_path):
        try:
            initial_data = np.genfromtxt(file_path, delimiter=",")
            target_points = int(GRID_BOUND * 2 / 1.0) + 1
            if initial_data.shape[0] != target_points or initial_data.shape[1] != target_points:
                self.log(f"调整初始束流尺寸: {initial_data.shape} -> ({target_points},{target_points})")
                new_data = np.zeros((target_points, target_points))
                min_size = min(initial_data.shape[0], target_points)
                new_data[:min_size, :min_size] = initial_data[:min_size, :min_size]
                initial_data = new_data
            
            lowres_grid = np.linspace(-GRID_BOUND, GRID_BOUND, target_points)
            interpolator = RegularGridInterpolator(
                (lowres_grid, lowres_grid),
                initial_data,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            xx, yy = np.meshgrid(self.grid, self.grid, indexing='ij')
            points = np.stack((xx, yy), axis=-1)
            highres_beam = interpolator(points)
            
            exp_peak_val_x = np.max(self.x_data[:, 1])
            grid_center = self.grid_points // 2
            center_val = np.mean(highres_beam[grid_center-1:grid_center+2, grid_center-1:grid_center+2])
            
            if center_val > 1e-6:
                calibration_factor = (exp_peak_val_x) / (center_val + SMALL_OFFSET)
                highres_beam *= calibration_factor
            
            center_val = np.max(highres_beam) + SMALL_OFFSET
            highres_beam[grid_center, grid_center] = center_val
            highres_beam = gaussian_filter(highres_beam, sigma=0.5)
            
            self.initial_beam = highres_beam
            self.max_val = center_val
            self.center_idx = grid_center
            
            self.log(f"加载成功: 中心值={self.max_val:.2f} nm/s, 尺寸={self.initial_beam.shape}")
            
        except Exception as e:
            self.log(f"加载初始束流失败: {str(e)}")
            r = np.sqrt(self.grid[:, None]**2 + self.grid[None, :]**2)
            self.initial_beam = 100 * np.exp(-r**2 / (2*5.0**2))
            self.max_val = np.max(self.initial_beam)
            self.center_idx = self.grid_points // 2
    
    def enforce_hybrid_unimodality(self, beam_matrix):
        """混合单峰性约束：内环使用方向性衰减，外环使用较弱约束"""
        r = self.r_matrix
        radial_bins = np.arange(0, np.max(r), RING_STEP)
        beam_new = beam_matrix.copy()
        
        center_val = beam_matrix[self.center_idx, self.center_idx]
        if center_val < np.max(beam_matrix):
            beam_new[self.center_idx, self.center_idx] = np.max(beam_matrix)
        
        for r_bin in radial_bins[1:]:
            # 根据半径选择不同的参数
            if r_bin <= INNER_RADIUS:
                decay_factors = INNER_DECAY_FACTORS
                min_slope = MIN_RADIAL_SLOPE_INNER
            else:
                decay_factors = OUTER_DECAY_FACTORS
                min_slope = MIN_RADIAL_SLOPE_OUTER
            
            inner_mask = (r >= r_bin - RING_STEP) & (r < r_bin - RING_STEP/2)
            inner_values = beam_matrix[inner_mask]
            
            if len(inner_values) == 0:
                continue
                
            mask = (r >= r_bin - RING_STEP/2) & (r < r_bin + RING_STEP/2)
            indices = np.where(mask)
            
            for idx in range(len(indices[0])):
                i, j = indices[0][idx], indices[1][idx]
                x = self.x_matrix[i, j]
                y = self.y_matrix[i, j]
                
                if abs(x) > abs(y):
                    decay_factor = decay_factors['x+'] if x > 0 else decay_factors['x-']
                else:
                    decay_factor = decay_factors['y+'] if y > 0 else decay_factors['y-']
                
                ref_val = np.max(inner_values) * decay_factor
                current_val = beam_matrix[i, j]
                
                if current_val > ref_val:
                    # 内环约束更强，外环约束更弱
                    weight = 0.7 if r_bin <= INNER_RADIUS else 0.4
                    beam_new[i, j] = weight * ref_val + (1 - weight) * current_val
                
                radial_gate = decay_factor * (np.max(inner_values) - min_slope * self.r_matrix[i, j])
                if beam_new[i, j] < radial_gate:
                    beam_new[i, j] = radial_gate
        
        current_max = np.max(beam_new)
        beam_new[self.center_idx, self.center_idx] = current_max + SMALL_OFFSET
        beam_new = gaussian_filter(beam_new, sigma=0.3)
        return beam_new

    def validate_unimodality(self, beam_matrix):
        center_val = beam_matrix[self.center_idx, self.center_idx]
        center_mask = (np.abs(self.x_matrix) < 0.5) & (np.abs(self.y_matrix) < 0.5)
        
        if np.max(beam_matrix[~center_mask]) + SMALL_OFFSET > center_val:
            return False, 0, 1.0
        
        max_filtered = maximum_filter(beam_matrix, size=3)
        local_maxima = (beam_matrix == max_filtered) & (beam_matrix > 0.01 * center_val)
        local_maxima[center_mask] = False
        
        peak_count = np.sum(local_maxima)
        if peak_count > 0:
            max_non_center = np.max(beam_matrix[local_maxima])
            max_ratio = max_non_center / center_val
        else:
            max_ratio = 0
        
        is_unimodal = (peak_count == 0)
        return is_unimodal, peak_count, max_ratio

    def simulate_etching(self, beam_matrix, axis='x'):
        kernel_radius = min(100, int(3 * self.drift_sigma / self.grid_spacing))
        kernel_points = 2 * kernel_radius + 1
        kernel_range = np.linspace(-kernel_radius, kernel_radius, kernel_points) * self.grid_spacing
        kernel = np.exp(-0.5 * (kernel_range/self.drift_sigma)**2)
        kernel /= np.sum(kernel) + SMALL_OFFSET
        
        profile = np.zeros(len(self.grid))
        actual_beam = beam_matrix * self.max_val
        active_mask = (self.r_matrix < self.opt_radius) & (actual_beam > 0.01 * np.max(actual_beam))
        
        if axis == 'x':
            for i in range(len(self.grid)):
                if np.any(active_mask[i, :]):
                    line = actual_beam[i, :]
                    if len(line) >= 5:
                        profile[i] = simpson(line, self.grid) / self.scan_velocity
        elif axis == 'y':
            for j in range(len(self.grid)):
                if np.any(active_mask[:, j]):
                    line = actual_beam[:, j]
                    if len(line) >= 5:
                        profile[j] = simpson(line, self.grid) / self.scan_velocity
        
        max_val = np.max(profile) if np.max(profile) > 0 else 1.0
        return profile / (max_val + SMALL_OFFSET)

    def calculate_error(self, sim_x, sim_y):
        center_error = 0
        edge_error = 0
        
        # X方向误差
        try:
            exp_x = self.x_data
            exp_x_pos = exp_x[:, 0]
            exp_x_val = exp_x[:, 1]
            
            exp_x_max = np.max(exp_x_val)
            exp_x_norm = exp_x_val / (exp_x_max + SMALL_OFFSET)
            
            sim_x_norm = sim_x / (np.max(sim_x) + SMALL_OFFSET)
            sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_norm)
            
            # 分离中心区域和边缘区域的误差
            center_mask = np.abs(exp_x_pos) <= INNER_RADIUS
            edge_mask = np.abs(exp_x_pos) > INNER_RADIUS
            valid_mask = exp_x_val > 0.5 * exp_x_max
            
            if np.any(center_mask & valid_mask):
                center_error += 0.4 * np.sqrt(np.mean((sim_x_interp[center_mask & valid_mask] - exp_x_norm[center_mask & valid_mask])**2))
            
            if np.any(edge_mask & valid_mask):
                edge_error += 0.6 * np.sqrt(np.mean((sim_x_interp[edge_mask & valid_mask] - exp_x_norm[edge_mask & valid_mask])**2))
        except:
            pass
        
        # Y方向误差
        try:
            exp_y = self.y_data
            exp_y_pos = exp_y[:, 0]
            exp_y_val = exp_y[:, 1]
            
            exp_y_max = np.max(exp_y_val)
            exp_y_norm = exp_y_val / (exp_y_max + SMALL_OFFSET)
            sim_y_norm = sim_y / (np.max(sim_y) + SMALL_OFFSET)
            sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_norm)
            
            center_mask_y = np.abs(exp_y_pos) <= INNER_RADIUS
            edge_mask_y = np.abs(exp_y_pos) > INNER_RADIUS
            valid_mask = exp_y_val > 0.5 * exp_y_max
            
            if np.any(center_mask_y & valid_mask):
                center_error += 0.3 * np.sqrt(np.mean((sim_y_interp[center_mask_y & valid_mask] - exp_y_norm[center_mask_y & valid_mask])**2))
            
            if np.any(edge_mask_y & valid_mask):
                edge_error += 0.7 * np.sqrt(np.mean((sim_y_interp[edge_mask_y & valid_mask] - exp_y_norm[edge_mask_y & valid_mask])**2))
        except:
            pass
        
        combined_error = center_error + edge_error
        return combined_error, center_error, edge_error

    def run_hybrid_optimization(self, inner_iter=70, outer_iter=50):
        self.log(f"开始混合优化：内环{inner_iter}次，外环{outer_iter}次")
        self.log(f"内环参数: X+:{INNER_DECAY_FACTORS['x+']:.3f}, Y+:{INNER_DECAY_FACTORS['y+']:.3f}")
        self.log(f"外环参数: X+:{OUTER_DECAY_FACTORS['x+']:.3f}, Y+:{OUTER_DECAY_FACTORS['y+']:.3f}")
        
        start_time = time.time()
        self.optimized_beam = self.initial_beam / self.max_val
        
        # 初始评估
        init_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        init_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        init_error, center_err, edge_err = self.calculate_error(init_sim_x, init_sim_y)
        best_error = init_error
        best_beam = self.optimized_beam.copy()
        
        self.history["abs_error"].append(init_error)
        self.history["center_error"].append(center_err)
        self.history["edge_error"].append(edge_err)
        self.log(f"初始误差: 总={init_error:.4f}, 中心={center_err:.4f}, 边缘={edge_err:.4f}")
        
        # 第一阶段：密集优化内环（0-10mm）
        self.log(">>> 第一阶段：优化内环")
        max_mutation_mag = 0.04
        
        for i in range(inner_iter):
            self.log_progress(i+1, inner_iter+outer_iter)
            mutation_factor = max_mutation_mag * (1 - i/inner_iter)
            
            # 只在内环区域变异
            mutation_mask = self.inner_mask.copy()
            current_beam = best_beam.copy()
            
            mutation = mutation_factor * np.random.randn(*current_beam.shape)
            current_beam[mutation_mask] += mutation[mutation_mask]
            current_beam = np.maximum(current_beam, 0)
            current_beam = np.minimum(current_beam, 1.0)
            
            # 应用混合约束
            candidate = self.enforce_hybrid_unimodality(current_beam)
            
            # 评估候选
            cand_sim_x = self.simulate_etching(candidate, 'x')
            cand_sim_y = self.simulate_etching(candidate, 'y')
            cand_error, cand_center_err, cand_edge_err = self.calculate_error(cand_sim_x, cand_sim_y)
            
            improvement = best_error - cand_error
            if improvement > 0:
                best_error = cand_error
                best_beam = candidate
                self.log(f"[内环 {i+1}/{inner_iter}] 改进! 总误差={cand_error:.6f} 中心={cand_center_err:.6f}")
            
            self.history["abs_error"].append(cand_error)
            self.history["center_error"].append(cand_center_err)
            self.history["edge_error"].append(cand_edge_err)
        
        # 第二阶段：优化外环（>10mm）
        self.log(">>> 第二阶段：优化外环")
        
        for i in range(outer_iter):
            self.log_progress(inner_iter+i+1, inner_iter+outer_iter)
            mutation_factor = max_mutation_mag * (0.8 - i/outer_iter*0.3)
            
            # 主要在外环区域变异
            mutation_mask = self.outer_mask.copy()
            current_beam = best_beam.copy()
            
            # 更强的变异以适应边缘
            mutation = mutation_factor * np.random.randn(*current_beam.shape)
            mutation_mask = mutation_mask & (current_beam > 0.01)
            current_beam[mutation_mask] += mutation[mutation_mask]
            current_beam = np.maximum(current_beam, 0)
            current_beam = np.minimum(current_beam, 1.0)
            
            # 应用较弱的约束
            candidate = self.enforce_hybrid_unimodality(current_beam)
            
            # 评估候选
            cand_sim_x = self.simulate_etching(candidate, 'x')
            cand_sim_y = self.simulate_etching(candidate, 'y')
            cand_error, cand_center_err, cand_edge_err = self.calculate_error(cand_sim_x, cand_sim_y)
            
            improvement = best_error - cand_error
            if improvement > 0 or (improvement > -0.01 and cand_edge_err < edge_err):
                best_error = cand_error
                edge_err = cand_edge_err
                best_beam = candidate
                self.log(f"[外环 {i+1}/{outer_iter}] 改进! 总误差={cand_error:.6f} 边缘={cand_edge_err:.6f}")
        
        # 最终优化结果
        sys.stdout.write("\n")
        self.optimized_beam = best_beam
        final_sim_x = self.simulate_etching(self.optimized_beam, 'x')
        final_sim_y = self.simulate_etching(self.optimized_beam, 'y')
        final_error, final_center_err, final_edge_err = self.calculate_error(final_sim_x, final_sim_y)
        
        output_beam_highres = self.optimized_beam * self.max_val
        np.savetxt("hybrid_optimized_beam_highres.csv", output_beam_highres, delimiter=",")
        output_beam_lowres = down_sample_beam(output_beam_highres, HIGH_PRECISION, 1.0)
        np.savetxt("hybrid_optimized_beam_lowres.csv", output_beam_lowres, delimiter=",")
        
        # 保存参数
        with open("hybrid_parameters.txt", "w") as f:
            f.write(f"# 混合束流优化参数\n")
            f.write(f"# 内环边界半径: {INNER_RADIUS}mm\n")
            f.write(f"# 内环环状步长: {RING_STEP}mm\n")
            f.write(f"# 内环方向衰减: X+:{INNER_DECAY_FACTORS['x+']}, X-:{INNER_DECAY_FACTORS['x-']}, Y+:{INNER_DECAY_FACTORS['y+']}, Y-:{INNER_DECAY_FACTORS['y-']}\n")
            f.write(f"# 外环方向衰减: X+:{OUTER_DECAY_FACTORS['x+']}, X-:{OUTER_DECAY_FACTORS['x-']}, Y+:{OUTER_DECAY_FACTORS['y+']}, Y-:{OUTER_DECAY_FACTORS['y-']}\n")
            f.write(f"# 初始误差: {init_error:.6f} (中心={center_err:.6f}, 边缘={edge_err:.6f})\n")
            f.write(f"# 最终误差: {final_error:.6f} (中心={final_center_err:.6f}, 边缘={final_edge_err:.6f})\n")
        
        elapsed_time = time.time() - start_time
        self.log(f"\n优化完成! 耗时: {elapsed_time:.1f}秒")
        self.log(f"初始误差: {init_error:.6f} → 最终误差: {final_error:.6f}")
        self.log(f"中心误差改进: {center_err-final_center_err:.6f}, 边缘误差改进: {edge_err-final_edge_err:.6f}")
        self.log(f"已保存高精度束流分布: hybrid_optimized_beam_highres.csv")
        self.log(f"已保存低精度束流分布: hybrid_optimized_beam_lowres.csv")
        
        # 可视化结果
        self.visualize_results(init_sim_x, init_sim_y, final_sim_x, final_sim_y)
        return output_beam_highres

    def visualize_results(self, init_sim_x, init_sim_y, final_sim_x, final_sim_y):
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle("混合束流优化结果", fontsize=20, y=0.98)
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
            
            # 1. 优化前后束流分布对比
            ax1 = plt.subplot(gs[0, 0])
            init_norm = self.initial_beam / self.max_val
            im1 = ax1.imshow(init_norm, cmap='viridis', extent=[-15, 15, -15, 15], origin='lower')
            plt.colorbar(im1, ax=ax1)
            ax1.set_title("初始束流分布")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            ax1.plot(INNER_RADIUS * np.cos(np.linspace(0, 2*np.pi, 100)), 
                     INNER_RADIUS * np.sin(np.linspace(0, 2*np.pi, 100)), 
                     'r--', alpha=0.7, label="内环边界")
            ax1.legend(loc='upper right')
            
            ax2 = plt.subplot(gs[0, 1])
            final_norm = self.optimized_beam
            im2 = ax2.imshow(final_norm, cmap='viridis', extent=[-15, 15, -15, 15], origin='lower')
            plt.colorbar(im2, ax=ax2)
            ax2.set_title("优化后束流分布")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.plot(INNER_RADIUS * np.cos(np.linspace(0, 2*np.pi, 100)), 
                     INNER_RADIUS * np.sin(np.linspace(0, 2*np.pi, 100)), 
                     'r--', alpha=0.7, label="内环边界")
            
            # 2. X方向拟合对比
            ax3 = plt.subplot(gs[1, 0])
            ax3.set_title("X方向截面拟合")
            ax3.plot(self.x_data[:, 0], self.x_data[:, 1]/np.max(self.x_data[:, 1]), 
                    'bo', alpha=0.5, markersize=4, label='实验数据')
            ax3.plot(self.grid, init_sim_x, 'r-', linewidth=1.5, label='初始拟合')
            ax3.plot(self.grid, final_sim_x, 'g-', linewidth=2, label='优化拟合')
            ax3.axvline(x=INNER_RADIUS, color='r', linestyle='--', alpha=0.5)
            ax3.axvline(x=-INNER_RADIUS, color='r', linestyle='--', alpha=0.5)
            ax3.text(INNER_RADIUS+0.5, 0.9, '外环区域', color='r')
            ax3.text(-INNER_RADIUS-2.5, 0.9, '外环区域', color='r')
            ax3.set_xlabel("X位置 (mm)")
            ax3.set_ylabel("归一化深度")
            ax3.legend(loc='best')
            ax3.grid(True, linestyle='--', alpha=0.6)
            
            # 3. Y方向拟合对比
            ax4 = plt.subplot(gs[1, 1])
            ax4.set_title("Y方向截面拟合")
            ax4.plot(self.y_data[:, 0], self.y_data[:, 1]/np.max(self.y_data[:, 1]), 
                    'go', alpha=0.5, markersize=4, label='实验数据')
            ax4.plot(self.grid, init_sim_y, 'm-', linewidth=1.5, label='初始拟合')
            ax4.plot(self.grid, final_sim_y, 'c-', linewidth=2, label='优化拟合')
            ax4.axvline(x=INNER_RADIUS, color='r', linestyle='--', alpha=0.5)
            ax4.axvline(x=-INNER_RADIUS, color='r', linestyle='--', alpha=0.5)
            ax4.text(INNER_RADIUS+0.5, 0.9, '外环区域', color='r')
            ax4.text(-INNER_RADIUS-2.5, 0.9, '外环区域', color='r')
            ax4.set_xlabel("Y位置 (mm)")
            ax4.set_ylabel("归一化深度")
            ax4.legend(loc='best')
            ax4.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.subplots_adjust(hspace=0.25, wspace=0.25)
            plt.savefig("hybrid_beam_optimization_result.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.log("结果可视化已保存")
            
        except Exception as e:
            self.log(f"可视化失败: {str(e)}")
            self.log(traceback.format_exc())
    
    def finalize(self):
        self.log("优化完成! 结果已保存:")
        self.log("  - hybrid_optimized_beam_highres.csv")
        self.log("  - hybrid_optimized_beam_lowres.csv")
        self.log("  - hybrid_parameters.txt")
        self.log("  - hybrid_beam_optimization_result.png")
        if self.log_file:
            self.log_file.close()

# ========主程序 ==================
def main():
    # 输入文件路径
    input_files = {
        "beam_traced_x_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",
        "beam_traced_y_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 80)
    print("混合束流优化引擎（内环方向性衰减 + 外环自由优化）".center(80))
    print(f"内环: {INNER_RADIUS}mm, 外环: {INNER_RADIUS}-{GRID_BOUND}mm".center(80))
    print("=" * 30)
    print("内环方向衰减参数:")
    print(f"  - X+方向: {INNER_DECAY_FACTORS['x+']:.3f}")
    print(f"  - X-方向: {INNER_DECAY_FACTORS['x-']:.3f}")
    print(f"  - Y+方向: {INNER_DECAY_FACTORS['y+']:.3f}")
    print(f"  - Y-方向: {INNER_DECAY_FACTORS['y-']:.3f}")
    print("外环方向衰减参数:")
    print(f"  - X+方向: {OUTER_DECAY_FACTORS['x+']:.3f}")
    print(f"  - X-方向: {OUTER_DECAY_FACTORS['x-']:.3f}")
    print(f"  - Y+方向: {OUTER_DECAY_FACTORS['y+']:.3f}")
    print(f"  - Y-方向: {OUTER_DECAY_FACTORS['y-']:.3f}")
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
        proceed = input("部分文件缺失，是否继续? (y/n): ").strip().lower()
        if proceed != 'y':
            print("程序终止")
            return
    
    try:
        optimizer = HybridBeamOptimizer(
            beam_traced_x_axis=input_files["beam_traced_x_axis"],
            beam_traced_y_axis=input_files["beam_traced_y_axis"],
            initial_guess_path=input_files["initial_beam"]
        )
        optimized_beam = optimizer.run_hybrid_optimization()
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("混合优化成功完成!".center(80))
        print("结果文件:")
        print("  - hybrid_optimized_beam_highres.csv")
        print("  - hybrid_optimized_beam_lowres.csv")
        print("  - hybrid_parameters.txt")
        print("  - hybrid_beam_optimization_result.png")
        print("=" * 80)
    except Exception as e:
        print(f"优化出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
