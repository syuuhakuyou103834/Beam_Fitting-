import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter

# ====================== 修复中文字体支持 ======================
def setup_plotting():
    """配置绘图环境，解决中文字体问题"""
    plt.rcParams.update({
        'font.sans-serif': ['SimHei', 'DejaVu Sans', 'Microsoft YaHei', 'Arial Unicode MS'],
        'axes.unicode_minus': False,
        'figure.dpi': 150,
        'savefig.dpi': 150
    })
    
setup_plotting()

class BeamEfficiencyOptimizer:
    def __init__(self, beam_traced_x_axis, beam_traced_y_axis, initial_guess_path, grid_bound=15.0, grid_points=31):
        """初始化优化器，使用梯度下降方法"""
        # 创建日志文件（UTF-8编码解决乱码问题）
        log_time = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"beam_optimization_log_{log_time}.txt", "w", encoding="utf-8")
        self.log("带梯度下降的离子束刻蚀效率优化引擎启动")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"输入文件说明:")
        self.log(f" - 离子束沿X轴移动时的Y方向截面数据: {beam_traced_x_axis}")
        self.log(f" - 离子束沿Y轴移动时的X方向截面数据: {beam_traced_y_axis}")
        self.log("=" * 30)

        # 网格系统
        self.grid_bound = grid_bound
        self.grid_points = grid_points
        self.grid = np.linspace(-grid_bound, grid_bound, grid_points)
        self.grid_spacing = 2 * grid_bound / (grid_points - 1)
        
        
        # 加载初始猜测
        self.load_initial_beam(initial_guess_path)
        
        # 加载实验数据
        self.beam_traced_x_axis = self.load_experimental_data(beam_traced_x_axis)  # 沿X轴移动 (测量Y截面)
        self.beam_traced_y_axis = self.load_experimental_data(beam_traced_y_axis)  # 沿Y轴移动 (测量X截面)
        
        
        # 优化参数
        self.scan_velocity = 30.0  # 扫描速度 (mm/s)
        self.opt_radius = 10.0  # 优化半径 (mm)
        self.learning_rate = 0.5  # 初始学习率
        
        # 创建优化掩膜
        self.create_optimization_mask()
        
        # 历史记录
        self.history = {
            "iteration": [],
            "abs_error": [],
            "rel_err_x": [],
            "rel_err_y": [],
            "learning_rate": [],
            "improvement": []
        }

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
            if self.initial_beam.shape != (self.grid_points, self.grid_points):
                self.log(f"警告: 初始束流尺寸应为{self.grid_points}x{self.grid_points}，实际为{self.initial_beam.shape}")
                # 尝试调整大小
                from scipy.interpolate import RectBivariateSpline
                original_grid = np.linspace(-15, 15, self.initial_beam.shape[0])
                target_grid = np.linspace(-15, 15, self.grid_points)
                spline = RectBivariateSpline(original_grid, original_grid, self.initial_beam)
                self.initial_beam = spline(target_grid, target_grid)
                self.log(f"已调整大小为 {self.initial_beam.shape}")
                
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
        """加载实验轮廓数据 - 增强健壮性"""
        self.log(f"加载实验数据: {file_path}")
        try:
            # 自动检测CSV格式
            data = np.genfromtxt(file_path, delimiter=',', encoding='utf-8')
            
            # 确保数据有位置和值两列
            if data.ndim != 2 or data.shape[1] != 2:
                # 尝试处理单列数据
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        data = []
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    x = float(parts[0])
                                    y = float(parts[1])
                                    data.append([x, y])
                                except:
                                    continue
                        data = np.array(data)
            else:
                # 位置和值可能被交换（值在前，位置在后）
                if np.max(np.abs(data[:, 0])) > 20:  # 位置通常不会大于20
                    data = data[:, [1, 0]]
            
            if data.shape[1] != 2:
                raise ValueError(f"文件 {file_path} 格式不正确，期待两列数据")
                
            # 按位置排序
            data = data[data[:, 0].argsort()]
            
            # 检查位置范围是否匹配网格
            if np.min(data[:, 0]) < -15 or np.max(data[:, 0]) > 15:
                self.log(f"警告: 数据位置范围[{np.min(data[:, 0]):.2f}, {np.max(data[:, 0]):.2f}]超出[-15, 15]网格范围")
            
            self.log(f"加载数据点: {len(data)} 个, 位置范围: [{data[0,0]:.2f}, {data[-1,0]:.2f}] mm")
            return data
        except Exception as e:
            self.log(f"加载失败: {str(e)}")
            self.log("尝试使用默认值...")
            return np.column_stack((self.grid, np.zeros_like(self.grid)))

    def create_optimization_mask(self):
        """创建优化区域掩膜"""
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.distance_from_center = np.sqrt(xx**2 + yy**2)
        self.optimization_mask = (self.distance_from_center <= self.opt_radius)
        self.log(f"优化区域包含 {np.sum(self.optimization_mask)} 个点 (半径 ≤ {self.opt_radius}mm)")

    def create_interpolator(self, beam_matrix):
        """创建双线性插值器"""
        return RegularGridInterpolator(
            (self.grid, self.grid),
            beam_matrix * self.max_val,  # 反归一化
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )

    def simulate_etching(self, beam_matrix, direction):
        """
        模拟指定方向的刻蚀轮廓
        
        方向定义:
        direction = "x": 束沿Y轴方向移动 (测量X方向轮廓)
        direction = "y": 束沿X轴方向移动 (测量Y方向轮廓)
        """
        interpolator = self.create_interpolator(beam_matrix)
        profile = np.zeros_like(self.grid)
        
        if direction == "x":  # 束沿Y轴移动 -> 测量X方向轮廓
            for j in range(len(self.grid)):
                y_pos = self.grid[j]
                path_points = np.column_stack((self.grid, np.full_like(self.grid, y_pos)))
                etch_rates = interpolator(path_points)
                profile[j] = trapezoid(etch_rates, self.grid)
        else:  # direction == "y": 束沿X轴移动 -> 测量Y方向轮廓
            for i in range(len(self.grid)):
                x_pos = self.grid[i]
                path_points = np.column_stack((np.full_like(self.grid, x_pos), self.grid))
                etch_rates = interpolator(path_points)
                profile[i] = trapezoid(etch_rates, self.grid)
        
        # 归一化
        max_profile = np.max(profile)
        return profile / max_profile if max_profile > 0 else np.zeros_like(profile)

    def calculate_error(self, sim_x_scan, sim_y_scan):
        """
        计算模拟结果与实验结果的误差
        - sim_x_scan: 模拟的束沿Y方向移动时的X方向轮廓
        - sim_y_scan: 模拟的束沿X方向移动时的Y方向轮廓
        """
        # 处理束沿Y移动时的X方向轮廓
        exp_x_data = self.beam_traced_y_axis  # 束沿Y移动时测量的X方向截面
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        
        # 归一化实验数据 - 防止除零
        exp_x_max = np.max(exp_x_val)
        exp_x_norm = exp_x_val / exp_x_max if exp_x_max > 1e-5 else np.zeros_like(exp_x_val)
        
        # 插值模拟数据到实验点位置
        sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_scan)
        
        # 计算绝对误差
        abs_err_x = np.mean(np.abs(sim_x_interp - exp_x_norm))
        
        # 计算相对误差
        rel_err_x = 0.0
        valid_mask = exp_x_norm > 0.05 * exp_x_max
        if np.any(valid_mask):
            rel_err_x = np.mean(np.abs((sim_x_interp[valid_mask] - exp_x_norm[valid_mask]) / exp_x_norm[valid_mask])) * 100
        
        # 处理束沿X移动时的Y方向轮廓
        exp_y_data = self.beam_traced_x_axis  # 束沿X移动时测量的Y方向截面
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        
        # 归一化实验数据 - 防止除零
        exp_y_max = np.max(exp_y_val)
        exp_y_norm = exp_y_val / exp_y_max if exp_y_max > 1e-5 else np.zeros_like(exp_y_val)
        
        # 插值模拟数据到实验点位置
        sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_scan)
        
        # 计算绝对误差
        abs_err_y = np.mean(np.abs(sim_y_interp - exp_y_norm))
        
        # 计算相对误差
        rel_err_y = 0.0
        valid_mask = exp_y_norm > 0.05 * exp_y_max
        if np.any(valid_mask):
            rel_err_y = np.mean(np.abs((sim_y_interp[valid_mask] - exp_y_norm[valid_mask]) / exp_y_norm[valid_mask])) * 100
        
        # 组合绝对误差
        abs_error = (abs_err_x + abs_err_y) / 2
        
        return abs_error, rel_err_x, rel_err_y, (sim_x_interp - exp_x_norm, sim_y_interp - exp_y_norm)

    def gradient_descent_step(self, beam_matrix, residuals):
        """梯度下降步 - 根据误差有方向地调整束流分布"""
        residuals_x, residuals_y = residuals
        
        # 将残差降采样到网格大小
        exp_points_count = max(len(residuals_x), len(residuals_y))
        grid_points_count = len(self.grid)
        
        if exp_points_count != grid_points_count:
            # 使用插值法将残差匹配到网格点
            from scipy.interpolate import interp1d
            exp_positions = np.linspace(-15, 15, exp_points_count)
            
            # 处理 X 方向残差
            if len(residuals_x) != grid_points_count:
                f_x = interp1d(exp_positions, residuals_x, kind='linear', fill_value="extrapolate")
                residuals_x_downsampled = f_x(self.grid)
            else:
                residuals_x_downsampled = residuals_x
            
            # 处理 Y 方向残差
            if len(residuals_y) != grid_points_count:
                f_y = interp1d(exp_positions, residuals_y, kind='linear', fill_value="extrapolate")
                residuals_y_downsampled = f_y(self.grid)
            else:
                residuals_y_downsampled = residuals_y
        else:
            residuals_x_downsampled = residuals_x
            residuals_y_downsampled = residuals_y
        
        grad_x, grad_y = np.zeros_like(beam_matrix), np.zeros_like(beam_matrix)
        
        # 计算 X 方向扫描的梯度
        for j in range(len(self.grid)):
            y_pos = self.grid[j]
            path_points = np.column_stack((self.grid, np.full_like(self.grid, y_pos)))
            
            # 计算每个点对当前位置的贡献
            for i in range(len(self.grid)):
                x_pos = self.grid[i]
                
                # 计算点(x_pos, y_pos)对路径的贡献
                # 沿 X 方向的路径点
                contrib_mask = np.linalg.norm(path_points - [x_pos, y_pos], axis=1) < self.grid_spacing * 1.5
                
                # 权重：距离越近贡献越大
                distances = np.linalg.norm(path_points - [x_pos, y_pos], axis=1)
                weights = np.exp(-distances**2 / (2 * (self.grid_spacing/2)**2))
                
                # === 使用调整后的残差 ===
                grad_x[i, j] += np.sum(residuals_x_downsampled[contrib_mask] * weights[contrib_mask])
        
        # 计算 Y 方向扫描的梯度
        for i in range(len(self.grid)):
            x_pos = self.grid[i]
            path_points = np.column_stack((np.full_like(self.grid, x_pos), self.grid))
            
            for j in range(len(self.grid)):
                y_pos = self.grid[j]
                
                # 计算点(x_pos, y_pos)对路径的贡献
                contrib_mask = np.linalg.norm(path_points - [x_pos, y_pos], axis=1) < self.grid_spacing * 1.5
                
                # 权重：距离越近贡献越大
                distances = np.linalg.norm(path_points - [x_pos, y_pos], axis=1)
                weights = np.exp(-distances**2 / (2 * (self.grid_spacing/2)**2))
                
                # === 使用调整后的残差 ===
                grad_y[i, j] += np.sum(residuals_y_downsampled[contrib_mask] * weights[contrib_mask])
        
        # 结合两个方向的梯度
        combined_grad = grad_x + grad_y
        
        # 应用优化掩膜 - 只在优化区域内更新
        combined_grad[~self.optimization_mask] = 0
        
        # 梯度归一化
        if np.abs(combined_grad).max() > 1e-5:
            combined_grad = combined_grad / np.abs(combined_grad).max()
        
        # 高斯平滑梯度（避免高频噪声）
        combined_grad = gaussian_filter(combined_grad, sigma=0.8)
        
        return combined_grad


    def enforce_distribution_constraints(self, beam_matrix):
        """强制执行三项约束"""
        # 1. 非负约束
        beam_matrix = np.maximum(beam_matrix, 0)
        
        # 2. 优化区域外清零
        beam_matrix[~self.optimization_mask] = 0
        
        # 3. 寻找中心点
        max_i, max_j = np.unravel_index(np.argmax(beam_matrix), beam_matrix.shape)
        center_pos = (self.grid[max_i], self.grid[max_j])
        
        # 4. 中心10mm半径内禁止零值点
        inner_radius_mask = self.distance_from_center <= 10.0
        inner_values = beam_matrix[inner_radius_mask]
        
        # 设置内部最小值为最大值的1%
        min_val = np.max(beam_matrix) * 0.01
        inner_values[inner_values < min_val] = min_val
        beam_matrix[inner_radius_mask] = inner_values
        
        return beam_matrix

    def run_optimization(self, max_iterations=100, target_abserror=0.01, target_relerror=15.0):
        """基于梯度下降的优化过程"""
        self.log(f"\n开始优化过程 (目标绝对误差: {target_abserror}, 目标相对误差: {target_relerror}%)")
        self.log(f"初始学习率: {self.learning_rate:.3f}")
        
        current_matrix = self.optimized_beam.copy()
        best_error = None
        best_matrix = current_matrix.copy()
        start_time = time.time()
        
        # 初始评估
        sim_x = self.simulate_etching(current_matrix, "x")
        sim_y = self.simulate_etching(current_matrix, "y")
        abs_error, rel_err_x, rel_err_y, residuals = self.calculate_error(sim_x, sim_y)
        
        # 记录初始状态
        self.history["iteration"].append(0)
        self.history["abs_error"].append(abs_error)
        self.history["rel_err_x"].append(rel_err_x)
        self.history["rel_err_y"].append(rel_err_y)
        self.history["learning_rate"].append(self.learning_rate)
        self.history["improvement"].append(0.0)
        
        self.log(f"初始误差: 绝对误差={abs_error:.4f} (束X动时Y向误差: {rel_err_x:.1f}%, 束Y动时X向误差: {rel_err_y:.1f}%)")
        
        iteration = 0
        convergence_count = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.log(f"\n迭代 {iteration} (学习率: {self.learning_rate:.4f})")
            
            # 1. 计算梯度
            gradient = self.gradient_descent_step(current_matrix, residuals)
            
            # 2. 应用梯度更新
            direction_vector = np.copy(gradient)
            direction_vector[gradient > 0] = 1
            direction_vector[gradient < 0] = -1
            direction_vector[gradient == 0] = 0
            
            # 3. 计算更新幅度
            update_vector = self.learning_rate * direction_vector
            
            # 4. 应用更新
            new_matrix = current_matrix - update_vector  # 减去梯度（方向已考虑）
            new_matrix = self.enforce_distribution_constraints(new_matrix)
            
            # 5. 评估新矩阵
            sim_x_new = self.simulate_etching(new_matrix, "x")
            sim_y_new = self.simulate_etching(new_matrix, "y")
            new_abs_error, new_rel_err_x, new_rel_err_y, new_residuals = self.calculate_error(sim_x_new, sim_y_new)
            
            # 6. 检查是否改进
            improvement = abs_error - new_abs_error
            
            if improvement > 0:  # 误差减小
                # 接受更新
                improvement_percent = improvement / abs_error * 100
                abs_error = new_abs_error
                rel_err_x = new_rel_err_x
                rel_err_y = new_rel_err_y
                current_matrix = new_matrix
                residuals = new_residuals
                
                # 记录历史
                self.history["iteration"].append(iteration)
                self.history["abs_error"].append(abs_error)
                self.history["rel_err_x"].append(rel_err_x)
                self.history["rel_err_y"].append(rel_err_y)
                self.history["learning_rate"].append(self.learning_rate)
                self.history["improvement"].append(improvement)
                
                # 略微增加学习率（但不超过上限）
                self.learning_rate = min(self.learning_rate * 1.1, 1.0)
                
                self.log(f"   接受更新: 绝对误差改善 {improvement_percent:.2f}%，新误差={abs_error:.5f}")
                self.log(f"   束X动时Y向误差: {rel_err_x:.1f}%, 束Y动时X向误差: {rel_err_y:.1f}%")
                
                # 检查是否满足目标
                if abs_error < target_abserror and rel_err_x < target_relerror and rel_err_y < target_relerror:
                    self.log(f"在{iteration}代达成目标误差!")
                    break
                    
                # 重置收敛计数
                convergence_count = 0
            else:
                # 未改进 - 减小学习率
                convergence_count += 1
                self.learning_rate *= 0.7  # 减小学习率
                
                self.log(f"⚠️ 拒绝更新: 没有改进或恶化 ({improvement:.6f})")
                self.log(f"   新学习率: {self.learning_rate:.4f}")
                
                # 如果学习率过小或多次未改进，尝试随机扰动跳出局部最小值
                if self.learning_rate < 1e-4 or convergence_count > 5:
                    # 随机扰动跳出局部最小
                    perturbation = np.random.normal(0, 0.01, current_matrix.shape)
                    current_matrix = np.maximum(current_matrix + perturbation, 0)
                    current_matrix = self.enforce_distribution_constraints(current_matrix)
                    self.log("应用随机扰动跳出局部最小")
                    convergence_count = 0
                    
                    # 重新评估当前矩阵
                    sim_x_new = self.simulate_etching(current_matrix, "x")
                    sim_y_new = self.simulate_etching(current_matrix, "y")
                    abs_error, rel_err_x, rel_err_y, residuals = self.calculate_error(sim_x_new, sim_y_new)
                    self.log(f"扰动后误差: {abs_error:.5f}")
            
            # 每10次记录一次进度
            if iteration % 10 == 0 or improvement > 0:
                elapsed = time.time() - start_time
                self.log(f"进度: 迭代 {iteration}, 绝对误差={abs_error:.5f}, 相对误差={rel_err_x:.1f}%/{rel_err_y:.1f}%, 用时 {elapsed:.1f}秒")
            
            # 保存每50代的中间结果
            if iteration % 50 == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"beam_intermediate_iter{iteration}.csv"
                np.savetxt(filename, current_matrix * self.max_val, delimiter=",")
                self.log(f"保存中间结果: {filename}")
        
        # 最终处理
        self.optimized_beam = current_matrix
        
        elapsed_time = time.time() - start_time
        self.log(f"\n优化完成! 总迭代次数: {iteration}")
        self.log(f"最终绝对误差: {abs_error:.5f} (初始={self.history['abs_error'][0]:.5f})")
        self.log(f"最终相对误差:")
        self.log(f"  束X移动时Y向误差: {rel_err_x:.1f}% (初始={self.history['rel_err_x'][0]:.1f}%)")
        self.log(f"  束Y移动时X向误差: {rel_err_y:.1f}% (初始={self.history['rel_err_y'][0]:.1f}%)")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        
        # 保存最终结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_beam_distribution_{timestamp}.csv"
        np.savetxt(filename, current_matrix * self.max_val, delimiter=",")
        self.log(f"优化后束流分布保存至: {filename}")
        
        return current_matrix, rel_err_x, rel_err_y, abs_error

    def visualize_results(self, iteration=None):
        """可视化优化结果 - 增强型"""
        try:
            if iteration is None:
                fig = plt.figure(figsize=(18, 12))
                suptitle = "离子束刻蚀效率优化结果"
            else:
                fig = plt.figure(figsize=(18, 12))
                suptitle = f"离子束刻蚀效率优化结果 (迭代 {iteration})"
            
            fig.suptitle(suptitle, fontsize=16, y=0.98)
            
            # 原始束流分布
            ax1 = plt.subplot(241)
            im1 = ax1.imshow(self.initial_beam, cmap="viridis", extent=[-15, 15, -15, 15])
            ax1.set_title("初始束流分布")
            plt.colorbar(im1, ax=ax1, label="刻蚀速率 (nm/s)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 优化后束流分布
            ax2 = plt.subplot(242)
            optimized_beam_full = self.optimized_beam * self.max_val
            im2 = ax2.imshow(optimized_beam_full, cmap="viridis", extent=[-15, 15, -15, 15])
            ax2.set_title("优化后束流分布")
            plt.colorbar(im2, ax=ax2, label="刻蚀速率 (nm/s)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            
            # 束流分布变化
            ax3 = plt.subplot(243)
            diff = abs(optimized_beam_full - self.initial_beam)
            im3 = ax3.imshow(diff, cmap="hot", extent=[-15, 15, -15, 15])
            ax3.set_title("束流分布变化")
            plt.colorbar(im3, ax=ax3, label="变化量 (nm/s)")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            
            # 优化区域掩膜
            ax4 = plt.subplot(244)
            im4 = ax4.imshow(self.optimization_mask, cmap="binary", extent=[-15, 15, -15, 15])
            ax4.set_title("优化区域 (半径 ≤ 10mm)")
            ax4.set_xlabel("X (mm)")
            ax4.set_ylabel("Y (mm)")
            
            # 束沿Y轴移动时的X方向轮廓对比
            ax5 = plt.subplot(425)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")
            
            exp_x_data = self.beam_traced_y_axis
            exp_x_val = exp_x_data[:, 1]
            exp_x_max = np.max(exp_x_val)
            exp_x_norm = exp_x_val / exp_x_max if exp_x_max > 0 else exp_x_val
            
            ax5.scatter(
                exp_x_data[:, 0], 
                exp_x_norm, 
                c="g", s=20, label="实验数据 (束沿Y)"
            )
            ax5.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            ax5.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            ax5.set_title("束沿Y移动: X方向轮廓")
            ax5.set_xlabel("X位置 (mm)")
            ax5.set_ylabel("归一化刻蚀深度")
            ax5.grid(True, alpha=0.3)
            ax5.legend(loc='best', fontsize=8)
            
            # 束沿X轴移动时的Y方向轮廓对比
            ax6 = plt.subplot(426)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")
            
            exp_y_data = self.beam_traced_x_axis
            exp_y_val = exp_y_data[:, 1]
            exp_y_max = np.max(exp_y_val)
            exp_y_norm = exp_y_val / exp_y_max if exp_y_max > 0 else exp_y_val
            
            ax6.scatter(
                exp_y_data[:, 0], 
                exp_y_norm, 
                c="g", s=20, label="实验数据 (束沿X)"
            )
            ax6.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            ax6.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            ax6.set_title("束沿X移动: Y方向轮廓")
            ax6.set_xlabel("Y位置 (mm)")
            ax6.set_ylabel("归一化刻蚀深度")
            ax6.grid(True, alpha=0.3)
            ax6.legend(loc='best', fontsize=8)
            
            # 误差收敛曲线
            ax7 = plt.subplot(427)
            ax7.plot(self.history["iteration"], self.history["abs_error"], "ko-", markersize=5)
            ax7.set_xlabel("迭代次数", fontsize=9)
            ax7.set_ylabel("绝对误差", color='k', fontsize=9)
            ax7.tick_params(axis='y', labelcolor='k', labelsize=8)
            ax7.grid(True, alpha=0.2)
            
            ax7b = ax7.twinx()
            ax7b.plot(self.history["iteration"], self.history["rel_err_x"], "r.-", label="束X动时Y向误差", markersize=5)
            ax7b.plot(self.history["iteration"], self.history["rel_err_y"], "g.-", label="束Y动时X向误差", markersize=5)
            ax7b.set_ylabel("相对误差 (%)", color='b', fontsize=9)
            ax7b.tick_params(axis='y', labelcolor='b', labelsize=8)
            ax7b.legend(loc='best', fontsize=8)
            ax7.set_title("误差收敛曲线", fontsize=10)
            
            # 学习率变化
            ax8 = plt.subplot(428)
            ax8.plot(self.history["iteration"], self.history["learning_rate"], "b^-", markersize=5)
            ax8.set_xlabel("迭代次数", fontsize=9)
            ax8.set_ylabel("学习率", color='b', fontsize=9)
            ax8.tick_params(axis='y', labelcolor='b', labelsize=8)
            ax8.set_yscale('log')
            ax8.grid(True, alpha=0.2)
            ax8.set_title("学习率变化", fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if iteration is None:
                filename = "beam_optimization_final_results.png"
            else:
                filename = f"beam_optimization_iter_{iteration}.png"
                
            plt.savefig(filename, dpi=200)
            plt.close(fig)
            self.log(f"优化结果可视化已保存至: {filename}")
        except Exception as e:
            self.log(f"可视化失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def finalize(self):
        """结束优化并关闭日志"""
        # 最终可视化
        self.visualize_results()
        
        self.log("\n优化完成!")
        self.log("=" * 40)
        initial_abs_err = self.history['abs_error'][0]
        final_abs_err = self.history['abs_error'][-1]
        improvement_abs = (initial_abs_err - final_abs_err) / initial_abs_err * 100
        
        initial_rel_x = self.history['rel_err_x'][0]
        final_rel_x = self.history['rel_err_x'][-1]
        improvement_rel_x = (initial_rel_x - final_rel_x) / initial_rel_x * 100
        
        initial_rel_y = self.history['rel_err_y'][0]
        final_rel_y = self.history['rel_err_y'][-1]
        improvement_rel_y = (initial_rel_y - final_rel_y) / initial_rel_y * 100
        
        self.log(f"结果: 绝对误差改善 {improvement_abs:.1f}% ({initial_abs_err:.5f} -> {final_abs_err:.5f})")
        self.log(f"      束X动时Y向误差改善 {improvement_rel_x:.1f}% ({initial_rel_x:.1f}% -> {final_rel_x:.1f}%)")
        self.log(f"      束Y动时X向误差改善 {improvement_rel_y:.1f}% ({initial_rel_y:.1f}% -> {final_rel_y:.1f}%)")
        self.log("=" * 40)
        self.log_file.close()

# ================== 主程序 ==================
def main():
    print("\n\n" + "=" * 80)
    print("基于梯度下降的非高斯离子束束流分布优化".center(80))
    print("=" * 80)
    
    # 配置文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查文件存在性
    files = {
        "beam_traced_x_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",  # 束沿X移动时测量的Y截面
        "beam_traced_y_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",  # 束沿Y移动时测量的X截面
        "initial_beam": "beamprofile.csv"
    }
    
    print("输入文件:")
    for name, path in files.items():
        full_path = os.path.join(base_dir, path)
        exist = "✓" if os.path.exists(full_path) else "✗"
        print(f" - {name}: {exist:1} {full_path}")
    
    # 检查文件是否存在
    if any(not os.path.exists(os.path.join(base_dir, path)) for path in files.values()):
        print("\n错误: 部分文件不存在，请检查文件路径！")
        return
    
    print("=" * 80)
    
    try:
        # 创建优化器
        optimizer = BeamEfficiencyOptimizer(
            beam_traced_x_axis=os.path.join(base_dir, files["beam_traced_x_axis"]),
            beam_traced_y_axis=os.path.join(base_dir, files["beam_traced_y_axis"]),
            initial_guess_path=os.path.join(base_dir, files["initial_beam"])
        )
        
        # 运行主优化
        optimizer.log("\n===== 主优化过程 =====")
        result, err_x, err_y, abs_err = optimizer.run_optimization(
            max_iterations=200,
            target_abserror=0.01,
            target_relerror=15.0
        )
        
        # 最终处理
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("优化完成! 最终误差:".center(80))
        print(f" - 绝对误差: {abs_err:.5f}")
        print(f" - 束沿X移动时Y向误差: {err_x:.1f}%")
        print(f" - 束沿Y移动时X向误差: {err_y:.1f}%")
        print("=" * 80)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        if 'optimizer' in locals():
            optimizer.log_file.close()

if __name__ == "__main__":
    
    main()
