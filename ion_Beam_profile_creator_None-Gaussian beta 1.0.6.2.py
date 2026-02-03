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
        self.opt_radius = 14.0  # 优化半径 (mm)，扩展到14mm以覆盖所有阶段
        self.max_mutations = 4  # 变异候选数

        # 创建优化掩膜
        self.create_optimization_mask()
        
        # 中心点位置
        self.center_i, self.center_j = self.find_center(self.initial_beam)
        self.log(f"初始中心点位置: ({self.grid[self.center_i]:.2f}, {self.grid[self.center_j]:.2f})")
        
        # 创建14个环状区域掩膜
        self.region_stages = []  # 存储阶段信息
        self.create_stage_masks()
        
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
        """创建14个环状阶段的掩膜"""
        self.stage_masks = []
        
        # 创建距离矩阵（以中心为原点）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_center = np.sqrt(
            (xx - self.grid[self.center_i])**2 + 
            (yy - self.grid[self.center_j])**2
        )
        
        # 定义14个阶段的半径边界
        stage_boundaries = [
            (0, 2), (2, 3), (3, 4), (4, 5),  # 阶段0-3
            (5, 6), (6, 7), (7, 8), (8, 9),  # 阶段4-7
            (9, 10), (10, 11), (11, 12),      # 阶段8-10
            (12, 13), (13, 14), (14, 15)      # 阶段11-13
        ]
        
        for i, (r_min, r_max) in enumerate(stage_boundaries):
            mask = (self.r_center >= r_min) & (self.r_center < r_max)
            mask = mask & self.optimization_mask
            point_count = np.sum(mask)
            
            self.stage_masks.append(mask)
            self.log(f"阶段{i}: {r_min}-{r_max}mm, 包含点数: {point_count}")
            
        # 剩余所有点（超出15mm的，主要是为了完整性）
        mask = (self.r_center >= 15.0)
        point_count = np.sum(mask)
        self.stage_masks.append(mask)
        self.log(f"剩余点: ≥15mm, 包含点数: {point_count}")
        
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
    
    def enforce_radial_constraints(self, beam_matrix):
        """
        强制执行径向约束
        1. 确保中心点位于物理中心附近
        2. 禁止中心区域零值点
        3. 强制径向单调递减
        """
        rows, cols = beam_matrix.shape
        
        # 重新确定中心点
        center_i, center_j = self.center_i, self.center_j
        center_pos = (self.grid[center_i], self.grid[center_j])
        max_val = beam_matrix[center_i, center_j]
        
        # 2. 创建距离矩阵
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt((xx - center_pos[0])**2 + (yy - center_pos[1])**2)
        
        # 3. 中心区域（半径≤5mm）禁止零值点
        inner_region = (r <= 5.0)
        low_points_mask = beam_matrix < max_val * 0.01
        inner_low_points = np.logical_and(inner_region, low_points_mask)
        low_points_count = np.sum(inner_low_points)
        
        if low_points_count > 0:
            base_value = max_val * 0.02
            beam_matrix[inner_low_points] = base_value
            self.log(f"禁止零值点: 设置 {low_points_count} 个点至少为 {base_value:.4f}")
        
        # 4. 沿射线方向的约束 - 确保单调递减
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 16个方向
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
            
            if len(valid_x) < 5:  # 至少需要5个点
                continue
                
            # 获取射线上的值
            ray_values = beam_matrix[valid_x, valid_y]
            
            # 强制单调递减
            for k in range(1, len(valid_x)):
                current_val = beam_matrix[valid_x[k], valid_y[k]]
                prev_val = beam_matrix[valid_x[k-1], valid_y[k-1]]
                
                # 确保严格递减（允许1%波动）
                if current_val > prev_val * 0.99:
                    new_val = prev_val * 0.98
                    beam_matrix[valid_x[k], valid_y[k]] = new_val
                    modified_points += 1
                
                # 设置递减上限
                upper_limit = prev_val * 0.95
                if current_val > upper_limit:
                    beam_matrix[valid_x[k], valid_y[k]] = upper_limit
                    modified_points += 1
            
        # 5. 确保所有值非负
        beam_matrix = np.maximum(beam_matrix, 0)
        
        if modified_points > 0:
            self.log(f"径向约束: 调整了 {modified_points} 个点")
        
        return beam_matrix

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
        基于误差分析的定向变异策略
        策略：如果当前位置的仿真值小于实验值，增加束流值；反之减小
        """
        new_beam = beam_matrix.copy()
        current_mask = self.stage_masks[self.current_stage_idx]
        
        # 计算当前位置的仿真值与实验值的误差
        errors = self.calculate_point_errors(beam_matrix, sim_x, sim_y)
        
        # 获取当前区域的索引
        indices = np.where(current_mask)
        
        # 每个点以概率0.7进行变异
        mutation_probability = 0.7
        mutation_mask = np.random.rand(*current_mask.shape) < mutation_probability
        mutation_mask = mutation_mask & current_mask
        
        mutation_indices = np.where(mutation_mask)
        
        center_val = beam_matrix[self.center_i, self.center_j]
        
        for idx in range(len(mutation_indices[0])):
            i, j = mutation_indices[0][idx], mutation_indices[1][idx]
            
            # 计算到中心的距离
            dist = self.r_center[i, j]
            
            # 误差方向（1表示需要增加，-1表示需要减少）
            error_direction = 1 if errors[i, j] < 0 else -1
            
            # 基于距离调整变异幅度（距离中心越远，变异幅度越小）
            dist_factor = np.exp(-dist / 8.0)
            
            # 基于当前值与中心值比例的因子
            if center_val > 1e-5:
                value_factor = max(0.1, min(1.0, new_beam[i, j] / center_val))
            else:
                value_factor = 0.5
                
            # 最终变异幅度
            effective_magnitude = magnitude * dist_factor * value_factor * np.random.rand()
            
            # 应用变异
            mutation = error_direction * effective_magnitude
            new_val = new_beam[i, j] + mutation
            
            # 不同区域应用不同上限
            r = self.r_center[i, j]
            if r < 2.0:  # 中心区域
                new_val = max(0, min(1.0, new_val))
            elif r < 5.0:  # 中间区域
                new_val = max(0, min(0.7, new_val))
            elif r < 10.0:  # 边缘区域
                new_val = max(0, min(0.5, new_val))
            else:  # 外部区域
                new_val = max(0, min(0.2, new_val))
                
            new_beam[i, j] = new_val
        
        # 应用径向约束
        return self.enforce_radial_constraints(new_beam)

    def calculate_point_errors(self, beam_matrix, sim_x, sim_y):
        """
        计算网格点上每一点的误差
        策略：比较仿真值和实验值的差异
        """
        # 网格点上的坐标
        positions = np.array([(x, y) for x in self.grid for y in self.grid])
        
        # 计算X方向的误差
        exp_x_data = self.beam_traced_y_axis
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        exp_x_max = np.max(exp_x_val)
        exp_x_norm = exp_x_val / exp_x_max
        sim_x_interp = np.interp(positions[:, 0], self.grid, sim_x)
        dev_x = sim_x_interp - np.interp(positions[:, 0], exp_x_pos, exp_x_norm)
        
        # 计算Y方向的误差
        exp_y_data = self.beam_traced_x_axis
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        exp_y_max = np.max(exp_y_val)
        exp_y_norm = exp_y_val / exp_y_max
        sim_y_interp = np.interp(positions[:, 1], self.grid, sim_y)
        dev_y = sim_y_interp - np.interp(positions[:, 1], exp_y_pos, exp_y_norm)
        
        # 计算综合误差
        errors = (dev_x + dev_y) / 2.0
        errors = errors.reshape((len(self.grid), len(self.grid)))
        
        return errors

    def calculate_magnitude(self, current_iter, max_iters):
        """动态计算变异幅度"""
        # 基础公式：初期高变异，后期低变异
        min_mag = 0.01  # 最小变异幅度
        base_magnitude = 0.15 * np.exp(-current_iter / (max_iters / 3))
        base_magnitude = max(base_magnitude, min_mag)
        
        # 根据阶段调整
        # 中心区域使用较大变异，外部区域使用较小变异
        stage_ratio = min(1.0, self.current_stage_idx / 14.0)
        magnitude = base_magnitude * (1.0 - stage_ratio * 0.7)
        
        # 添加随机波动 (±30%)
        magnitude *= 0.7 + np.random.rand() * 0.6
        
        return magnitude

    def optimize_stage(self, current_matrix, stage_idx, max_iterations):
        """执行单个阶段的优化"""
        self.current_stage_idx = stage_idx
        
        # 计算当前阶段半径范围
        r_min = stage_idx * 1.0
        r_max = min(15.0, r_min + 1.0)
        if stage_idx == 0:
            r_min = 0
        
        stage_name = f"环状区域: {r_min:.1f}-{r_max:.1f} mm"
        self.log(f"\n=== 开始优化阶段 {stage_idx+1}/14: {stage_name} ===")
        self.log(f"当前阶段包含点数: {np.sum(self.stage_masks[stage_idx])}")
        
        # 获取当前最佳模拟轮廓
        sim_x = self.simulate_etching(current_matrix, "x")
        sim_y = self.simulate_etching(current_matrix, "y")
        abs_error, rel_err_x, rel_err_y = self.calculate_error(sim_x, sim_y)
        
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
            magnitude = self.calculate_magnitude(iteration, max_iterations)
            
            self.log(f"迭代 {iteration}: 阶段{stage_idx+1}/{14}, 变异幅度={magnitude:.5f}")
            
            # 创建变异候选
            candidate = self.mutate_beam(best_matrix, magnitude, sim_x, sim_y)
            
            # 冻结之前优化过的区域
            candidate[frozen_mask] = best_matrix[frozen_mask]
            
            # 评估候选
            cand_sim_x = self.simulate_etching(candidate, "x")
            cand_sim_y = self.simulate_etching(candidate, "y")
            cand_abs_error, cand_rel_err_x, cand_rel_err_y = self.calculate_error(cand_sim_x, cand_sim_y)
            
            # 检查是否改进
            if cand_abs_error < best_abs_error:
                improvement = best_abs_error - cand_abs_error
                self.log(f"改进: Δ={improvement:.5f}, 新误差={cand_abs_error:.4f}")
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
                
                # 保存阶段结果
                self.optimized_beam = best_matrix
            else:
                stagnation_count += 1
                self.log(f"无改进，维持绝对误差: {best_abs_error:.4f}")
            
            # 检查收敛
            if best_abs_error < 0.001 * (15 - stage_idx):  # 根据阶段调整收敛标准
                self.log(f"阶段收敛：绝对误差{best_abs_error:.4f} < {0.001*(15-stage_idx):.4f}")
                break
            elif stagnation_count > 3:
                self.log(f"连续3次无改进，结束当前阶段优化")
                break
        
        self.log(f"阶段完成: 最佳绝对误差={best_abs_error:.4f}")
        return best_matrix

    def run_optimization(self):
        """运行14阶段优化过程"""
        # 初始评估
        current_matrix = self.optimized_beam.copy()
        sim_x = self.simulate_etching(current_matrix, "x")
        sim_y = self.simulate_etching(current_matrix, "y")
        abs_error0, rel_err_x0, rel_err_y0 = self.calculate_error(sim_x, sim_y)
        
        # 添加初始状态到历史记录
        self.history["iteration"].append(0)
        self.history["abs_error"].append(abs_error0)
        self.history["rel_err_x"].append(rel_err_x0)
        self.history["rel_err_y"].append(rel_err_y0)
        
        start_time = time.time()
        total_iterations = 0
        
        # 执行14个阶段优化
        stage_iterations = [5, 4, 4, 4,  # 0-3阶段
                           4, 4, 4, 4,   # 4-7阶段
                           3, 3, 3,      # 8-10阶段
                           3, 3, 2]      # 11-13阶段
        
        for stage_idx in range(14):
            max_iter = stage_iterations[stage_idx] if stage_idx < len(stage_iterations) else 3
            current_matrix = self.optimize_stage(current_matrix, stage_idx, max_iter)
            self.stage_ends.append(len(self.history["iteration"]) - 1)
            total_iterations += max_iter
        
        # 最终评估
        self.optimized_beam = current_matrix
        optimized_beam_full = self.optimized_beam * self.max_val
        
        # 保存结果
        np.savetxt("optimized_beam_distribution.csv", optimized_beam_full, delimiter=",")
        
        # 最终误差评估
        sim_x = self.simulate_etching(self.optimized_beam, "x")
        sim_y = self.simulate_etching(self.optimized_beam, "y")
        final_abs_error, final_rel_err_x, final_rel_err_y = self.calculate_error(sim_x, sim_y)
        
        elapsed_time = time.time() - start_time
        self.log(f"\n优化完成! 总迭代次数: {len(self.history['iteration'])}")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        self.log(f"最终绝对误差: {final_abs_error:.4f} (初始={abs_error0:.4f})")
        self.log(f"最终相对误差: 束X动时Y向={final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%)")
        self.log(f"              束Y动时X向={final_rel_err_y:.1f}% (初始={rel_err_y0:.1f}%)")
        
        return self.optimized_beam, final_rel_err_x, final_rel_err_y

    def visualize_results(self):
        """可视化优化结果"""
        try:
            fig = plt.figure(figsize=(18, 14))
            fig.suptitle("离子束刻蚀效率优化结果 (14阶段优化)", fontsize=20)
            
            # 原始束流分布
            ax1 = plt.subplot(331)
            im1 = ax1.imshow(self.initial_beam, cmap="viridis", extent=[-15, 15, -15, 15])
            ax1.set_title("初始束流分布")
            plt.colorbar(im1, ax=ax1, label="刻蚀速率 (nm/s)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 优化后束流分布
            optimized_beam_full = self.optimized_beam * self.max_val
            ax2 = plt.subplot(332)
            im2 = ax2.imshow(optimized_beam_full, cmap="viridis", extent=[-15, 15, -15, 15])
            ax2.set_title("优化后束流分布")
            plt.colorbar(im2, ax=ax2, label="刻蚀速率 (nm/s)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            
            # 差异分布
            ax3 = plt.subplot(333)
            diff = optimized_beam_full - self.initial_beam
            vmax = np.max(np.abs(diff))
            im3 = ax3.imshow(diff, cmap="coolwarm", extent=[-15, 15, -15, 15], vmin=-vmax, vmax=vmax)
            ax3.set_title("束流分布变化")
            plt.colorbar(im3, ax=ax3, label="变化量 (nm/s)")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            
            # 径向分布展示
            ax4 = plt.subplot(334)
            radial_bins = np.linspace(0, 15, 60)
            radial_profile = []
            for r in radial_bins:
                mask = (self.r_center >= r) & (self.r_center < r + 0.25)
                if np.any(mask):
                    radial_profile.append(np.mean(optimized_beam_full[mask]))
            
            ax4.plot(radial_bins[:len(radial_profile)], radial_profile, 'r-', linewidth=2)
            ax4.set_title("径向分布")
            ax4.set_xlabel("半径 (mm)")
            ax4.set_ylabel("刻蚀速率 (nm/s)")
            ax4.grid(True)
            ax4.set_xlim([0, 15])
            
            # 束沿Y移动时的X方向轮廓
            ax5 = plt.subplot(335)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")
            
            ax5.scatter(
                self.beam_traced_y_axis[:, 0], 
                self.beam_traced_y_axis[:, 1]/np.max(self.beam_traced_y_axis[:, 1]), 
                c="g", s=30, alpha=0.6, label="实验数据 (束沿Y)"
            )
            ax5.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            ax5.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            ax5.set_title("束沿Y移动时的X轴截面")
            ax5.set_xlabel("X位置 (mm)")
            ax5.set_ylabel("归一化刻蚀深度")
            ax5.grid(True)
            ax5.legend()
            
            # 束沿X移动时的Y方向轮廓
            ax6 = plt.subplot(336)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")
            
            ax6.scatter(
                self.beam_traced_x_axis[:, 0], 
                self.beam_traced_x_axis[:, 1]/np.max(self.beam_traced_x_axis[:, 1]), 
                c="g", s=30, alpha=0.6, label="实验数据 (束沿X)"
            )
            ax6.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            ax6.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            ax6.set_title("束沿X移动时的Y轴截面")
            ax6.set_xlabel("Y位置 (mm)")
            ax6.set_ylabel("归一化刻蚀深度")
            ax6.grid(True)
            ax6.legend()
            
            # 14个阶段对比
            ax7 = plt.subplot(313)
            stage_avg = []
            stage_names = []
            for i, mask in enumerate(self.stage_masks[:14]):
                r_min = i
                r_max = i+1
                if r_max > 14:
                    r_max = 15
                stage_name = f"{r_min}-{r_max}mm"
                if np.any(mask):
                    stage_avg.append(np.mean(optimized_beam_full[mask] / self.max_val))
                    stage_names.append(stage_name)
            
            ax7.bar(range(len(stage_avg)), stage_avg, color='skyblue')
            ax7.set_title("14个优化阶段的平均束流强度")
            ax7.set_xticks(range(len(stage_avg)))
            ax7.set_xticklabels(stage_names, rotation=45)
            ax7.set_ylabel("归一化束流强度")
            ax7.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])
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
    print("非高斯离子束束流分布优化 (14阶段环状优化)".center(70))
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
        
        # 执行14阶段优化
        result, err_x, err_y = optimizer.run_optimization()
        
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
