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
        self.opt_radius = 15.0  # 优化半径 (mm)，覆盖15mm半径
        self.max_mutations = 4  # 变异候选数

        # 创建优化掩膜
        self.create_optimization_mask()
        
        # 中心点位置
        self.center_i, self.center_j = self.find_center(self.initial_beam)
        self.log(f"初始中心点位置: ({self.grid[self.center_i]:.2f}, {self.grid[self.center_j]:.2f})")
        
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
        """创建11个阶段的点集掩膜"""
        self.stage_masks = []
        total_mask = np.zeros((self.grid_points, self.grid_points), dtype=bool)
        
        # 生成所有网格坐标
        xx, yy = np.meshgrid(range(self.grid_points), range(self.grid_points), indexing='ij')
        
        # 第一阶段 (k=0): x=0 或 y=0 的点
        stage0_mask = ((np.abs(self.grid[xx]) < 1e-5) | (np.abs(self.grid[yy]) < 1e-5))
        self.stage_masks.append(stage0_mask)
        total_mask = total_mask | stage0_mask
        self.log(f"阶段0: x=0或y=0的点, 包含点数: {np.sum(stage0_mask)}")
        
        # 第二阶段到第十阶段 (k=1到9): |x|=k 或 |y|=k 的点，排除之前阶段
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
            self.log(f"阶段{k}: |x|={k}或|y|={k}的点, 包含点数: {np.sum(stage_mask)}")
        
        # 第十一阶段 (k=10): 剩余所有点
        stage10_mask = ~total_mask
        self.stage_masks.append(stage10_mask)
        self.log(f"阶段10: 剩余点, 包含点数: {np.sum(stage10_mask)}")
        
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
        1. 确保中心点位于物理中心
        2. 强制径向单调递减
        """
        rows, cols = beam_matrix.shape
        
        # 重新确定中心点
        center_i, center_j = self.center_i, self.center_j
        center_pos = (self.grid[center_i], self.grid[center_j])
        max_val = beam_matrix[center_i, center_j]
        
        # 强制径向单调递减 - 使用16个方向
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        modified_points = 0
        
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            max_radius = self.opt_radius * 1.2
            steps = np.arange(0, max_radius + 0.5, 0.5)
            distances = steps
            
            # 计算射线上的网格点索引
            x_indices = np.round(center_i + dx * (distances / self.grid_spacing)).astype(int)
            y_indices = np.round(center_j + dy * (distances / self.grid_spacing)).astype(int)
            
            # 筛选有效索引
            valid_mask = (x_indices >= 0) & (x_indices < rows) & (y_indices >= 0) & (y_indices < cols)
            valid_x = x_indices[valid_mask]
            valid_y = y_indices[valid_mask]
            
            if len(valid_x) < 5:
                continue
                
            # 强制单调递减
            for idx in range(1, len(valid_x)):
                i, j = valid_x[idx], valid_y[idx]
                current_val = beam_matrix[i, j]
                prev_i, prev_j = valid_x[idx-1], valid_y[idx-1]
                prev_val = beam_matrix[prev_i, prev_j]
                
                # 确保当前值不超过前一个值
                if current_val > prev_val:
                    # 设为前一个值的98%
                    new_val = prev_val * 0.98
                    beam_matrix[i, j] = new_val
                    modified_points += 1
        
        # 确保所有值非负
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
        mutation_probability = 0.7
        
        # 中心值
        center_val = beam_matrix[self.center_i, self.center_j]
        
        for idx in range(len(indices[0])):
            i, j = indices[0][idx], indices[1][idx]
            
            # 以一定概率跳过变异
            if np.random.rand() > mutation_probability:
                continue
            
            # 计算误差方向（1表示需要增加，-1表示需要减少）
            error_direction = 1 if errors[i, j] < 0 else -1
            
            # 变异幅度（带随机性）
            effective_magnitude = magnitude * (0.8 + 0.4 * np.random.rand())
            
            # 应用变异
            mutation = error_direction * effective_magnitude
            new_val = new_beam[i, j] + mutation
            
            # 确保值在合理范围内
            if np.abs(self.grid[i]) < 5.0 or np.abs(self.grid[j]) < 5.0:  # 靠近中心区域
                new_val = max(0, min(1.0, new_val))
            else:  # 远离中心区域
                new_val = max(0, min(0.7, new_val))
                
            new_beam[i, j] = new_val
        
        # 应用径向约束
        return self.enforce_radial_constraints(new_beam)

    def calculate_point_errors(self, beam_matrix, sim_x, sim_y):
        """
        计算网格点上每一点的误差
        策略：比较仿真值和实验值的差异
        """
        # 网格点上的坐标
        positions = np.array([(self.grid[i], self.grid[j]) for i in range(self.grid_points) for j in range(self.grid_points)])
        positions = positions.reshape((self.grid_points, self.grid_points, 2))
        
        # 计算X方向的误差
        exp_x_data = self.beam_traced_y_axis
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        exp_x_max = np.max(exp_x_val)
        exp_x_norm = exp_x_val / exp_x_max
        sim_x_interp = np.interp(positions[:, :, 0], self.grid, sim_x)
        
        exp_x_interp = np.interp(positions[:, :, 0], exp_x_pos, exp_x_norm)
        
        # 计算Y方向的误差
        exp_y_data = self.beam_traced_x_axis
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        exp_y_max = np.max(exp_y_val)
        exp_y_norm = exp_y_val / exp_y_max
        sim_y_interp = np.interp(positions[:, :, 1], self.grid, sim_y)
        
        exp_y_interp = np.interp(positions[:, :, 1], exp_y_pos, exp_y_norm)
        
        # 计算综合误差
        errors = (sim_x_interp - exp_x_interp + sim_y_interp - exp_y_interp) / 2.0
        
        return errors

    def calculate_magnitude(self, current_iter, max_iters, stage_idx):
        """动态计算变异幅度"""
        # 基础公式
        min_mag = 0.005  # 最小变异幅度
        max_mag = 0.15   # 最大变异幅度
        
        # 阶段自适应调整
        stage_factor = 1.0 - (stage_idx / (self.num_stages - 1)) * 0.7
        
        # 迭代衰减
        decay_factor = max(0.1, 1.0 - current_iter / max_iters)
        
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
            stage_name = "位置点集: x=0或y=0"
        elif stage_idx < 10:
            stage_name = f"位置点集: |x|={stage_idx}或|y|={stage_idx}"
        else:
            stage_name = "位置点集: 剩余点"
        
        self.log(f"\n=== 开始优化阶段 {stage_idx+1}/{self.num_stages}: {stage_name} ===")
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
            magnitude = self.calculate_magnitude(iteration, max_iterations, stage_idx)
            
            self.log(f"迭代 {iteration}: 阶段{stage_idx+1}/{self.num_stages}, 变异幅度={magnitude:.5f}")
            
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
            if best_abs_error < 0.005:
                self.log(f"阶段收敛：绝对误差{best_abs_error:.4f} < 0.005")
                break
            elif stagnation_count > 3:
                self.log(f"连续3次无改进，结束当前阶段优化")
                break
        
        self.log(f"阶段完成: 最佳绝对误差={best_abs_error:.4f}")
        return best_matrix

    def run_optimization(self):
        """运行11阶段优化过程"""
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
        
        # 设置每个阶段的迭代次数
        stage_iterations = [7, 6, 5, 5, 4, 4, 4, 3, 3, 3, 5]
        
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
        
        elapsed_time = time.time() - start_time
        self.log(f"\n优化完成! 总迭代次数: {len(self.history['iteration'])}")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        self.log(f"最终绝对误差: {final_abs_error:.4f} (初始={abs_error0:.4f})")
        self.log(f"最终相对误差: 束X动时Y向={final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%)")
        self.log(f"              束Y动时X向={final_rel_err_y:.1f}% (初始={rel_err_y0:.1f}%)")
        
        return self.optimized_beam, final_rel_err_x, final_rel_err_y

    def visualize_results(self):
        """可视化优化结果 - 特别展示坐标位置点集"""
        try:
            fig = plt.figure(figsize=(15, 12))
            fig.suptitle("离子束刻蚀效率优化结果 (11阶段位置点集优化)", fontsize=16)
            
            # 1. 原始束流分布 - 标记位置点集
            ax1 = plt.subplot(231)
            im1 = ax1.imshow(self.initial_beam, cmap="viridis", extent=[-15, 15, -15, 15])
            
            # 标记位置点集
            for i, mask in enumerate(self.stage_masks):
                indices = np.where(mask)
                if len(indices[0]) > 0:
                    x = self.grid[indices[0]]
                    y = self.grid[indices[1]]
                    ax1.scatter(x, y, s=40, marker=f'${i}$', alpha=0.8)
            
            ax1.set_title("初始束流分布 (带位置点集标记)")
            plt.colorbar(im1, ax=ax1, label="刻蚀速率 (nm/s)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 2. 优化后束流分布
            optimized_beam_full = self.optimized_beam * self.max_val
            ax2 = plt.subplot(232)
            im2 = ax2.imshow(optimized_beam_full, cmap="viridis", extent=[-15, 15, -15, 15])
            ax2.set_title("优化后束流分布")
            plt.colorbar(im2, ax=ax2, label="刻蚀速率 (nm/s)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            
            # 3. 差异分布
            ax3 = plt.subplot(233)
            diff = optimized_beam_full - self.initial_beam
            vmax = np.max(np.abs(diff))
            im3 = ax3.imshow(diff, cmap="coolwarm", extent=[-15, 15, -15, 15], vmin=-vmax, vmax=vmax)
            ax3.set_title("束流分布变化")
            plt.colorbar(im3, ax=ax3, label="变化量 (nm/s)")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            
            # 4. 束沿Y移动时的X方向轮廓
            ax4 = plt.subplot(234)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")
            
            ax4.scatter(
                self.beam_traced_y_axis[:, 0], 
                self.beam_traced_y_axis[:, 1]/np.max(self.beam_traced_y_axis[:, 1]), 
                c="g", s=30, alpha=0.6, label="实验数据 (束沿Y)"
            )
            ax4.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            ax4.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            ax4.set_title("束沿Y移动时的X轴截面")
            ax4.set_xlabel("X位置 (mm)")
            ax4.set_ylabel("归一化刻蚀深度")
            ax4.grid(True)
            ax4.legend()
            
            # 5. 束沿X移动时的Y方向轮廓
            ax5 = plt.subplot(235)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")
            
            ax5.scatter(
                self.beam_traced_x_axis[:, 0], 
                self.beam_traced_x_axis[:, 1]/np.max(self.beam_traced_x_axis[:, 1]), 
                c="g", s=30, alpha=0.6, label="实验数据 (束沿X)"
            )
            ax5.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            ax5.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            ax5.set_title("束沿X移动时的Y轴截面")
            ax5.set_xlabel("Y位置 (mm)")
            ax5.set_ylabel("归一化刻蚀深度")
            ax5.grid(True)
            ax5.legend()
            
            # 6. 误差收敛曲线
            if len(self.history["iteration"]) > 1:
                ax6 = plt.subplot(236)
                iterations = self.history["iteration"]
                abs_errors = self.history["abs_error"]
                rel_err_x = self.history["rel_err_x"]
                rel_err_y = self.history["rel_err_y"]
                
                # 左侧Y轴 - 绝对误差
                ax6.plot(iterations, abs_errors, "k-", label="绝对误差")
                ax6.set_xlabel("迭代次数")
                ax6.set_ylabel("绝对误差", color='k')
                ax6.tick_params(axis='y', labelcolor='k')
                ax6.grid(True)
                
                # 右侧Y轴 - 相对误差
                ax6b = ax6.twinx()
                ax6b.plot(iterations, rel_err_x, "r--", label="束X动时Y向误差")
                ax6b.plot(iterations, rel_err_y, "g--", label="束Y动时X向误差")
                ax6b.set_ylabel("相对误差 (%)", color='b')
                ax6b.tick_params(axis='y', labelcolor='b')
                
                # 标记阶段转换
                for i, end_idx in enumerate(self.stage_ends):
                    ax6.axvline(end_idx, color=f'C{i}', linestyle='--', alpha=0.7, label=f"阶段{i}结束")
                
                # 创建组合图例
                lines, labels = ax6.get_legend_handles_labels()
                lines2, labels2 = ax6b.get_legend_handles_labels()
                ax6.legend(lines + lines2, labels + labels2, loc='upper right')
                
                ax6.set_title("误差收敛曲线")
            
            plt.tight_layout()
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
    
    print("=" * 76)
    print("非高斯离子束束流分布优化 (11阶段位置点集优化)".center(76))
    print(f"输入文件:")
    print(f" - 束沿X移动时的Y截面: {files['beam_traced_x_axis']}")
    print(f" - 束沿Y移动时的X截面: {files['beam_traced_y_axis']}")
    print(f" - 初始束流分布: {files['initial_beam']}")
    print("=" * 76)
    
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
        
        # 执行11阶段优化
        result, err_x, err_y = optimizer.run_optimization()
        
        # 可视化
        optimizer.visualize_results()
        
        # 最终报告
        optimizer.finalize()
        
        print("\n" + "=" * 76)
        print("优化完成! 最终误差:".center(76))
        print(f" - 束沿X移动时Y向误差: {err_x:.1f}%")
        print(f" - 束沿Y移动时X向误差: {err_y:.1f}%")
        print(f"结果文件: optimized_beam_distribution.csv, beam_optimization_results.png")
        print("=" * 76)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        if 'optimizer' in locals() and hasattr(optimizer, 'log_file'):
            optimizer.log_file.close()

if __name__ == "__main__":
    main()
