import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter

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
    def __init__(self, beam_traced_x_axis, initial_guess_path, grid_bound=15.0, 
                 highres_points=121, lowres_points=31):
        """
        更新: 支持双精度网格系统
        - highres_points: 优化精度下的网格点数 (0.25mm间隔)
        - lowres_points: 原始精度下的网格点数 (1mm间隔)
        """
        # 创建日志文件
        self.log_file = open("beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("离子束刻蚀效率优化引擎 (高精度版)")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"多精度网格系统: 优化精度=0.25mm ({highres_points}x{highres_points})")
        self.log(f"输入文件说明: 初始束流={initial_guess_path}, 实验数据={beam_traced_x_axis}")
        self.log("=" * 30)
        
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
        
        # 加载实验数据
        self.beam_traced_x_axis = self.load_experimental_data(beam_traced_x_axis)
        
        # 加载并校准初始猜测
        self.load_initial_beam(initial_guess_path, self.beam_traced_x_axis)
        
        # 优化参数
        self.scan_velocity = 30.0  # 扫描速度 (mm/s)
        self.opt_radius = 15.0     # 优化半径 (mm)
        self.max_mutations = 4     # 变异候选数
        self.drift_sigma = 1.8     # 漂移校正标准差 (mm)

        # 创建高精度优化掩膜
        self.create_optimization_mask()
        
        # 中心点位置
        self.center_i, self.center_j = self.find_center(self.initial_beam)
        self.log(f"初始中心点位置: ({self.grid[self.center_i]:.2f}, {self.grid[self.center_j]:.2f})")
        
        # 创建距离矩阵（以中心为原点）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_center = np.sqrt(
            (xx - self.grid[self.center_i])**2 + 
            (yy - self.grid[self.center_j])**2
        )
        
        # 创建11个阶段的高精度点集掩膜
        self.num_stages = 11
        self.stage_masks = []  # 存储阶段信息
        self.create_stage_masks()
        
        self.stage_ends = []  # 存储阶段结束点的迭代索引
        
        # 历史记录
        self.history = {
            "iteration": [],
            "abs_error": [],
            "rel_err_x": [],
            "max_val": [],
            "etch_vol_error": []  # 新增：刻蚀体积误差
        }
        self.optimized_beam = self.initial_beam / self.max_val  # 初始优化束流
        
        # 应用初始约束确保中心辐射状分布
        self.optimized_beam = self.enforce_radial_constraints(self.optimized_beam, strict=True)
        self.log("初始径向约束已完成")

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

    def load_initial_beam(self, file_path, exp_data):
        """加载、插值并校准初始束流分布"""
        try:
            # 1. 加载原始1mm精度数据
            lowres_beam = np.genfromtxt(file_path, delimiter=",")
            if lowres_beam.shape != (self.lowres_points, self.lowres_points):
                self.log(f"警告: 初始束流尺寸应为{self.lowres_points}x{self.lowres_points}，实际为{lowres_beam.shape}")
                # 尝试自动调整尺寸
                if lowres_beam.size == self.lowres_points * self.lowres_points:
                    lowres_beam = lowres_beam.reshape((self.lowres_points, self.lowres_points))
                else:
                    raise ValueError(f"初始束流尺寸不兼容 ({lowres_beam.shape} vs {self.lowres_points}x{self.lowres_points})")
            
            # 2. 插值到0.25mm高精度网格
            # 创建插值器 - 使用 RegularGridInterpolator
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
            
            # 组合坐标点
            points = np.stack((xx_highres, yy_highres), axis=-1)
            
            # 执行插值
            highres_beam = interpolator(points)
            
            # 3. 应用自适应校准
            # 检测实验数据峰值位置 (x轴截面)
            exp_peak_x = exp_data[np.argmax(exp_data[:, 1]), 0]
            center_idx = np.argmin(np.abs(self.highres_grid - exp_peak_x))
            self.log(f"实验峰值位置: x={exp_peak_x:.2f}mm, 当前中心: x={self.grid[center_idx]:.2f}mm")
            
            # 创建径向衰减模板 (指数衰减模型)
            xx, yy = np.meshgrid(self.highres_grid, self.highres_grid, indexing="ij")
            r = np.sqrt((xx - exp_peak_x)**2 + yy**2)
            decay_template = np.exp(-r/5.0)
            
            # 应用峰值对齐的校准
            peak_adjusted = highres_beam * decay_template
            
            # 计算修正因子 (基于实验峰值)
            exp_peak_val = np.max(exp_data[:, 1])
            
            # 在中心区域计算平均值 (±0.5mm)
            center_mask = (r <= 0.5)
            sim_peak_val = np.mean(peak_adjusted[center_mask]) if np.any(center_mask) else 1.0
            
            if sim_peak_val < 1e-6:
                self.log(f"警告: 校准后中心强度过低 ({sim_peak_val:.2f})，跳过校准")
                self.initial_beam = highres_beam
            else:
                calibration_factor = exp_peak_val / sim_peak_val
                self.log(f"校准因子: {calibration_factor:.2f} (实验峰值={exp_peak_val:.2f}, 模拟峰值={sim_peak_val:.2f})")
                
                # 应用刻蚀深度校准 (考虑束流强度与深度的平方关系)
                self.initial_beam = peak_adjusted * np.sqrt(calibration_factor)
            
            # 确保所有值为非负
            self.initial_beam = np.maximum(self.initial_beam, 0)
            
            # 更新最大值
            self.max_val = np.max(self.initial_beam)
            self.log(f"高精度初始束流: 尺寸={self.initial_beam.shape}, 最大速率={self.max_val:.2f} nm/s")
            
            # 记录插值前后峰值的差异
            orig_peak = np.max(lowres_beam)
            new_peak = np.max(self.initial_beam)
            self.log(f"插值校准: 原始峰值={orig_peak:.2f} → 新峰值={new_peak:.2f} ({new_peak/orig_peak:.1%})")
            
        except Exception as e:
            self.log(f"加载失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
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
                if data.shape[0] % 2 == 0:
                    data = data.reshape((-1, 2))
                else:
                    raise ValueError(f"实验数据格式错误: {data.shape}")
                    
            # 确保数据按X位置排序
            sorted_idx = np.argsort(data[:, 0])
            data = data[sorted_idx]
            
            self.log(f"加载数据点: {len(data)} 个, 位置范围: [{data[0,0]:.2f}, {data[-1,0]:.2f}] mm")
            return data
        except Exception as e:
            self.log(f"加载失败: {str(e)}")
            raise

    def create_optimization_mask(self):
        """创建高精度优化区域掩膜"""
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.distance_from_center = np.sqrt(xx**2 + yy**2)
        self.optimization_mask = (self.distance_from_center <= self.opt_radius)
        self.log(f"高精度优化区域包含 {np.sum(self.optimization_mask)} 个点")

    def create_stage_masks(self):
        """创建11个阶段的高精度点集掩膜"""
        self.stage_masks = []
        total_mask = np.zeros((self.highres_points, self.highres_points), dtype=bool)
        
        # 生成所有网格坐标
        xx, yy = np.meshgrid(range(self.highres_points), range(self.highres_points), indexing='ij')
        
        # 第一阶段 (集合1): x=0 或 y=0 的点
        stage0_mask = ((np.abs(self.grid[xx]) < 1e-5) | (np.abs(self.grid[yy]) < 1e-5))
        self.stage_masks.append(stage0_mask)
        total_mask = total_mask | stage0_mask
        self.log(f"集合1 (阶段0): 坐标轴点 ({np.sum(stage0_mask)})")
        
        # 第二阶段到第十阶段 (集合2~集合10): |x|=k 或 |y|=k 的点，排除之前阶段
        for k in range(1, 10):
            k_value = float(k)
            # 定义位置阈值 (考虑浮点误差)
            threshold = self.grid_spacing
            # 找到坐标绝对值约为k的位置
            stage_mask = (
                (np.abs(np.abs(self.grid[xx]) - k_value) < threshold) | 
                (np.abs(np.abs(self.grid[yy]) - k_value) < threshold)
            )
            # 排除之前阶段已选择的点
            stage_mask = stage_mask & ~total_mask
            self.stage_masks.append(stage_mask)
            total_mask = total_mask | stage_mask
            self.log(f"集合{k+1} (阶段{k}: |坐标|={k}mm点 ({np.sum(stage_mask)})")
        
        # 第十一阶段 (集合11): 剩余所有点
        stage10_mask = ~total_mask
        self.stage_masks.append(stage10_mask)
        self.log(f"集合11 (阶段10): 剩余点 ({np.sum(stage10_mask)})")
        
        # 当前阶段索引
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
    
    def enforce_radial_constraints(self, beam_matrix, strict=False):
        """
        强制执行径向约束 - 针对高精度网格优化
        strict=True: 使用更严格的约束
        """
        rows, cols = beam_matrix.shape
        
        # 更新中心点位置
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        self.center_i, self.center_j = center_i, center_j
        center_pos = (self.grid[center_i], self.grid[center_j])
        max_val = beam_matrix[center_i, center_j]
        
        # 1. 确保中心点位于物理中心
        center_distance = np.sqrt(center_pos[0]**2 + center_pos[1]**2)
        if center_distance > 1.5:
            # 在-1.0~1.0mm范围内寻找最大值
            center_region = (
                (np.abs(self.grid) < 1.0)[:, None] & 
                (np.abs(self.grid) < 1.0)[None, :]
            )
            
            if np.any(center_region):
                region_values = beam_matrix.copy()
                region_values[~center_region] = -99
                center_i, center_j = np.unravel_index(np.argmax(region_values), region_values.shape)
                max_val = beam_matrix[center_i, center_j]
                center_pos = (self.grid[center_i], self.grid[center_j])
                beam_matrix = beam_matrix / beam_matrix[center_i, center_j] * max_val
        
        # 2. 创建距离矩阵 (以中心点为原点)
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt((xx - center_pos[0])**2 + (yy - center_pos[1])**2)
        
        # 3. 中心区域 (半径≤5mm) 禁止零值点
        inner_region = r <= 5.0
        zero_points_mask = beam_matrix < max_val * 0.05
        inner_zero_points = np.logical_and(inner_region, zero_points_mask)
        if np.sum(inner_zero_points) > 0:
            base_value = np.percentile(beam_matrix[inner_region], 10)
            beam_matrix[inner_zero_points] = base_value
            self.log(f"中心区域最低强度设定: {base_value:.4f} (max={max_val:.4f})")
        
        # 4. 沿36个射线方向强制单调递减
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        decay_factor = 0.97  # 每毫米衰减因子
        
        for angle in angles:
            dx, dy = np.cos(angle)*0.25, np.sin(angle)*0.25  # 高精度步长
            
            # 计算射线上的点 (1mm间隔评估)
            distances = np.arange(0, self.opt_radius + 1, 1.0)
            ray_points_i, ray_points_j = [], []
            
            for dist in distances:
                x_pos = center_pos[0] + dx * dist
                y_pos = center_pos[1] + dy * dist
                # 找到最近的网格点
                x_idx = np.argmin(np.abs(self.grid - x_pos))
                y_idx = np.argmin(np.abs(self.grid - y_pos))
                # 只添加唯一点
                if (x_idx, y_idx) not in zip(ray_points_i, ray_points_j):
                    ray_points_i.append(x_idx)
                    ray_points_j.append(y_idx)
            
            # 应用单调递减约束
            prev_val = max_val
            for count, (i, j) in enumerate(zip(ray_points_i, ray_points_j)):
                dist = r[i, j]
                target_val = max_val * (decay_factor ** dist)
                
                if beam_matrix[i, j] > min(prev_val, target_val * 1.1):
                    new_val = min(prev_val * decay_factor, target_val)
                    beam_matrix[i, j] = max(0, min(new_val, prev_val))
                
                prev_val = beam_matrix[i, j]
        
        # 5. 添加高斯平滑减少高精度网格噪声
        if strict:
            beam_matrix = gaussian_filter(beam_matrix, sigma=0.6)
        
        # 确保所有值非负
        beam_matrix = np.maximum(beam_matrix, 0)
        
        return beam_matrix

    def simulate_etching(self, beam_matrix):
        """模拟束沿Y轴移动时的X轮廓 (含漂移校正)"""
        interpolator = self.create_interpolator(beam_matrix)
        profile = np.zeros_like(self.grid)
        
        # 漂移卷积核 (高斯平滑)
        kernel_radius = int(3 * self.drift_sigma / self.grid_spacing)  # 3-sigma范围
        x_kernel = np.arange(-kernel_radius, kernel_radius + 1) * self.grid_spacing
        kernel = np.exp(-x_kernel**2 / (2 * self.drift_sigma**2))
        kernel = kernel / np.sum(kernel)
        
        # 创建路径点 - 高精度计算
        for j in range(len(self.grid)):
            y_pos = self.grid[j]
            path_points = np.column_stack((self.grid, np.full_like(self.grid, y_pos)))
            
            # 获取基本刻蚀速率
            etch_rates = interpolator(path_points)
            
            # 添加漂移扩散效应 (高斯卷积)
            etched_rates = np.convolve(etch_rates, kernel, mode='same')
            
            # 积分得到深度
            profile[j] = trapezoid(etched_rates, dx=self.grid_spacing)
        
        # 归一化
        max_profile = np.max(profile)
        return profile / max_profile if max_profile > 0 else profile

    def calculate_error(self, sim_x_scan):
        """计算高精度模拟结果与实验数据的绝对误差"""
        exp_x_data = self.beam_traced_x_axis
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        
        # 归一化实验数据
        exp_x_max = np.max(exp_x_val) if np.max(exp_x_val) > 0 else 1.0
        exp_x_norm = exp_x_val / exp_x_max
        
        # 归一化模拟数据
        sim_x_max = np.max(sim_x_scan) if np.max(sim_x_scan) > 0 else 1.0
        sim_x_norm = sim_x_scan / sim_x_max
        
        # 高精度插值 (匹配实验数据0.25mm精度)
        sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_norm)
        
        # 计算X误差
        abs_dev_x = np.abs(sim_x_interp - exp_x_norm)
        
        # 计算相对误差
        non_zero_x = exp_x_norm > 1e-5
        rel_err_x = np.mean(abs_dev_x[non_zero_x] / exp_x_norm[non_zero_x]) * 100 if np.any(non_zero_x) else 100.0
        
        # 计算综合绝对误差
        abs_error = np.mean(abs_dev_x)
        
        return abs_error, rel_err_x

    def mutate_beam(self, beam_matrix, magnitude, sim_x):
        """基于误差分析的定向变异策略 - 针对高精度坐标位置点集优化"""
        new_beam = beam_matrix.copy()
        current_mask = self.stage_masks[self.current_stage_idx]
        
        # 获取当前区域的索引
        indices = np.where(current_mask)
        
        # 动态变异参数设置
        center_val = beam_matrix[self.center_i, self.center_j]
        recent_error = np.mean(self.history["abs_error"][-5:]) if len(self.history["abs_error"]) > 5 else 0.3
        max_mag = min(0.3, max(0.03, 0.15 * recent_error * 3))
        
        for idx in range(len(indices[0])):
            i, j = indices[0][idx], indices[1][idx]
            
            # 随机跳过部分点 (早期优化阶段减少变异密度)
            if (self.current_stage_idx < 5 and np.random.rand() > 0.7) or \
               (self.current_stage_idx >= 5 and np.random.rand() > 0.85):
                continue
            
            # 计算误差方向 (1表示需要增加，-1表示需要减少)
            # 简化误差计算: 当前位置与目标值偏差
            center_val = beam_matrix[self.center_i, self.center_j]
            target_fraction = np.exp(-self.r_center[i, j]/7.0)
            target_val = center_val * target_fraction
            error_direction = 1 if beam_matrix[i, j] < target_val else -1
            
            # 变异幅度 (随距离衰减)
            dist_to_center = self.r_center[i, j]
            dist_factor = np.exp(-dist_to_center / 8.0)
            effective_magnitude = max_mag * dist_factor * (0.7 + 0.6*np.random.rand())
            
            # 应用变异
            mutation = error_direction * effective_magnitude
            new_val = new_beam[i, j] + mutation
            
            # 值范围约束 (基于位置)
            if dist_to_center < 5.0:  # 中心区域 (0-5mm)
                new_val = max(center_val*0.2, min(1.0, new_val))
            elif dist_to_center < 10.0:  # 过渡区域 (5-10mm)
                new_val = max(0, min(0.6, new_val))
            else:  # 边缘区域 (10-15mm)
                new_val = max(0, min(0.3, new_val))
                
            new_beam[i, j] = new_val
        
        # 应用径向约束 (带高斯平滑)
        return self.enforce_radial_constraints(new_beam, strict=True)

    def validate_etch_depth(self, beam_matrix):
        """验证高精度积分总量一致性"""
        # 1. 计算理论刻蚀总量
        grid_area = self.grid_spacing**2
        total_vals = np.sum(beam_matrix) * grid_area * self.max_val
        
        # 2. 由实验X截面计算Y积分
        exp_x_val = self.beam_traced_x_axis[:, 1]
        exp_x_pos = self.beam_traced_x_axis[:, 0]
        x_integral = np.trapz(exp_x_val, exp_x_pos)
        
        # 实验中Y范围
        y_range = self.grid[-1] - self.grid[0]
        exp_total = x_integral * y_range
        
        # 计算相对误差
        rel_error = (total_vals - exp_total) / exp_total
        self.log(f"刻蚀体积验证: 模拟值={total_vals:.1f} | 实验值={exp_total:.1f} | 误差={rel_error*100:.1f}%")
        return rel_error

    def optimize_stage(self, current_matrix, stage_idx, max_iterations):
        """执行单个高精度阶段优化"""
        self.current_stage_idx = stage_idx
        
        # 阶段描述
        if stage_idx == 0:
            stage_name = "坐标轴点"
        elif stage_idx < 10:
            stage_name = f"|坐标|={stage_idx}mm点"
        else:
            stage_name = "边缘区域"
            
        self.log(f"\n=== 优化阶段 {stage_idx+1}/11: {stage_name} ({np.sum(self.stage_masks[stage_idx])}点) ===")
        
        # 获取当前最佳模拟轮廓
        sim_x = self.simulate_etching(current_matrix)
        abs_error, rel_err_x = self.calculate_error(sim_x)
        etch_vol_err = self.validate_etch_depth(current_matrix)
        
        # 添加到历史记录
        self.history["etch_vol_error"].append(etch_vol_err)
        self.log(f"初始误差: 绝对={abs_error:.5f}, X截面相对误差={rel_err_x:.1f}%")
        
        best_abs_error = abs_error
        best_matrix = current_matrix.copy()
        stagnation_count = 0
        
        # 冻结前一阶段的区域
        frozen_mask = np.zeros_like(self.optimization_mask, dtype=bool)
        for i in range(stage_idx):
            frozen_mask = frozen_mask | self.stage_masks[i]
        
        for iteration in range(1, max_iterations + 1):
            # 动态变异幅度
            magnitude = min(0.2, 0.05 + 0.15 / (iteration + 1))
            
            # 创建变异候选
            candidate = self.mutate_beam(best_matrix, magnitude, sim_x)
            
            # 冻结已优化区域
            candidate[frozen_mask] = best_matrix[frozen_mask]
            candidate = self.enforce_radial_constraints(candidate)
            
            # 评估候选
            cand_sim_x = self.simulate_etching(candidate)
            cand_abs_error, cand_rel_err_x = self.calculate_error(cand_sim_x)
            cand_etch_err = self.validate_etch_depth(candidate)
            
            # 检查改进情况
            improvement = best_abs_error - cand_abs_error
            vol_improved = abs(cand_etch_err) < abs(self.history["etch_vol_error"][-1])
            
            if cand_abs_error < best_abs_error and vol_improved:
                improvement_percent = 100 * improvement / best_abs_error
                self.log(f"[S{stage_idx+1} I{iteration}] 改进: 绝对误差↓{100*improvement:.2f}% ({improvement_percent:.1f}%)")
                
                best_abs_error = cand_abs_error
                best_matrix = candidate.copy()
                sim_x = cand_sim_x
                stagnation_count = 0
                
                # 更新历史记录
                iter_num = len(self.history["iteration"])
                self.history["iteration"].append(iter_num)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(cand_rel_err_x)
                self.history["etch_vol_error"].append(cand_etch_err)
                self.optimized_beam = best_matrix
            else:
                stagnation_count += 1
                stop_reason = "体积约束" if not vol_improved else "误差未改善"
                self.log(f"[S{stage_idx+1} I{iteration}] 无改进: {stop_reason}")
            
            # 收敛检查
            if stagnation_count >= 3: 
                self.log(f"连续{stagnation_count}次无改进，结束阶段优化")
                break
            elif best_abs_error < 0.005 and vol_improved:
                self.log(f"达到高精度收敛阈值")
                break
        
        self.log(f"阶段完成: 最佳绝对误差={best_abs_error:.5f}")
        return best_matrix

    def run_optimization(self):
        """运行高精度11阶段优化过程"""
        # 初始评估
        sim_x = self.simulate_etching(self.initial_beam / self.max_val)
        abs_error0, rel_err_x0 = self.calculate_error(sim_x)
        etch_vol_err0 = self.validate_etch_depth(self.initial_beam / self.max_val)
        
        # 历史记录初始化
        self.history["iteration"].append(0)
        self.history["abs_error"].append(abs_error0)
        self.history["rel_err_x"].append(rel_err_x0)
        self.history["etch_vol_error"].append(etch_vol_err0)
        self.log(f"初始绝对误差: {abs_error0:.4f}")
        
        start_time = time.time()
        self.optimized_beam = self.initial_beam / self.max_val
        
        # 阶段迭代设置 (高精度优化需减少每阶段迭代次数)
        stage_iterations = [6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 4]
        
        # 执行11阶段优化
        for stage_idx in range(self.num_stages):
            max_iter = stage_iterations[stage_idx]
            self.optimized_beam = self.optimize_stage(
                self.optimized_beam, 
                stage_idx, 
                max_iter
            )
            self.stage_ends.append(len(self.history["iteration"]) - 1)
        
        # 全局精细优化 (2轮)
        self.log("\n===== 全局精细优化 =====")
        for global_round in range(2):
            self.log(f"=== 全局轮次 {global_round+1}/2 (高精度) ===")
            self.optimized_beam = self.optimize_stage(
                self.optimized_beam,
                self.num_stages - 1, 
                6
            )
        
        # 降采样到原始1mm精度
        highres_beam = self.optimized_beam * self.max_val
        optimized_beam_lowres = self.downsample_to_lowres(highres_beam)
        
        # 结果保存
        np.savetxt("optimized_beam_highres.csv", highres_beam, delimiter=",")
        np.savetxt("optimized_beam_distribution.csv", optimized_beam_lowres, delimiter=",")
        
        # 最终评估
        sim_x = self.simulate_etching(self.optimized_beam)
        final_abs_error, final_rel_err_x = self.calculate_error(sim_x)
        final_etch_err = self.validate_etch_depth(self.optimized_beam)
        
        # 性能统计
        elapsed_time = time.time() - start_time
        iter_count = len(self.history["iteration"]) - 1
        per_iter_time = elapsed_time / iter_count if iter_count > 0 else 0
        
        self.log(f"\n优化完成! 总迭代次数: {iter_count}")
        self.log(f"总耗时: {elapsed_time:.1f}秒 ({per_iter_time:.2f}秒/迭代)")
        self.log(f"最终绝对误差: {final_abs_error:.5f} (初始={abs_error0:.5f})")
        self.log(f"最终X截面相对误差: {final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%)")
        self.log(f"最终刻蚀体积误差: {final_etch_err*100:.1f}% (初始={etch_vol_err0*100:.1f}%)")
        
        # 结果可视化
        self.plot_radial_distribution(highres_beam)
        self.visualize_results()
        
        return optimized_beam_lowres, final_abs_error

    def downsample_to_lowres(self, highres_beam):
        """将高精度优化结果降采样到1mm精度"""
        self.log("将121x121高精度结果降采样到31x31矩阵")
        
        # 创建插值器
        interpolator = RegularGridInterpolator(
            (self.highres_grid, self.highres_grid),
            highres_beam,
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
        
        # 生成目标网格点 (1mm精度)
        xx_low, yy_low = np.meshgrid(self.lowres_grid, self.lowres_grid, indexing="ij")
        points_lowres = np.column_stack((xx_low.ravel(), yy_low.ravel()))
        
        # 插值采样
        lowres_values = interpolator(points_lowres)
        lowres_beam = lowres_values.reshape(self.lowres_points, self.lowres_points)
        
        # 确保非负值
        lowres_beam[lowres_beam < 0] = 0
        
        # 保存采样前后的峰值差异
        high_peak = np.max(highres_beam)
        low_peak = np.max(lowres_beam)
        self.log(f"降采样结果: 高分辨率峰值={high_peak:.2f} → 低分辨率峰值={low_peak:.2f}")
        
        return lowres_beam

    def plot_radial_distribution(self, beam_full):
        """绘制径向分布验证图"""
        plt.figure(figsize=(12, 6))
        
        # 1. 高精度径向分布
        r_flat = self.r_center.flatten()
        values = beam_full.flatten()
        
        # 计算平均值分箱
        bins = np.linspace(0, 15, 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        radial_avg = []
        for i in range(len(bins)-1):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            radial_mean = np.mean(values[mask]) if np.any(mask) else 0
            radial_avg.append(radial_mean)
        
        plt.scatter(r_flat, values, alpha=0.3, s=5, label='数据点')
        plt.plot(bin_centers, radial_avg, 'r-', linewidth=3, label='径向平均值')
        decay_ref = radial_avg[0] * np.exp(-bin_centers/5)
        plt.plot(bin_centers, decay_ref, 'b--', linewidth=2.5, label='指数衰减参考')
        
        plt.title("高精度束流径向分布 (0.25mm)")
        plt.xlabel("到中心点的距离 (mm)")
        plt.ylabel("刻蚀速率 (nm/s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("beam_radial_distribution.png", dpi=200)
        plt.close()
        self.log("束流径向分布图已保存")

    def visualize_results(self):
        """简化可视化优化结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 原始束流分布 (1mm精度)
        plt.subplot(231)
        orig_lowres = np.genfromtxt("beamprofile.csv", delimiter=",")
        plt.imshow(orig_lowres, cmap="viridis", extent=[-15, 15, -15, 15])
        plt.title("原始束流分布 (1mm)")
        plt.colorbar(label="刻蚀速率 (nm/s)")
        
        # 2. 高精度优化结果
        plt.subplot(232)
        highres_beam = self.optimized_beam * self.max_val
        plt.imshow(highres_beam, cmap="viridis", extent=[-15, 15, -15, 15])
        plt.title("优化后束流分布 (0.25mm)")
        plt.colorbar(label="刻蚀速率 (nm/s)")
        
        # 3. 降采样结果
        plt.subplot(233)
        lowres_beam = np.genfromtxt("optimized_beam_distribution.csv", delimiter=",")
        plt.imshow(lowres_beam, cmap="viridis", extent=[-15, 15, -15, 15])
        plt.title("降采样输出结果 (1mm)")
        plt.colorbar(label="刻蚀速率 (nm/s)")
        
        # 4. 拟合结果对比
        plt.subplot(212)
        sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val)
        sim_x_optim = self.simulate_etching(self.optimized_beam)
        
        # 实验数据与模拟结果对比
        plt.scatter(
            self.beam_traced_x_axis[:, 0], 
            self.beam_traced_x_axis[:, 1]/np.max(self.beam_traced_x_axis[:, 1]), 
            c="g", s=30, alpha=0.7, label="实验数据"
        )
        plt.plot(self.grid, sim_x_initial, "b--", linewidth=1.5, label="初始模拟")
        plt.plot(self.grid, sim_x_optim, "r-", linewidth=2, label="优化后模拟")
        
        plt.title("束沿Y移动时的X轴截面拟合")
        plt.xlabel("X位置 (mm)")
        plt.ylabel("归一化刻蚀深度")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("beam_optimization_results.png", bbox_inches='tight')
        plt.close()
        self.log("优化结果可视化已保存")

    def finalize(self):
        """结束优化并关闭日志"""
        self.log("\n优化完成!")
        self.log("结果文件:")
        self.log(f"  - optimized_beam_highres.csv (高精度优化结果, 121x121)")
        self.log(f"  - optimized_beam_distribution.csv (标准分辨率输出, 31x31)")
        self.log(f"  - beam_optimization_results.png (可视化报告)")
        self.log(f"  - beam_radial_distribution.png (径向分布)")
        self.log_file.close()

# ================== 主程序 ==================
def main():
    # 检查文件存在性
    files = {
        "beam_traced_x_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 80)
    print("高精度离子束刻蚀效率优化".center(80))
    print(f"优化精度: 0.25mm (121x121)")
    print(f"输出精度: 1mm (31x31)")
    print(f"输入文件:")
    print(f" - 束流移动截面: {files['beam_traced_x_axis']}")
    print(f" - 初始束流分布: {files['initial_beam']}")
    print("=" * 80)
    
    # 文件检查
    missing = []
    for name, path in files.items():
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")
    
    if missing:
        print("错误: 以下文件不存在:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)
    
    # 运行优化器
    try:
        optimizer = BeamEfficiencyOptimizer(
            beam_traced_x_axis=files["beam_traced_x_axis"],
            initial_guess_path=files["initial_beam"],
            highres_points=121,  # 0.25mm精度
            lowres_points=31     # 1mm精度
        )
        
        # 执行优化
        optimizer.run_optimization()
        
        # 生成报告
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("优化成功! 结果文件:".center(80))
        print(f" - 高精度结果 (121x121): optimized_beam_highres.csv")
        print(f" - 标准输出 (31x31): optimized_beam_distribution.csv")
        print(f" - 可视化报告: beam_optimization_results.png")
        print(f" - 径向分布图: beam_radial_distribution.png")
        print(f" - 详细日志: beam_optimization_log.txt")
        print("=" * 80)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
