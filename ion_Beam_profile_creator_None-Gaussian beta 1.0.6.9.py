import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
import cv2  # 用于等高线检测
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
        self.log("离子束刻蚀效率优化引擎启动 (凸性约束版)")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        self.log(f"输入文件说明:")
        self.log(f" - 离子束沿X轴移动时的Y方向截面数据: {beam_traced_x_axis}")
        self.log(f" - 离子束沿Y轴移动时的X方向截面数据: {beam_traced_y_axis}")
        self.log("=" * 30)

        # 保存文件路径（关键修复！）
        self.beam_traced_x_axis_file_path = beam_traced_x_axis  # 保存X轴移动时测量的Y截面文件路径
        self.beam_traced_y_axis_file_path = beam_traced_y_axis  # 保存Y轴移动时测量的X截面文件路径
        
        # 添加高分辨率标志
        self.high_res_mode = False
        self.original_resolution = grid_points
        self.original_grid_bound = grid_bound   

        # 网格系统
        self.grid_bound = grid_bound
        self.grid_points = grid_points = 31
        self.grid = np.linspace(-grid_bound, grid_bound, grid_points)
        self.grid_spacing = 2 * grid_bound / (grid_points - 1)
        
        ########################################################
        # 首先创建最外圈掩膜（在加载束流数据前）
        ########################################################
        # 创建最外圈零值区掩膜 (|x|>=14 或 |y|>=14)
        xx, yy = np.meshgrid(self.grid, self.grid, indexing='ij')
        self.outer_ring_mask = (np.abs(xx) >= 14) | (np.abs(yy) >= 14)
        outer_ring_count = np.sum(self.outer_ring_mask)
        self.log(f"最外圈零值区: |x|>=14 或 |y|>=14, 包含 {outer_ring_count} 个点")

        ########################################################
        # 加载实验数据（必须先于初始束流加载）
        ########################################################

        self.log(f"加载实验数据")
        self.beam_traced_x_axis = self.load_experimental_data(beam_traced_x_axis)  # 沿X轴移动 (测量Y截面)
        self.beam_traced_y_axis = self.load_experimental_data(beam_traced_y_axis)  # 沿Y轴移动 (测量X截面)
        
        
        ########################################################
        # 然后加载初始猜测
        ########################################################
        self.initial_guess_path = initial_guess_path
        self.load_initial_beam(initial_guess_path)
        
        
        # 优化参数
        self.scan_velocity = 30.0  # 扫描速度 (mm/s)
        self.opt_radius = 15.0  # 优化半径 (mm)，覆盖15mm半径
        self.max_mutations = 4  # 变异候选数

        # 创建优化掩膜
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
            "max_val": [],
            "peak_violations": [],
            "convex_violations": []  # 新增：凸性违规计数
        }
        self.optimized_beam = self.initial_beam / self.max_val  # 初始优化束流

        # ==== 添加最外圈零值约束 ====
        xx, yy = np.meshgrid(self.grid, self.grid, indexing='ij')
        self.outer_ring_mask = (np.abs(xx) >= 14) | (np.abs(yy) >= 14)
        outer_ring_count = np.sum(self.outer_ring_mask)
        self.log(f"最外圈零值区: |x|>=14 或 |y|>=14, 包含 {outer_ring_count} 个点")

        # 应用外圈零值约束
        self.optimized_beam[self.outer_ring_mask] = 0
        self.log("强制应用最外圈零值约束: 设置最外圈刻蚀效率为0")
                
        # 应用初始约束确保中心辐射状分布
        self.optimized_beam, violations = self.enforce_radial_constraints(self.optimized_beam, strict=True)
        self.log(f"初始径向约束完成，修正峰值违规: {violations}处")
        
        # 应用初始行/列单峰约束
        self.optimized_beam, row_violations, col_violations = self.enforce_unimodal_row_col_constraints(self.optimized_beam)
        self.log(f"初始单峰约束完成，行违规: {row_violations}处, 列违规: {col_violations}处")
        
        # 检查初始凸性违规
        init_convex_violations = self.check_convex_violations(self.optimized_beam)
        self.log(f"初始等高线凸性违规: {init_convex_violations}处")

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
        """加载初始束流分布并进行强度调整"""
        self.log(f"加载初始束流猜测: {file_path}")
        try:
            self.initial_beam = np.genfromtxt(file_path, delimiter=",")
            if self.initial_beam.shape != (31, 31):
                self.log(f"错误: 初始束流尺寸应为31x31，实际为{self.initial_beam.shape}")
                raise ValueError("初始束流尺寸不匹配")
            
            # 确保最外圈为0
            self.initial_beam[self.outer_ring_mask] = 0
            self.log("初始束流分布已强制设置最外圈(|x|>=14 或 |y|>=14)为零值")
            
            ##################################################################
            # === 新增部分: 根据实验数据调整初始束流强度 ===
            ##################################################################
            # 1. 使用当前初始束流模拟沿Y轴移动（束沿Y轴移动 -> 测量X方向轮廓）
            interpolator = RegularGridInterpolator(
                (self.grid, self.grid),
                self.initial_beam,  # 使用原始未归一化的束流
                method="linear",
                bounds_error=False,
                fill_value=0.0
            )
            
            # 模拟沿Y轴移动时的X方向轮廓
            sim_x = np.zeros_like(self.grid)
            for j in range(len(self.grid)):
                y_pos = self.grid[j]
                path_points = np.column_stack((self.grid, np.full_like(self.grid, y_pos)))
                etch_rates = interpolator(path_points)
                sim_x[j] = trapezoid(etch_rates, dx=self.grid_spacing)
            
            # 2. 找到模拟轮廓的最大值
            max_initial = np.max(sim_x)
            self.log(f"初始束流模拟X方向最大刻蚀速率: {max_initial:.4f} nm/s")
            
            # 3. 获取实验数据的最大刻蚀速率（束沿Y轴移动时的X截面数据）
            exp_y_axis = self.beam_traced_y_axis
            exp_x_val = exp_y_axis[:, 1]
            max_crosssection = np.max(exp_x_val)
            self.log(f"实验数据X方向最大刻蚀速率: {max_crosssection:.4f} nm/s")
            
            # 4. 计算比例因子 (卷积移动速度30mm/s)
            m = max_crosssection / (max_initial/30)
            self.log(f"初始束流强度比例因子: {m:.4f}")
            
            # 5. 调整初始束流强度
            self.initial_beam *= m
            self.log(f"已将初始束流强度缩放 {m:.4f} 倍")
            ##################################################################
            
            # 然后进行归一化
            self.max_val = np.max(self.initial_beam)
            if self.max_val == 0:
                raise ValueError("最大刻蚀速率为零，无效输入")
            
            # 归一化束流(保持比例)
            self.optimized_beam = self.initial_beam / self.max_val
            self.log(f"最大刻蚀速率: {self.max_val:.2f} nm/s")
            
            # 检查初始束流的单峰性
            row_violations, col_violations = self.check_unimodal_violations(self.optimized_beam)
            if row_violations + col_violations > 0:
                self.log(f"警告: 初始束流存在行/列峰违规: {row_violations}行, {col_violations}列")
            
        except Exception as e:
            self.log(f"加载失败: {str(e)}")
            raise


    def upsample_beam(self, low_res_matrix, new_resolution=121):
        """将束流矩阵上采样到高分辨率"""
        # 原始网格
        low_res_grid = np.linspace(-self.grid_bound, self.grid_bound, len(self.grid))
        
        # 新高分辨率网格
        self.original_grid_bound = self.grid_bound
        self.original_resolution = len(self.grid)
        new_grid = np.linspace(-self.grid_bound, self.grid_bound, new_resolution)
        self.grid = new_grid
        self.grid_points = new_resolution
        
        # 创建插值器
        interp = RegularGridInterpolator(
            (low_res_grid, low_res_grid),
            low_res_matrix,
            method='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 生成高分辨率网格点
        xx, yy = np.meshgrid(new_grid, new_grid, indexing='ij')
        points = np.column_stack((xx.flatten(), yy.flatten()))
        
        # 执行插值
        hi_res_data = interp(points).reshape(new_resolution, new_resolution)
        
        # 重新创建所有依赖网格的属性
        self.create_optimization_mask()
        self.center_i, self.center_j = self.find_center(hi_res_data)
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.r_center = np.sqrt(
            (xx - self.grid[self.center_i])**2 + 
            (yy - self.grid[self.center_j])**2
        )
        self.create_stage_masks()
        
        # 添加最外圈零值约束
        self.outer_ring_mask = (np.abs(xx) >= 14) | (np.abs(yy) >= 14)
        
        return hi_res_data
    
    def downsample_beam(self, hi_res_matrix, target_resolution=31):
        """将高分辨率矩阵降采样到原始分辨率"""
        # 确保原始分辨率边界存在
        if not hasattr(self, 'original_grid_bound'):
            self.original_grid_bound = self.grid_bound
            self.log(f"警告: 恢复降采样时的原始边界 {self.original_grid_bound}mm")
        
        # 创建高分辨率网格
        hi_res_size = hi_res_matrix.shape[0]
        hi_res_grid = np.linspace(-self.original_grid_bound, self.original_grid_bound, hi_res_size)
        
        # 创建目标分辨率网格
        target_grid = np.linspace(-self.original_grid_bound, self.original_grid_bound, target_resolution)
        
        # 创建插值器
        interp = RegularGridInterpolator(
            (hi_res_grid, hi_res_grid),
            hi_res_matrix,
            method='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 重建原始分辨率掩码
        xx, yy = np.meshgrid(target_grid, target_grid, indexing='ij')
        outer_mask = (np.abs(xx) >= 14) | (np.abs(yy) >= 14)
        
        # 执行降采样
        points = np.array([[x, y] for x in target_grid for y in target_grid])
        lo_res_data = interp(points).reshape(target_resolution, target_resolution)
        
        # 确保边界为零
        lo_res_data[outer_mask] = 0
        
        return lo_res_data
  
    
    def plot_resolution_comparison(self, initial_beam, pre_hi_res, final_result):
        """绘制分辨率对比图，显示优化过程的结果演进"""
        try:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle("高分辨率优化过程对比", fontsize=16)
            
            # 1. 初始低分辨率束流
            im1 = axes[0].imshow(initial_beam, cmap="viridis", extent=[-15, 15, -15, 15])
            axes[0].set_title(f"初始 (31x31)")
            fig.colorbar(im1, ax=axes[0])
            
            # 2. 高分辨率优化前（原始分辨率）
            lo_res_pre = self.downsample_beam(pre_hi_res, initial_beam.shape[0])
            im2 = axes[1].imshow(lo_res_pre, cmap="viridis", extent=[-15, 15, -15, 15])
            axes[1].set_title(f"高分辨率优化前")
            fig.colorbar(im2, ax=axes[1])
            
            # 3. 高分辨率优化结果
            axes[2].set_title(f"高分辨率优化结果 (121x121)")
            hi_res_normalized = pre_hi_res / np.max(pre_hi_res)
            # 只显示中心区域 (x/y from -10 to 10) 避免空白区域过多
            axes[2].imshow(hi_res_normalized, cmap="viridis", 
                        extent=[-15, 15, -15, 15])
            axes[2].set_xlim(-10, 10)
            axes[2].set_ylim(-10, 10)
            
            # 4. 最终结果（降采样回31x31）
            im4 = axes[3].imshow(final_result, cmap="viridis", extent=[-15, 15, -15, 15])
            axes[3].set_title(f"最终优化结果 (31x31)")
            fig.colorbar(im4, ax=axes[3])
            
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.savefig("resolution_comparison.png", bbox_inches='tight')
            self.log("分辨率比较图已保存")
            
        except Exception as e:
            self.log(f"分辨率对比图创建失败: {str(e)}")


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
        if not hasattr(self, 'distance_from_center'):
            xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
            self.distance_from_center = np.sqrt(xx**2 + yy**2)
        
        # 确保outer_ring_mask存在（修复高分辨率后丢失的问题）
        if not hasattr(self, 'outer_ring_mask') or self.outer_ring_mask is None:
            xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
            self.outer_ring_mask = (np.abs(xx) >= 14) | (np.abs(yy) >= 14)
        
        # 从优化区域排除最外圈零值区
        self.optimization_mask = (self.distance_from_center <= self.opt_radius) & ~self.outer_ring_mask
        
        # 添加详细的尺寸日志
        self.log(f"优化掩码创建 - 网格尺寸: {self.grid.shape}, 掩码尺寸: {self.optimization_mask.shape}")
        self.log(f"优化区域包含 {np.sum(self.optimization_mask)} 个点 (半径 ≤ {self.opt_radius}mm)")



    def create_stage_masks(self):
        """创建11个阶段的点集掩膜"""
        self.stage_masks = []
        total_mask = np.zeros((self.grid_points, self.grid_points), dtype=bool)
        
        # 生成所有网格坐标
        xx, yy = np.meshgrid(range(self.grid_points), range(self.grid_points), indexing='ij')
        
        # 第一阶段 (集合1): x=0 或 y=0 的点
        stage0_mask = ((np.abs(self.grid[xx]) < 1e-5) | (np.abs(self.grid[yy]) < 1e-5))
        self.stage_masks.append(stage0_mask)
        total_mask = total_mask | stage0_mask
        self.log(f"集合1 (阶段0): x=0或y=0的点, 包含点数: {np.sum(stage0_mask)}")
        
        # 第二阶段到第十阶段 (集合2~集合10): |x|=k 或 |y|=k 的点，排除之前阶段
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
            self.log(f"集合{k+1} (阶段{k}: |x|={k}或|y|={k}的点, 包含点数: {np.sum(stage_mask)}")
        
        # 第十一阶段 (集合11): 剩余所有点
        stage10_mask = ~total_mask
        self.stage_masks.append(stage10_mask)
        self.log(f"集合11 (阶段10): 剩余点, 包含点数: {np.sum(stage10_mask)}")
        
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
    
    def enforce_radial_constraints(self, beam_matrix, strict=False):
        """
        强制执行径向约束 - 强化版本
        strict=True: 使用更严格的约束
        返回修正后的矩阵和修正点数量
        """
        rows, cols = beam_matrix.shape
        
        # 更新中心点位置（如果最大值位置改变）
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        self.center_i, self.center_j = center_i, center_j
        center_pos = (self.grid[center_i], self.grid[center_j])
        max_val = beam_matrix[center_i, center_j]
        
        # 记录调整点数
        modified_points = 0
        
        # 1. 确保中心点位于物理中心
        center_distance = np.sqrt(center_pos[0]**2 + center_pos[1]**2)
        if center_distance > 1.5:
            self.log(f"中心点({center_pos[0]:.2f}, {center_pos[1]:.2f})偏离过大({center_distance:.2f}mm)，调整到附近最高点")
            
            # 在-1.0~1.0mm范围内寻找最大值
            region_i = np.where((np.abs(self.grid) < 1.0))[0]
            region_j = np.where((np.abs(self.grid) < 1.0))[0]
            
            if region_i.size > 0 and region_j.size > 0:
                region_values = beam_matrix[region_i[:, None], region_j]
                max_in_center = np.max(region_values)
                max_idx = np.unravel_index(np.argmax(region_values), (region_i.size, region_j.size))
                center_i, center_j = region_i[max_idx[0]], region_j[max_idx[1]]
                max_val = max_in_center
                center_pos = (self.grid[center_i], self.grid[center_j])
                self.log(f"新中心点位置: ({center_pos[0]:.2f}, {center_pos[1]:.2f}), 值={max_val:.4f}")
        
        # 2. 创建距离矩阵（以中心点为原点）
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt((xx - center_pos[0])**2 + (yy - center_pos[1])**2)
        
        # 3. 中心区域（半径≤5mm）禁止零值点
        inner_region = r <= 5.0
        zero_points_mask = beam_matrix < max_val * 0.01
        inner_zero_points = np.logical_and(inner_region, zero_points_mask)
        zero_points_count = np.sum(inner_zero_points)
        
        if zero_points_count > 0:
            base_value = max_val * 0.02
            beam_matrix[inner_zero_points] = base_value
            modified_points += zero_points_count
            self.log(f"禁止零值点: 设置 {zero_points_count} 个点至少为 {base_value:.4f}")
        
        # 4. 沿射线方向强制单调递减
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)  # 36个方向
        decay_factor = 0.96  # 每单位距离的衰减因子
        
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            max_radius = self.opt_radius * 1.2
            
            # 计算射线上的点
            ray_points = []
            distances = np.linspace(0, max_radius, 100)  # 100个点
            for dist in distances:
                x_pos = center_pos[0] + dx * dist
                y_pos = center_pos[1] + dy * dist
                # 找到最近的网格点
                x_idx = np.argmin(np.abs(self.grid - x_pos))
                y_idx = np.argmin(np.abs(self.grid - y_pos))
                ray_points.append((x_idx, y_idx, dist))
            
            # 移除重复点
            unique_points = {}
            for x_idx, y_idx, dist in ray_points:
                key = (x_idx, y_idx)
                if key not in unique_points or dist < unique_points[key][0]:
                    unique_points[key] = (dist, x_idx, y_idx)
            
            # 按距离排序
            sorted_points = sorted(unique_points.values(), key=lambda x: x[0])
            
            # 应用单调递减约束
            prev_val = max_val
            for idx, (dist, x_idx, y_idx) in enumerate(sorted_points):
                current_val = beam_matrix[x_idx, y_idx]
                
                # 计算衰减目标值
                target_val = max_val * np.exp(-dist * (1 - decay_factor) / decay_factor)
                
                # 确保当前值不超过前一个点且符合目标值
                if current_val > min(prev_val, target_val * 1.2):
                    new_val = min(prev_val * 0.99, target_val * (1.0 + 0.1 * np.random.randn()))
                    beam_matrix[x_idx, y_idx] = max(0, min(new_val, prev_val))
                    modified_points += 1
                
                # 更新前一个值
                prev_val = beam_matrix[x_idx, y_idx]
        
        # 5. 确保所有值非负
        beam_matrix = np.maximum(beam_matrix, 0)
        
        beam_matrix[self.outer_ring_mask] = 0

        return beam_matrix, modified_points

    def enforce_unimodal_row_col_constraints(self, beam_matrix):
        """
        强制执行行/列单峰约束，确保每个行和列只有一个峰值
        返回修正后的矩阵和行违规数、列违规数
        """
        rows, cols = beam_matrix.shape
        center_i, center_j = self.center_i, self.center_j
        
        row_violations = 0
        col_violations = 0
        
        # 对每一行应用单峰约束
        for i in range(rows):
            row = beam_matrix[i, :].copy()
            current_peak = center_j  # 初始假设峰值为中心列
            
            # 寻找实际峰值位置
            peak_idx = np.argmax(row)
            
            # 检查峰值位置是否合理（应在中心附近）
            if abs(peak_idx - center_j) > 3:  # 允许偏离中心点3个位置
                self.log(f"行{i}峰值偏离较大: 在位置{peak_idx} (中心:{center_j})")
                peak_idx = center_j  # 强制设为中心点位置
            
            # 检查峰值是否为局部极值点
            if not self.is_peak(row, peak_idx):
                # 调整到最近的合格位置
                peak_idx = self.find_nearest_peak(row, peak_idx)
                row[peak_idx] = np.max(row)  # 恢复最大值
                row_violations += 1
            
            # 确保从左到峰值位置非递减
            for j in range(1, peak_idx + 1):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1] * 0.99  # 轻微递减允许
                    row_violations += 1
            
            # 确保从峰值位置到右端非递增
            for j in range(peak_idx, cols - 1):
                if row[j + 1] > row[j]:
                    row[j + 1] = row[j] * 0.99  # 轻微递减允许
                    row_violations += 1
            
            beam_matrix[i, :] = row
        
        # 对每一列应用单峰约束
        for j in range(cols):
            col = beam_matrix[:, j].copy()
            current_peak = center_i  # 初始假设峰值为中心行
            
            # 寻找实际峰值位置
            peak_idx = np.argmax(col)
            
            # 检查峰值位置是否合理
            if abs(peak_idx - center_i) > 3:
                self.log(f"列{j}峰值偏离较大: 在位置{peak_idx} (中心:{center_i})")
                peak_idx = center_i
            
            # 检查峰值是否为局部极值点
            if not self.is_peak(col, peak_idx):
                peak_idx = self.find_nearest_peak(col, peak_idx)
                col[peak_idx] = np.max(col)
                col_violations += 1
            
            # 确保从上到峰值位置非递减
            for i in range(1, peak_idx + 1):
                if col[i] < col[i - 1]:
                    col[i] = col[i - 1] * 0.99
                    col_violations += 1
            
            # 确保从峰值位置到下端非递增
            for i in range(peak_idx, rows - 1):
                if col[i + 1] > col[i]:
                    col[i + 1] = col[i] * 0.99
                    col_violations += 1
            
            beam_matrix[:, j] = col
        
        # 确保最外圈保持为零

        beam_matrix[self.outer_ring_mask] = 0

        return beam_matrix, row_violations, col_violations
    
    def enforce_unimodal_for_optimized_rows_cols(self, beam_matrix, indices_dict):
        """仅检测和处理阶段优化相关行/列的单峰性"""
        rows_to_check = set(indices_dict[0])
        cols_to_check = set(indices_dict[1])
        
        row_violations = 0
        col_violations = 0
        
        # 检查相关行
        for i in rows_to_check:
            row = beam_matrix[i, :].copy()
            old_row = row.copy()
            
            # 寻找峰值位置
            peak_idx = np.argmax(row)
            
            # 确保从左到峰值位置非递减
            for j in range(1, peak_idx + 1):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1] * 0.99
                    row_violations += 1
            
            # 确保从峰值位置到右端非递增
            for j in range(peak_idx, self.grid_points - 1):
                if row[j + 1] > row[j]:
                    row[j + 1] = row[j] * 0.99
                    row_violations += 1
            
            if not np.array_equal(old_row, row):
                beam_matrix[i, :] = row
        
        # 检查相关列
        for j in cols_to_check:
            col = beam_matrix[:, j].copy()
            old_col = col.copy()
            
            # 寻找峰值位置
            peak_idx = np.argmax(col)
            
            # 确保从上到峰值位置非递减
            for i in range(1, peak_idx + 1):
                if col[i] < col[i - 1]:
                    col[i] = col[i - 1] * 0.99
                    col_violations += 1
            
            # 确保从峰值位置到下端非递增
            for i in range(peak_idx, self.grid_points - 1):
                if col[i + 1] > col[i]:
                    col[i + 1] = col[i] * 0.99
                    col_violations += 1
            
            if not np.array_equal(old_col, col):
                beam_matrix[:, j] = col
        
        return row_violations, col_violations
    

    def enforce_convexity_constraints(self, beam_matrix):
        """
        强制执行等高线凸性约束 - 确保所有等高线都是凸多边形
        使用凸包算法修正非凸等高线
        """
        beam_matrix = np.maximum(beam_matrix, 0)  # 确保非负
        
        # 生成等高线层级
        min_val = np.min(beam_matrix[beam_matrix > 0])
        max_val = np.max(beam_matrix)
        contour_levels = np.linspace(min_val * 1.1, max_val * 0.95, 10)
        
        # 创建等高线图像
        normalized = ((beam_matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # 检查每个层级的等高线
        for level in contour_levels:
            # 创建当前层级的二值图像
            threshold = (level - min_val) / (max_val - min_val) * 255
            _, binary_img = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
            
            # 查找等高线
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 简化等高线（减少点数）
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:  # 不足3个点无法形成多边形
                    continue
                
                # 检查凸性
                if not cv2.isContourConvex(approx):
                    # 计算凸包
                    hull = cv2.convexHull(approx)
                    
                    # 创建凸包图像
                    mask = np.zeros_like(binary_img)
                    cv2.drawContours(mask, [hull], -1, 255, -1)
                    
                    # 获取凸包内部点
                    y_idx, x_idx = np.where(mask > 0)
                    
                    # 获取凸包点的网格坐标
                    grid_x, grid_y = self.grid, self.grid
                    hull_coords = []
                    
                    for point in hull[:, 0]:
                        # 找到最近的网格点
                        idx_x = np.argmin(np.abs(grid_x - point[0]))
                        idx_y = np.argmin(np.abs(grid_y - point[1]))
                        hull_coords.append((idx_y, idx_x))
                    
                    # 确保等高线凸性边界内部的值至少等于当前层级
                    for y, x in zip(y_idx, x_idx):
                        if beam_matrix[y, x] < level:
                            beam_matrix[y, x] = level * np.random.uniform(0.98, 1.02)
        
        beam_matrix[self.outer_ring_mask] = 0

        return beam_matrix
    
    def check_convex_violations(self, beam_matrix):
        """
        检查等高线凸性违规
        返回违规的等高线数量
        """
        beam_matrix = np.maximum(beam_matrix, 0)  # 确保非负
        violation_count = 0
        
        # 生成等高线层级
        min_val = np.min(beam_matrix[beam_matrix > 0])
        max_val = np.max(beam_matrix)
        contour_levels = np.linspace(min_val * 1.1, max_val * 0.95, 10)
        
        # 创建等高线图像
        normalized = ((beam_matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        # 检查每个层级的等高线凸性
        for level in contour_levels:
            # 创建当前层级的二值图像
            threshold = (level - min_val) / (max_val - min_val) * 255
            _, binary_img = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
            
            # 查找等高线
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 简化等高线
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                
                # 检查凸性
                if not cv2.isContourConvex(approx):
                    violation_count += 1
        
        return violation_count

    def is_peak(self, array, index):
        """
        检查给定索引是否为局部极大值点
        """
        n = len(array)
        # 边界检查
        if index == 0:
            return array[index] >= array[index + 1]
        elif index == n - 1:
            return array[index] >= array[index - 1]
        else:
            return (array[index] >= array[index - 1] and 
                    array[index] >= array[index + 1])

    def find_nearest_peak(self, array, current_index):
        """
        查找最近的合格峰值位置
        """
        n = len(array)
        # 向左找
        left_peak = current_index
        for i in range(current_index - 1, -1, -1):
            if self.is_peak(array, i):
                left_peak = i
                break
        
        # 向右找
        right_peak = current_index
        for i in range(current_index + 1, n):
            if self.is_peak(array, i):
                right_peak = i
                break
        
        # 返回最近的峰值位置
        left_dist = current_index - left_peak
        right_dist = right_peak - current_index
        
        if left_dist <= right_dist:
            return left_peak
        else:
            return right_peak

    def check_unimodal_violations(self, beam_matrix):
        """
        检查行/列单峰违规情况
        返回行违规数，列违规数
        """
        rows, cols = beam_matrix.shape
        row_violations = 0
        col_violations = 0
        
        # 检查行违规
        for i in range(rows):
            row = beam_matrix[i, :]
            peak = np.argmax(row)
            # 检查左边是否递减
            for j in range(1, peak):
                if row[j] < row[j-1]:
                    row_violations += 1
            # 检查右边是否递增
            for j in range(peak, cols-1):
                if row[j+1] > row[j]:
                    row_violations += 1
        
        # 检查列违规
        for j in range(cols):
            col = beam_matrix[:, j]
            peak = np.argmax(col)
            # 检查上边是否递减
            for i in range(1, peak):
                if col[i] < col[i-1]:
                    col_violations += 1
            # 检查下边是否递增
            for i in range(peak, rows-1):
                if col[i+1] > col[i]:
                    col_violations += 1
        
        return row_violations, col_violations

    def simulate_etching(self, beam_matrix_normalized, direction):
        """
        修改后的模拟刻蚀轮廓方法，支持不同分辨率
        """
        # 使用归一化的束流乘以最大值得到实际刻蚀速率
        actual_beam = beam_matrix_normalized * self.max_val
        
        # 当前网格间距
        dx = self.grid_spacing
        
        # 创建插值器
        interpolator = RegularGridInterpolator(
            (self.grid, self.grid),
            actual_beam,
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
        
        profile = np.zeros_like(self.grid)
        
        if direction == "x":  # 束沿Y轴移动 -> 测量沿X方向的轮廓
            # 对于高分辨率网格，使用每2个点采样一次以提高效率
            step = 2 if len(self.grid) > 50 else 1
            for idx in range(0, len(self.grid), step):
                y_pos = self.grid[idx]
                path_points = np.column_stack((self.grid, np.full_like(self.grid, y_pos)))
                etch_rates = interpolator(path_points)
                
                # 计算积分结果后乘以1/30
                integral_result = trapezoid(etch_rates, dx=dx)
                profile[idx] = integral_result * (1/self.scan_velocity)
            
            # 如果需要，平滑结果
            if step > 1:
                profile = np.interp(self.grid, self.grid[::step], profile[::step])
        
        else:  # direction == "y": 束沿X轴移动 -> 测量沿Y方向的轮廓
            # 对于高分辨率网格，使用每2个点采样一次以提高效率
            step = 2 if len(self.grid) > 50 else 1
            for idx in range(0, len(self.grid), step):
                x_pos = self.grid[idx]
                path_points = np.column_stack((np.full_like(self.grid, x_pos), self.grid))
                etch_rates = interpolator(path_points)
                
                # 计算积分结果后乘以1/30
                integral_result = trapezoid(etch_rates, dx=dx)
                profile[idx] = integral_result * (1/self.scan_velocity)
            
            # 如果需要，平滑结果
            if step > 1:
                profile = np.interp(self.grid, self.grid[::step], profile[::step])

        return profile



    def mutate_beam(self, beam_matrix, magnitude, sim_x, sim_y):
        new_beam = beam_matrix.copy()
        current_mask = self.stage_masks[self.current_stage_idx]
        
        # 获取当前阶段优化区域的坐标
        indices = np.where(current_mask)
        positions = list(zip(indices[0], indices[1]))
        np.random.shuffle(positions)  # 随机打乱顺序
        
        # ==== 核心修改：针对每个优化点进行迭代式约束修复 ====
        for i, j in positions:
            # 跳过变异概率
            if self.current_stage_idx < 5 and np.random.rand() > 0.8: continue
            if self.current_stage_idx >= 5 and np.random.rand() > 0.6: continue
            
            # 获取原始值和距离
            original_val = new_beam[i, j]
            dist_to_center = self.r_center[i, j]    
            
            # 计算误差方向
            error = self.calculate_point_error(i, j, sim_x, sim_y)
            error_direction = 1 if error < 0 else -1
            
            # 计算变异幅度
            dist_factor = np.exp(-dist_to_center / 6.0)
            effective_magnitude = magnitude * dist_factor * (0.8 + 0.4 * np.random.rand())
            mutation = error_direction * effective_magnitude
            new_val = original_val + mutation
            
            # 应用距离限制
            if dist_to_center < 5.0: new_val = np.clip(new_val, 0.1, 1.0)
            elif dist_to_center < 10.0: new_val = np.clip(new_val, 0, 0.7)
            else: new_val = np.clip(new_val, 0, 0.4)
            
            new_beam[i, j] = new_val
            
            # ==== 时刻保持约束1：局部径向约束 ====
            # 确保优化点满足径向减小的要求
            center_val = new_beam[self.center_i, self.center_j]
            if dist_to_center < 5.0 and new_val < 0.05 * center_val:
                new_beam[i, j] = 0.05 * center_val  # 设置最小值
            
            # 确保不会出现比近中心点更大的值
            for r_offset in range(1, 5):
                for angle in np.linspace(0, 2*np.pi, 8):
                    closer_i = int(self.center_i + r_offset * np.cos(angle))
                    closer_j = int(self.center_j + r_offset * np.sin(angle))
                    if (0 <= closer_i < self.grid_points and 0 <= closer_j < self.grid_points and 
                        self.r_center[closer_i, closer_j] < dist_to_center and
                        new_beam[i, j] > new_beam[closer_i, closer_j]):
                        new_beam[i, j] = new_beam[closer_i, closer_j] * 0.95
        
        # ==== 时刻保持约束2：单峰性约束 ====
        row_violations, col_violations = self.enforce_unimodal_for_optimized_rows_cols(
            new_beam, indices)
        
        # ==== 时刻保持约束4：最外圈零值 (约束3在最后才应用) ====
        new_beam[self.outer_ring_mask] = 0
        
        return new_beam, len(positions), row_violations, col_violations
    


    def calculate_point_error(self, i, j, sim_x, sim_y):
        """
        计算单个网格点(i,j)的误差
        返回标量误差值（单个数值）- 修复分辨率兼容性问题
        主要修复：移除文件重载逻辑，直接使用实验数据属性
        """
        # 网格点上的坐标
        grid = self.grid
        pos_x = grid[j]
        pos_y = grid[i]
        
        # 直接使用已加载的实验数据
        exp_x_data = self.beam_traced_y_axis  # 束沿Y移动时测量的X截面
        exp_y_data = self.beam_traced_x_axis  # 束沿X移动时测量的Y截面
        
        # 提取位置和值
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        
        # 检查并适应分辨率
        if len(grid) != len(exp_x_pos):
            # 模拟数据插值到实验数据位置
            if not hasattr(sim_x, "__len__") or len(sim_x) != len(grid):
                self.log(f"警告: 无效的模拟数据X尺寸: {len(sim_x)} != {len(grid)}")
                sim_x_interp = np.zeros_like(exp_x_pos)
            else:
                sim_x_interp = np.interp(exp_x_pos, grid, sim_x)
            
            if not hasattr(sim_y, "__len__") or len(sim_y) != len(grid):
                self.log(f"警告: 无效的模拟数据Y尺寸: {len(sim_y)} != {len(grid)}")
                sim_y_interp = np.zeros_like(exp_y_pos)
            else:
                sim_y_interp = np.interp(exp_y_pos, grid, sim_y)
        else:
            sim_x_interp = sim_x
            sim_y_interp = sim_y
        
        # 计算权重
        sigma = 1.5  # 高斯权重标准偏差
        x_weights = np.exp(-(exp_x_pos - pos_x)**2 / (2 * sigma**2))
        y_weights = np.exp(-(exp_y_pos - pos_y)**2 / (2 * sigma**2))
        
        # 计算加权误差
        try:
            x_error = np.sum(np.abs(sim_x_interp - exp_x_val) * x_weights) / np.sum(x_weights)
            y_error = np.sum(np.abs(sim_y_interp - exp_y_val) * y_weights) / np.sum(y_weights)
            
            return (x_error + y_error) / 2
        except Exception as e:
            self.log(f"计算点加权误差出错: {str(e)}")
            return 0.0  # 返回默认零误差



    
    def calculate_global_error(self, sim_x, sim_y):
        """
        修改后的全局误差计算方法
        """
        # 处理束沿Y移动时的X方向轮廓
        exp_x_data = self.beam_traced_y_axis
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        
        # 插值模拟数据到实验点位置
        sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x)
        
        # 计算X误差（使用绝对误差）
        abs_dev_x = np.abs(sim_x_interp - exp_x_val)
        
        # 处理束沿X移动时的Y方向轮廓
        exp_y_data = self.beam_traced_x_axis
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        
        # 插值模拟数据到实验点位置
        sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y)
        
        # 计算Y误差（使用绝对误差）
        abs_dev_y = np.abs(sim_y_interp - exp_y_val)
        
        # 计算相对误差（基于均值）
        mean_exp_x = np.mean(exp_x_val)
        mean_exp_y = np.mean(exp_y_val)
        
        # 使用均方根误差作为相对误差（更为稳定）
        rel_err_x = np.sqrt(np.mean((sim_x_interp - exp_x_val)**2)) / mean_exp_x * 100 if mean_exp_x > 0 else 100.0
        rel_err_y = np.sqrt(np.mean((sim_y_interp - exp_y_val)**2)) / mean_exp_y * 100 if mean_exp_y > 0 else 100.0
        
        # 计算综合绝对误差
        abs_error = (np.mean(abs_dev_x) + np.mean(abs_dev_y)) / 2
        
        return abs_error, rel_err_x, rel_err_y 
    

    def calculate_magnitude(self, current_iter, max_iters, stage_idx):
        """动态计算变异幅度"""
        # 基础公式
        min_mag = 0.015  # 最小变异幅度
        max_mag = 0.35   # 最大变异幅度
        
        # 阶段自适应调整
        stage_factor = 1.0 - (stage_idx / (self.num_stages - 1)) * 0.6
        
        # 迭代衰减
        decay_factor = max(0.2, 1.0 - current_iter / max_iters)
        
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
            stage_name = "集合1 (坐标轴点)"
        elif stage_idx < 10:
            stage_name = f"集合{stage_idx+1} (|坐标|={stage_idx}点)"
        else:
            stage_name = "集合11 (剩余点)"
        
        self.log(f"\n=== 开始优化阶段 {stage_idx+1}/{self.num_stages}: {stage_name} ===")
        self.log(f"当前阶段包含点数: {np.sum(self.stage_masks[stage_idx])}")
        
        # 检查当前矩阵的单峰性
        row_violations, col_violations = self.check_unimodal_violations(current_matrix)
        peak_violations = row_violations + col_violations
        self.history["peak_violations"].append(peak_violations)
        self.log(f"阶段开始时峰违规: {row_violations}行, {col_violations}列")
        
        # 检查当前矩阵的凸性违规
        convex_violations = self.check_convex_violations(current_matrix)
        self.history["convex_violations"].append(convex_violations)
        self.log(f"阶段开始时凸性违规: {convex_violations}处")
        
        # 获取当前最佳模拟轮廓
        try:
            sim_x = self.simulate_etching(current_matrix, "x")
            sim_y = self.simulate_etching(current_matrix, "y")
            abs_error, rel_err_x, rel_err_y = self.calculate_global_error(sim_x, sim_y)
        except Exception as e:
            self.log(f"模拟误差计算失败: {str(e)}")
            raise
        
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
            try:
                candidate, radial_modified, row_violations, col_violations = self.mutate_beam(
                    best_matrix, magnitude, sim_x, sim_y)
            except Exception as e:
                self.log(f"变异失败: {str(e)}")
                # 添加详细错误日志
                self.log(f"变异失败: {str(e)}")
                self.log(f"当前阶段: {self.current_stage_idx}, 迭代: {iteration}")
                self.log(f"网格尺寸: {self.grid.shape}, 优化对象尺寸: {best_matrix.shape}")
                self.log(f"掩码尺寸: {self.optimization_mask.shape if hasattr(self, 'optimization_mask') and self.optimization_mask is not None else '无掩码'}")
                break  # 跳出当前迭代
                
            
            # 记录约束应用情况
            self.log(f"约束应用: 径向调整={radial_modified}点, 行违规={row_violations}, 列违规={col_violations}")
            
            # 冻结之前优化过的区域
            candidate[frozen_mask] = best_matrix[frozen_mask]
            
            # 评估候选
            try:
                cand_sim_x = self.simulate_etching(candidate, "x")
                cand_sim_y = self.simulate_etching(candidate, "y")
                cand_abs_error, cand_rel_err_x, cand_rel_err_y = self.calculate_global_error(cand_sim_x, cand_sim_y)
            except Exception as e:
                self.log(f"评估候选失败: {str(e)}")
                # 添加详细错误日志
                self.log(f"评估候选失败: {str(e)}")
                self.log(f"当前阶段: {self.current_stage_idx}, 迭代: {iteration}")
                self.log(f"网格尺寸: {self.grid.shape}, 候选尺寸: {candidate.shape}")
                self.log(f"模拟X尺寸: {sim_x.shape if hasattr(sim_x, 'shape') else '无数据'}")
                self.log(f"模拟Y尺寸: {sim_y.shape if hasattr(sim_y, 'shape') else '无数据'}")
                break
            
            # 检查是否改进
            if cand_abs_error < best_abs_error:
                improvement = best_abs_error - cand_abs_error
                self.log(f"改进: Δ={improvement:.5f}, 新绝对误差={cand_abs_error:.4f}")
                self.log(f"相对误差变化: 束X动时Y向={cand_rel_err_x - self.history['rel_err_x'][-1]:+.1f}%, "
                        f"束Y动时X向={cand_rel_err_y - self.history['rel_err_y'][-1]:+.1f}%")
                
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
                
                # 检查优化后的单峰性
                row_violations, col_violations = self.check_unimodal_violations(best_matrix)
                peak_violations = row_violations + col_violations
                self.history["peak_violations"].append(peak_violations)
                self.log(f"峰违规检查: {row_violations}行, {col_violations}列")
                
                # 检查优化后的凸性
                convex_violations = self.check_convex_violations(best_matrix)
                self.history["convex_violations"].append(convex_violations)
                self.log(f"凸性违规检查: {convex_violations}处")
                
                # 保存阶段结果
                self.optimized_beam = best_matrix
            else:
                stagnation_count += 1
                self.log(f"无改进，维持绝对误差: {best_abs_error:.4f}, 已连续 {stagnation_count} 次")
            
            # 检查收敛
            if best_abs_error < 0.01:
                self.log(f"阶段收敛：绝对误差{best_abs_error:.4f} < 0.01")
                break
            elif stagnation_count >= max(2, max_iterations//2):  # 提前退出机制
                self.log(f"连续{stagnation_count}次无改进，结束当前阶段优化")
                break
        
        # 最终确保无峰违规
        row_violations, col_violations = self.check_unimodal_violations(best_matrix)
        if row_violations + col_violations > 0:
            best_matrix, _, _ = self.enforce_unimodal_row_col_constraints(best_matrix)
            self.log(f"阶段结束前强制修正峰违规: 行={row_violations}, 列={col_violations}")
        
        # 最终确保无凸性违规
        convex_violations = self.check_convex_violations(best_matrix)
        if convex_violations > 0:
            best_matrix = self.enforce_convexity_constraints(best_matrix)
            self.log(f"阶段结束前强制修正凸性违规: {convex_violations}处")
        
        self.log(f"阶段完成: 最佳绝对误差={best_abs_error:.4f}")
        return best_matrix

    def run_optimization(self):
        """运行11阶段优化过程，包含高分辨率优化"""
        # 初始评估
        current_matrix = self.optimized_beam.copy()
        sim_x = self.simulate_etching(current_matrix, "x")
        sim_y = self.simulate_etching(current_matrix, "y")
        abs_error0, rel_err_x0, rel_err_y0 = self.calculate_global_error(sim_x, sim_y)
        
        # 初始峰违规检查
        row_violations0, col_violations0 = self.check_unimodal_violations(current_matrix)
        peak_violations0 = row_violations0 + col_violations0
        
        # 初始凸性违规检查
        convex_violations0 = self.check_convex_violations(current_matrix)
        
        # 添加初始状态到历史记录
        self.history["iteration"].append(0)
        self.history["abs_error"].append(abs_error0)
        self.history["rel_err_x"].append(rel_err_x0)
        self.history["rel_err_y"].append(rel_err_y0)
        self.history["peak_violations"].append(peak_violations0)
        self.history["convex_violations"].append(convex_violations0)
        
        start_time = time.time()
        
        #######################
        # 步骤1: 原始分辨率热身优化 (运行3-5轮迭代)
        #######################
        self.log("进行原始分辨率热身优化 (3轮迭代)")
        for _ in range(3):
            # 只运行核心区域的优化
            for stage_idx in range(3):  # 前3个阶段是核心区域
                current_matrix = self.optimize_stage(current_matrix, stage_idx, 1)
        
        #######################
        # 步骤2: 上采样到高分辨率
        #######################
        hi_res_matrix = self.upsample_beam(current_matrix, new_resolution=121)
        
        # === 关键修复：彻底删除所有依赖网格的属性 ===
        # 删除所有依赖网格的属性，强制重新创建
        for attr in ['distance_from_center', 'optimization_mask', 
                    'outer_ring_mask', 'r_center', 
                    'stage_masks', 'grid', 'grid_spacing',
                    'peak_check_grid', 'r_center']:
            if hasattr(self, attr):
                delattr(self, attr)
                self.log(f"[分辨率切换] 已删除属性: {attr}")

        # 保存原始分辨率数据
        self.original_resolution = 31
        self.original_grid_bound = self.grid_bound

        # 保存原始最大刻蚀速率
        orig_max_val = self.max_val

        # 更新为高分辨率系统
        self.optimized_beam = hi_res_matrix
        self.high_res_mode = True

        #######################
        # 步骤3: 在高分辨率上运行完整优化
        #######################
        # === 关键修复：重新创建高分辨率网格系统 ===
        self.grid_points = 121
        self.grid_spacing = 2 * self.grid_bound / (self.grid_points - 1)
        self.grid = np.linspace(-self.grid_bound, self.grid_bound, self.grid_points)
        
        self.log(f"[分辨率切换] 创建新高分辨率网格: {self.grid_points}点, 边界±{self.grid_bound}mm, 间距{self.grid_spacing:.4f}mm")
        
        # 构建完整的网格系统坐标系
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        
        # 重新创建所有关键属性（按正确顺序）
        # 1. 外圈零值区掩膜
        self.outer_ring_mask = (np.abs(xx) >= 14) | (np.abs(yy) >= 14)
        outer_ring_count = np.sum(self.outer_ring_mask)
        self.log(f"[分辨率切换] 外圈零值区点数: {outer_ring_count}")
        
        # 2. 中心距离矩阵
        self.distance_from_center = np.sqrt(xx**2 + yy**2)
        
        # 3. 更新中心点位置
        self.center_i, self.center_j = self.find_center(self.optimized_beam)
        center_pos = (self.grid[self.center_i], self.grid[self.center_j])
        self.log(f"[分辨率切换] 新中心点位置: ({center_pos[0]:.2f}, {center_pos[1]:.2f})")
        
        # 4. 径向距离矩阵（基于中心点）
        self.r_center = np.sqrt(
            (xx - center_pos[0])**2 + 
            (yy - center_pos[1])**2
        )
        
        # 5. 优化掩膜
        self.create_optimization_mask()
        
        # 6. 阶段掩膜
        self.create_stage_masks()
        
        # 强制重新计算单峰约束相关矩阵
        delattr(self, 'peak_check_grid') if hasattr(self, 'peak_check_grid') else None
        
        # === 关键修复：确保使用高分辨率束流 ===
        current_matrix = self.optimized_beam
        
        # === 关键修复：更新阶段终点的记录 ===
        self.stage_ends = []
        self.current_stage_idx = 0
        
        #######################
        # 步骤3（续）: 执行高分辨率优化
        #######################
        # 确保标志已设置为高分辨率模式
        self.high_res_mode = True

        # === 记录实验数据尺寸 ===
        if hasattr(self, 'beam_traced_y_axis'):
            self.log(f"[分辨率切换] X实验数据尺寸: {self.beam_traced_y_axis.shape}")
        if hasattr(self, 'beam_traced_x_axis'):
            self.log(f"[分辨率切换] Y实验数据尺寸: {self.beam_traced_x_axis.shape}")
        
        # 确保实验数据点与网格点数量一致
        # 如果网格是121点，实验数据是31点，则重新加载
        # 但根据日志，似乎已经是121点，所以不需要额外处理）

        # 设置每个阶段的迭代次数（适应高分辨率）
        hi_res_stage_iterations = [
            20,  # 核心轴线
            18, 16, 14, 12, # 接近中心区域
            10, 8, 6, 4,    # 中距离区域
            15   # 剩余点（最多）
        ]
        
        for stage_idx in range(self.num_stages):
            self.log(f"\n[高分辨率] 开始阶段 {stage_idx+1}/{self.num_stages}")
            max_iter = hi_res_stage_iterations[stage_idx] if stage_idx < len(hi_res_stage_iterations) else 10
            self.log(f"阶段 {stage_idx+1}将运行最多 {max_iter} 次迭代")
            
            current_matrix = self.optimize_stage(current_matrix, stage_idx, max_iter)
            self.stage_ends.append(len(self.history["iteration"]) - 1)
            
            # 阶段性保存
            np.save(f"temp_beam_stage_{stage_idx+1}", current_matrix)
            
            #######################
            # 步骤4: 降采样回原始分辨率
            #######################
            low_res_optimized = self.downsample_beam(current_matrix, self.original_resolution)
            
            # 重建原始分辨率系统
            self.original_grid_points = self.original_resolution
            self.grid = np.linspace(-self.original_grid_bound, self.original_grid_bound, self.original_grid_points)
            self.grid_spacing = 2 * self.original_grid_bound / (self.original_grid_points - 1)
            
            # 重建网格依赖的属性
            xx, yy = np.meshgrid(self.grid, self.grid, indexing='ij')
            self.outer_ring_mask = (np.abs(xx) >= 14) | (np.abs(yy) >= 14)  # 重建31x31掩码
            
            # 必须先创建优化掩码后才能调用find_center
            self.create_optimization_mask()
            
            # 恢复原始最大刻蚀速率
            self.max_val = orig_max_val
            self.optimized_beam = low_res_optimized
            
            # 最终评估
            optimized_beam_full = self.optimized_beam * self.max_val
            
            # 保存结果
            np.savetxt("optimized_beam_distribution.csv", optimized_beam_full, delimiter=",")
            
            # 最终误差评估
            sim_x = self.simulate_etching(self.optimized_beam, "x")
            sim_y = self.simulate_etching(self.optimized_beam, "y")
            final_abs_error, final_rel_err_x, final_rel_err_y = self.calculate_global_error(sim_x, sim_y)
            
            # 最终峰违规检查
            row_violations, col_violations = self.check_unimodal_violations(self.optimized_beam)
            peak_violations = row_violations + col_violations

            # 在最终评估前应用凸性约束
            self.log("应用最终等高线凸性约束")
            self.optimized_beam = self.enforce_convexity_constraints(self.optimized_beam)
            
            # 最终凸性违规检查
            convex_violations = self.check_convex_violations(self.optimized_beam)
            if convex_violations > 0:
                self.log(f"警告: 最终凸性违规: {convex_violations}处")
                # 尝试最后一次修正
                self.optimized_beam = self.enforce_convexity_constraints(self.optimized_beam)
            
            elapsed_time = time.time() - start_time
            self.log(f"\n优化完成! 总迭代次数: {len(self.history['iteration'])}")
            self.log(f"总耗时: {elapsed_time:.1f}秒")
            self.log(f"最终绝对误差: {final_abs_error:.4f} (初始={abs_error0:.4f})")
            self.log(f"最终相对误差: 束X动时Y向={final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%)")
            self.log(f"              束Y动时X向={final_rel_err_y:.1f}% (初始={rel_err_y0:.1f}%)")
            self.log(f"最终峰违规: {row_violations}行, {col_violations}列")
            self.log(f"最终凸性违规: {convex_violations}处")
            
            # 绘制径向分布验证图
            self.plot_resolution_comparison(
                initial_beam=self.initial_beam,
                pre_hi_res=current_matrix,
                final_result=optimized_beam_full
            )
            
            return self.optimized_beam, final_rel_err_x, final_rel_err_y


    def plot_unimodal_verification(self, beam_full):
        """绘制行/列单峰性验证图"""
        try:
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle("束流分布行/列单峰性验证", fontsize=16)
            
            # 1. 示例行分布
            ax1 = plt.subplot(221)
            row_idx = self.grid_points // 2  # 中间行
            row_data = beam_full[row_idx, :]
            ax1.plot(self.grid, row_data, 'b-', linewidth=1.5)
            ax1.scatter(self.grid, row_data, c='r', s=20)
            
            # 标注峰值位置
            peak_j = np.argmax(row_data)
            ax1.axvline(self.grid[peak_j], color='g', linestyle='--', alpha=0.7)
            ax1.scatter([self.grid[peak_j]], [row_data[peak_j]], s=100, marker='x', c='g')
            
            ax1.set_title(f"Y={self.grid[row_idx]:.1f}mm 行束流分布")
            ax1.set_xlabel("X位置 (mm)")
            ax1.set_ylabel("刻蚀速率 (nm/s)")
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # 2. 示例列分布
            ax2 = plt.subplot(222)
            col_idx = self.grid_points // 2  # 中间列
            col_data = beam_full[:, col_idx]
            ax2.plot(self.grid, col_data, 'b-', linewidth=1.5)
            ax2.scatter(self.grid, col_data, c='r', s=20)
            
            # 标注峰值位置
            peak_i = np.argmax(col_data)
            ax2.axvline(self.grid[peak_i], color='g', linestyle='--', alpha=0.7)
            ax2.scatter([self.grid[peak_i]], [col_data[peak_i]], s=100, marker='x', c='g')
            
            ax2.set_title(f"X={self.grid[col_idx]:.1f}mm 列束流分布")
            ax2.set_xlabel("Y位置 (mm)")
            ax2.set_ylabel("刻蚀速率 (nm/s)")
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # 3. 峰偏离度图
            ax3 = plt.subplot(223)
            peak_deviations = np.zeros((self.grid_points, self.grid_points))
            center_x = self.grid[self.center_j]
            center_y = self.grid[self.center_i]
            
            # 计算每个点的峰偏离度
            for i in range(self.grid_points):
                row = beam_full[i, :]
                row_peak = np.argmax(row)
                peak_deviations[i, :] = np.abs(self.grid - self.grid[row_peak])
            
            # 绘制
            im3 = ax3.imshow(peak_deviations, cmap="viridis", extent=[-15, 15, -15, 15])
            plt.colorbar(im3, ax=ax3, label="峰位置偏离 (mm)")
            ax3.scatter([center_x], [center_y], s=100, marker='*', c='r', label='中心点')
            ax3.set_title("行峰位置偏离中心图")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            ax3.legend()
            
            # 4. 违规历史曲线
            if len(self.history["iteration"]) > 1:
                ax4 = plt.subplot(224)
                iterations = self.history["iteration"]
                abs_errors = self.history["abs_error"]
                peak_violations = self.history["peak_violations"]
                
                # 左侧Y轴 - 绝对误差
                ax4.plot(iterations, abs_errors, "k-", label="绝对误差")
                ax4.set_xlabel("迭代次数")
                ax4.set_ylabel("绝对误差", color='k')
                ax4.tick_params(axis='y', labelcolor='k')
                ax4.grid(True)
                
                # 右侧Y轴 - 峰违规
                ax4b = ax4.twinx()
                ax4b.plot(iterations, peak_violations, "r--", label="峰违规次数")
                ax4b.set_ylabel("峰违规次数", color='r')
                ax4b.tick_params(axis='y', labelcolor='r')
                ax4b.set_ylim(-1, max(peak_violations)*1.2)
                
                # 同时添加凸性违规曲线
                if "convex_violations" in self.history:
                    convex_violations = self.history["convex_violations"]
                    ax4b.plot(iterations, convex_violations, "b--", label="凸性违规次数")
                
                # 标记阶段转换
                for i, end_idx in enumerate(self.stage_ends):
                    ax4.axvline(end_idx, color=f'C{i}', linestyle='--', alpha=0.7)
                
                # 图例
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4b.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                ax4.set_title("约束违规历史")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig("unimodal_verification.png", bbox_inches='tight')
            self.log("束流单峰性验证图已保存")
            plt.close(fig)
        except Exception as e:
            self.log(f"单峰性验证图创建失败: {str(e)}")

    def plot_radial_distribution(self, beam_full):
        """绘制径向分布验证图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # 添加最外圈零值区标记
            outer_x, outer_y = np.where(self.outer_ring_mask)
            outer_pos_x = self.grid[outer_x]
            outer_pos_y = self.grid[outer_y]
            ax.scatter(outer_pos_x, outer_pos_y, 
                    color='gray', s=30, alpha=0.5,
                    label="零值区(|x|>=14或|y|>=14)")

            # 然后在绘制其他数据之前添加图例参考：
            ax.scatter([], [], color='gray', s=30, label="零值区")
            
            # 创建距离数组
            r = self.r_center.flatten()
            values = beam_full.flatten()
            
            # 添加中心点
            ax.scatter(0, beam_full[self.center_i, self.center_j], 
                      color='red', s=100, marker='*', 
                      label='中心点值')
            
            # 绘制所有点随距离的分布
            ax.scatter(r, values, alpha=0.5, s=20)
            
            # 计算平均值
            bins = np.linspace(0, 15, 20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            radial_avg = []
            for i in range(len(bins)-1):
                mask = (r >= bins[i]) & (r < bins[i+1])
                if np.any(mask):
                    radial_avg.append(np.mean(values[mask]))
                else:
                    radial_avg.append(0)
            
            # 绘制平均趋势线
            ax.plot(bin_centers, radial_avg, 'r-', linewidth=3, 
                   label='径向平均值')
            
            # 添加指数衰减参考线
            decay_ref = radial_avg[0] * np.exp(-bin_centers/5)
            ax.plot(bin_centers, decay_ref, 'b--', linewidth=2.5,
                   label='指数衰减参考 (λ=5mm)')
            
            ax.set_title("束流强度径向分布")
            ax.set_xlabel("到中心点的距离 (mm)")
            ax.set_ylabel("刻蚀速率 (nm/s)")
            ax.grid(True)
            ax.legend()
            
            fig.savefig("beam_radial_distribution.png", dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.log("束流径向分布验证图已保存")
        except Exception as e:
            self.log(f"径向分布图保存失败: {str(e)}")
    
    def plot_convexity_verification(self, beam_full):
        """绘制等高线凸性验证图"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle("等高线凸性验证", fontsize=16)
            
            # 归一化束流以生成等高线
            min_val = np.min(beam_full[beam_full > 0])
            max_val = np.max(beam_full)
            normalized = (beam_full - min_val) / (max_val - min_val)
            
            # 在ax1上绘制优化后的束流等高线
            ax1.set_title("优化后束流分布")
            contour_levels = np.linspace(0.1, 0.9, 5)
            contour_set = ax1.contourf(self.grid, self.grid, normalized, 
                                      levels=contour_levels, cmap="viridis")
            contour_lines = ax1.contour(self.grid, self.grid, normalized, 
                                      levels=contour_levels, colors='k', linewidths=1)
            plt.colorbar(contour_set, ax=ax1, label="归一化束流强度")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            ax1.set_aspect('equal', 'box')
            
            # 在ax2上绘制单独的等高线用于凸性分析
            ax2.set_title("等高线凸包验证")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_aspect('equal', 'box')
            
            level_colors = ['blue', 'green', 'orange', 'purple', 'red']

            # 添加最外圈零值区标记
            outer_x, outer_y = np.where(self.outer_ring_mask)
            outer_pos_x = self.grid[outer_x]
            outer_pos_y = self.grid[outer_y]
            ax1.scatter(outer_pos_x, outer_pos_y, 
                    color='gray', s=20, alpha=0.5,
                    label="零值区(|x|>=14或|y|>=14)")
            ax2.scatter(outer_pos_x, outer_pos_y, 
                    color='gray', s=20, alpha=0.5,
                    label="零区(|x|>=14或|y|>=14)")
            
            for i, level in enumerate(contour_levels):
                # 创建当前层级的二值图像
                _, binary_img = cv2.threshold((normalized * 255).astype(np.uint8), 
                                             level * 255, 255, cv2.THRESH_BINARY)
                
                # 查找等高线
                contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # 简化等高线
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 计算凸包
                    hull = cv2.convexHull(approx)
                    
                    # 绘制原始等高线
                    approx_pts = approx.reshape(-1, 2)
                    ax2.plot(approx_pts[:, 0] / 2 - 15, approx_pts[:, 1] / 2 - 15, 
                             color=level_colors[i], linewidth=2, 
                             label=f'{level:.1f}' if hull.shape[0] == approx.shape[0] else '')
                    
                    # 绘制凸包
                    hull_pts = hull.reshape(-1, 2)
                    ax2.plot(hull_pts[:, 0] / 2 - 15, hull_pts[:, 1] / 2 - 15, 
                             '--', color=level_colors[i], linewidth=1.5)
            
            # 添加图例
            ax2.legend(title='等高线层级', loc='upper right')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig("convexity_verification.png", bbox_inches='tight')
            self.log("等高线凸性验证图已保存")
            plt.close(fig)
        except Exception as e:
            self.log(f"等高线凸性验证图创建失败: {str(e)}")
    
    def visualize_results(self):
        """可视化优化结果"""
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle("离子束刻蚀效率优化结果 (凸性约束版)", fontsize=16)
            
            # 1. 原始束流分布
            ax1 = plt.subplot(231)
            im1 = ax1.imshow(self.initial_beam, cmap="viridis", 
                            extent=[-15, 15, -15, 15])
            
            # 添加最外圈零值区标记
            outer_indices = np.where(self.outer_ring_mask)
            outer_x = self.grid[outer_indices[0]]
            outer_y = self.grid[outer_indices[1]]
            ax1.scatter(outer_x, outer_y, s=30, alpha=0.6, 
                    color='gray', marker='s', label='零值区')
            
            # 标记位置点集阶段
            for i, mask in enumerate(self.stage_masks):
                indices = np.where(mask)
                if len(indices[0]) > 0:
                    x = self.grid[indices[0]]
                    y = self.grid[indices[1]]
                    ax1.scatter(x, y, s=40, marker=f'.', 
                               alpha=0.7, label=f'集合{i+1}')
            
            ax1.set_title("初始束流分布")
            plt.colorbar(im1, ax=ax1, label="刻蚀速率 (nm/s)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 2. 优化后束流分布
            optimized_beam_full = self.optimized_beam * self.max_val
            ax2 = plt.subplot(232)
            im2 = ax2.imshow(optimized_beam_full, cmap="viridis", 
                            extent=[-15, 15, -15, 15])
            
            # 在优化后束流分布图 (ax2) 中添加相同的标记：
            ax2.scatter(outer_x, outer_y, s=30, alpha=0.6,      
            color='gray', marker='s', label='零值区')

            # 添加等高线
            min_val = np.min(optimized_beam_full[optimized_beam_full > 0])
            max_val = np.max(optimized_beam_full)
            normalized = (optimized_beam_full - min_val) / (max_val - min_val)
            contour_levels = np.linspace(0.1, 0.9, 10)
            cont = ax2.contour(self.grid, self.grid, normalized,
                              levels=contour_levels, colors='w', linewidths=0.7, alpha=0.7)
            ax2.clabel(cont, fmt='%1.1f', colors='k', fontsize=8)
            ax2.set_title("优化后束流分布")
            plt.colorbar(im2, ax=ax2, label="刻蚀速率 (nm/s)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            
            # 3. 束沿Y移动时的X方向轮廓
            ax3 = plt.subplot(233)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")

            # 实验数据直接使用原始值
            ax3.scatter(
                self.beam_traced_y_axis[:, 0], 
                self.beam_traced_y_axis[:, 1],  # <- 实际值（nm）
                c="g", s=30, alpha=0.6, label="实验数据 (束沿Y)"
            )
            ax3.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            ax3.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            ax3.set_title("束沿Y移动时的X轴截面")
            ax3.set_xlabel("X位置 (mm)")
            ax3.set_ylabel("刻蚀深度 (nm)")  # <- 修改为物理单位标签
            ax3.grid(True)
            ax3.legend()

            # 4. 束沿X移动时的Y方向轮廓
            ax4 = plt.subplot(234)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")

            # 实验数据直接使用原始值
            ax4.scatter(
                self.beam_traced_x_axis[:, 0], 
                self.beam_traced_x_axis[:, 1],  # <- 实际值（nm）
                c="g", s=30, alpha=0.6, label="实验数据 (束沿X)"
            )
            ax4.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            ax4.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            ax4.set_title("束沿X移动时的Y轴截面")
            ax4.set_xlabel("Y位置 (mm)")
            ax4.set_ylabel("刻蚀深度 (nm)")  # <- 修改为物理单位标签
            ax4.grid(True)
            ax4.legend()
            
            # 5. 误差收敛曲线
            if len(self.history["iteration"]) > 1:
                ax5 = plt.subplot(235)
                iterations = self.history["iteration"]
                abs_errors = self.history["abs_error"]
                rel_err_x = self.history["rel_err_x"]
                rel_err_y = self.history["rel_err_y"]
                
                # 左侧Y轴 - 绝对误差
                ax5.plot(iterations, abs_errors, "ko-", label="绝对误差")
                ax5.set_xlabel("迭代次数")
                ax5.set_ylabel("绝对误差", color='k')
                ax5.tick_params(axis='y', labelcolor='k')
                ax5.grid(True)
                
                # 标记阶段转换
                for i, end_idx in enumerate(self.stage_ends):
                    ax5.axvline(end_idx, color=f'C{i}', linestyle='--', alpha=0.7, label=f"阶段{i+1}结束")
                
                # 右侧Y轴 - 相对误差
                ax5b = ax5.twinx()
                ax5b.plot(iterations, rel_err_x, "r--", label="束X动时Y向误差")
                ax5b.plot(iterations, rel_err_y, "g--", label="束Y动时X向误差")
                ax5b.set_ylabel("相对误差 (%)", color='b')
                ax5b.tick_params(axis='y', labelcolor='b')
                
                # 创建组合图例
                lines1, labels1 = ax5.get_legend_handles_labels()
                lines2, labels2 = ax5b.get_legend_handles_labels()
                ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                ax5.set_title("误差收敛曲线")
            
            # 6. 等高线凸性验证图
            if os.path.exists("convexity_verification.png"):
                from matplotlib import image
                ax6 = plt.subplot(236)
                img = image.imread("convexity_verification.png")
                ax6.imshow(img)
                ax6.axis('off')
                ax6.set_title("等高线凸性验证")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
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
        self.log(f"  - beam_radial_distribution.png (径向分布验证)")
        self.log(f"  - unimodal_verification.png (单峰性验证)")
        self.log(f"  - convexity_verification.png (等高线凸性验证)")
        self.log(f"  - beam_optimization_log.txt (详细日志)")
        self.log_file.close()

# ================== 主程序 ==================
def main():
    # 检查文件存在性
    files = {
        "beam_traced_x_axis": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",  # 束沿X移动时测量的Y截面
        "beam_traced_y_axis": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",  # 束沿Y移动时测量的X截面
        "initial_beam": "beamprofile.csv"
    }
    
    print("=" * 80)
    print("离子束刻蚀效率优化 (凸性约束版)".center(80))
    print(f"输入文件:")
    print(f" - 束沿X移动时的Y截面: {files['beam_traced_x_axis']}")
    print(f" - 束沿Y移动时的X截面: {files['beam_traced_y_axis']}")
    print(f" - 初始束流分布: {files['initial_beam']}")
    print("=" * 80)
    
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
            beam_traced_x_axis=files["beam_traced_x_axis"],
            beam_traced_y_axis=files["beam_traced_y_axis"],
            initial_guess_path=files["initial_beam"]
        )
        
        # 执行11阶段优化
        result, err_x, err_y = optimizer.run_optimization()
        
        # 可视化
        optimizer.visualize_results()
        
        # 最终报告
        optimizer.finalize()
        
        print("\n" + "=" * 80)
        print("优化完成! 最终误差:".center(80))
        print(f" - 束沿X移动时Y向误差: {err_x:.1f}%")
        print(f" - 束沿Y移动时X向误差: {err_y:.1f}%")
        print(f"结果文件:")
        print(f"  - optimized_beam_distribution.csv (优化后束流分布)")
        print(f"  - beam_optimization_results.png (可视化结果)")
        print(f"  - beam_radial_distribution.png (径向分布验证)")
        print(f"  - unimodal_verification.png (单峰性验证)")
        print(f"  - convexity_verification.png (等高线凸性验证)")
        print(f"  - beam_optimization_log.txt (详细日志)")
        print("=" * 80)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        if 'optimizer' in locals() and hasattr(optimizer, 'log_file'):
            optimizer.log_file.close()

if __name__ == "__main__":
    main()
