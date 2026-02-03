import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid

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
        """初始化优化器，修正了实验数据文件定义：
        - beam_traced_x_axis: 离子束沿X轴移动时测量的Y方向截面
        - beam_traced_y_axis: 离子束沿Y轴移动时测量的X方向截面
        """
        # 创建日志文件（UTF-8编码解决乱码问题）
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
        self.opt_radius = 10.0  # 优化半径 (mm)
        
        # 创建优化掩膜
        self.create_optimization_mask()
        
        # 历史记录
        self.history = {
            "iteration": [],
            "abs_error": [],
            "rel_err_x": [],
            "rel_err_y": [],
            "max_val": []
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
        rows, cols = beam_matrix.shape
        # (原有的峰值中心约束保持不变)
        max_idx = np.argmax(beam_matrix)
        center_idx = np.unravel_index(max_idx, beam_matrix.shape)
        center_i, center_j = center_idx
        center_pos = (self.grid[center_i], self.grid[center_j])
        
        center_distance = np.sqrt(center_pos[0]**2 + center_pos[1]**2)
        if center_distance > 2.0:
            center_region_indices = np.where((np.abs(self.grid) < 1.0) & (np.abs(self.grid) < 1.0))
            if center_region_indices[0].size > 0:
                center_region_values = beam_matrix[center_region_indices]
                max_in_center = np.max(center_region_values)
                max_idx = np.argmax(center_region_values)
                center_i, center_j = center_region_indices[0][0][max_idx], center_region_indices[1][0][max_idx]
                beam_matrix[center_i, center_j] = max(beam_matrix[center_i, center_j], max_in_center)
                center_pos = (self.grid[center_i], self.grid[center_j])
                self.log(f"修正中心偏离: {center_pos}, 值={max_in_center:.4f}")
        
        max_idx = np.argmax(beam_matrix)
        center_i, center_j = np.unravel_index(max_idx, beam_matrix.shape)
        center_pos = (self.grid[center_i], self.grid[center_j])
        max_val = beam_matrix[center_i, center_j]
        self.log(f"当前中心点位置: ({center_pos[0]:.2f}, {center_pos[1]:.2f}), 值={max_val:.4f}")
        
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt((xx - center_pos[0])**2 + (yy - center_pos[1])**2)
        inner_region = (r <= 10.0)
        low_points_mask = beam_matrix < max_val * 0.01
        inner_low_points = np.logical_and(inner_region, low_points_mask)
        low_points_count = np.sum(inner_low_points)
        if low_points_count > 0:
            self.log(f"内区域发现 {low_points_count} 个低值点 (<{max_val * 0.01:.4f})")
            base_value = max_val * 0.02
            beam_matrix[inner_low_points] = base_value
            self.log(f"禁止零值点: 设置 {low_points_count} 个点至少为 {base_value:.4f} (最大值的{base_value/max_val*100:.1f}%)")
        
        # ============== 新增的速率形态约束 ==============
        def calculate_rate_profile(values):
            """计算下降速率曲线并找到拐点"""
            # 计算一阶导数 (下降速率)
            deriv = -np.diff(values)  # 转换为正数表示下降幅度
            # 计算二阶导数 (速率变化率)
            sec_deriv = np.diff(deriv)
            
            # 寻找拐点位置 (二阶导数为零的点)
            inflection_idx = -1
            max_rate_idx = -1
            max_rate = 0
            
            # 寻找下降速率峰值
            for i in range(len(deriv)):
                if deriv[i] > max_rate:
                    max_rate = deriv[i]
                    max_rate_idx = i
            
            # 检查波峰后是否递减
            valid = True
            if len(deriv) > 2:
                # 波峰前应递增 (i < max_rate_index)
                for i in range(1, max_rate_idx):
                    if deriv[i] < deriv[i-1]:
                        valid = False
                        break
                
                # 波峰后应递减 (i > max_rate_index)
                if valid:
                    for i in range(max_rate_idx + 1, len(deriv)):
                        if deriv[i] > deriv[i-1]:
                            valid = False
                            break
            
            return valid, max_rate_idx, deriv
        
        # ============== 沿射线方向的约束 ==============
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        modified_points = 0
        
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            steps = np.arange(0, 10.5, 0.5)
            x_indices = np.round(center_i + dx * (steps / self.grid_spacing)).astype(int)
            y_indices = np.round(center_j + dy * (steps / self.grid_spacing)).astype(int)
            
            valid_mask = (x_indices >= 0) & (x_indices < rows) & \
                        (y_indices >= 0) & (y_indices < cols)
            valid_x = x_indices[valid_mask]
            valid_y = y_indices[valid_mask]
            
            if len(valid_x) < 4:  # 至少需要4个点形成曲线
                continue
            
            # 获取射线方向上的值序列
            ray_values = beam_matrix[valid_x, valid_y]
            
            # 步骤1: 强制单调递减 (现有约束)
            prev_value = ray_values[0]
            for k in range(1, len(valid_x)):
                i, j = valid_x[k], valid_y[k]
                current_val = ray_values[k]
                r_current = np.sqrt((self.grid[i]-center_pos[0])**2 + (self.grid[j]-center_pos[1])**2)
                if r_current > 10.0:
                    continue
                    
                if current_val > prev_value:
                    new_val = min(prev_value * 0.98, current_val)
                    beam_matrix[i, j] = new_val
                    current_val = new_val
                    modified_points += 1
                
                upper_limit = prev_value * 0.95
                if current_val > upper_limit:
                    beam_matrix[i, j] = upper_limit
                    current_val = upper_limit
                    modified_points += 1
                
                prev_value = current_val
            
            # 重新获取更新后的值序列
            ray_values = beam_matrix[valid_x, valid_y]
            
            # 步骤2: 应用新的速率形态约束
            # 检查当前下降速率曲线是否符合要求
            valid_profile, max_rate_idx, deriv = calculate_rate_profile(ray_values)
            
            # 如果不符合速率变化要求
            if not valid_profile or (len(deriv) >= 3 and max_rate_idx < len(deriv) - 1 and deriv[-1] > 0.2 * max(deriv)):
                # 创建理想速率曲线 (单峰，边缘减小)
                target_rates = np.zeros(len(ray_values)-1)
                if max_rate_idx < 0 or max_rate_idx >= len(target_rates):
                    max_rate_idx = len(target_rates) // 2
                    
                # 生成理想单峰速率曲线
                left_len = max_rate_idx + 1
                right_len = len(target_rates) - max_rate_idx
                target_rates[:max_rate_idx] = np.linspace(0.2, 1.0, max_rate_idx)  # 递增
                target_rates[max_rate_idx] = 1.0
                target_rates[max_rate_idx+1:] = np.linspace(0.8, 0.01, right_len-1)  # 递减
                
                # 转换为目标值序列
                target_values = np.zeros_like(ray_values)
                target_values[0] = ray_values[0]
                normalized_deriv = deriv
                if max(deriv) > 0:
                    normalized_deriv = deriv / max(deriv)
                rate_scale = np.mean(abs(np.diff(ray_values))) / np.mean(target_rates) if np.mean(target_rates) > 0 else 1.0
                
                for i in range(1, len(ray_values)):
                    step_size = np.interp(i-1, np.arange(len(target_rates)), target_rates) * rate_scale
                    target_values[i] = target_values[i-1] - step_size
                
                # 应用调整
                for k in range(len(valid_x)):
                    # 保留中心区域不变
                    if k < 3:  # 中心点不变
                        continue
                        
                    i, j = valid_x[k], valid_y[k]
                    beam_matrix[i, j] = target_values[k]
                modified_points += len(valid_x) - 3
                self.log(f"方向 {np.rad2deg(angle):.0f}°: 调整速率曲线形态")
            # 结束新约束
        # 结束角度循环
        
        if modified_points > 0:
            self.log(f"径向约束: 调整了 {modified_points} 个点")
        
        beam_matrix = np.maximum(beam_matrix, 0)
        return beam_matrix


    def simulate_etching(self, beam_matrix, direction):
        """
        模拟指定方向的刻蚀轮廓
        
        修正参数定义:
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
        计算模拟结果与实验结果的误差
        - sim_x_scan: 模拟的束沿Y方向移动时的X方向轮廓
        - sim_y_scan: 模拟的束沿X方向移动时的Y方向轮廓
        """
        # 处理束沿Y移动时的X方向轮廓
        exp_x_data = self.beam_traced_y_axis  # 束沿Y移动时测量的X方向截面
        exp_x_pos = exp_x_data[:, 0]
        exp_x_val = exp_x_data[:, 1]
        
        # 归一化实验数据时保证分母不为零
        exp_x_max = np.max(exp_x_val)
        if exp_x_max <= 0:
            exp_x_norm = np.zeros_like(exp_x_val)
            self.log("警告：X方向实验数据的最大值为零")
        else:
            exp_x_norm = exp_x_val / exp_x_max
        
        # 插值模拟数据到实验点位置
        sim_x_interp = np.interp(exp_x_pos, self.grid, sim_x_scan)
        
        # 计算相对误差时防止除以零
        abs_dev_x = np.abs(sim_x_interp - exp_x_norm)
        rel_err_x = 0.0
        
        # 只对非零值点计算相对误差
        non_zero_x = exp_x_norm > 1e-5  # 设置一个小的阈值防止除以零
        if np.sum(non_zero_x) > 0:
            rel_err_x = np.mean(abs_dev_x[non_zero_x] / exp_x_norm[non_zero_x]) * 100
        
        # 处理束沿X移动时的Y方向轮廓
        exp_y_data = self.beam_traced_x_axis  # 束沿X移动时测量的Y方向截面
        exp_y_pos = exp_y_data[:, 0]
        exp_y_val = exp_y_data[:, 1]
        
        # 归一化实验数据时保证分母不为零
        exp_y_max = np.max(exp_y_val)
        if exp_y_max <= 0:
            exp_y_norm = np.zeros_like(exp_y_val)
            self.log("警告：Y方向实验数据的最大值为零")
        else:
            exp_y_norm = exp_y_val / exp_y_max
        
        # 插值模拟数据到实验点位置
        sim_y_interp = np.interp(exp_y_pos, self.grid, sim_y_scan)
        
        # 计算相对误差时防止除以零
        abs_dev_y = np.abs(sim_y_interp - exp_y_norm)
        rel_err_y = 0.0
        
        # 只对非零值点计算相对误差
        non_zero_y = exp_y_norm > 1e-5  # 设置一个小的阈值防止除以零
        if np.sum(non_zero_y) > 0:
            rel_err_y = np.mean(abs_dev_y[non_zero_y] / exp_y_norm[non_zero_y]) * 100
        
        # 计算绝对误差
        abs_err_x = np.mean(abs_dev_x)
        abs_err_y = np.mean(abs_dev_y)
        abs_error = (abs_err_x + abs_err_y) / 2
        
        return abs_error, rel_err_x, rel_err_y


    def mutate_beam(self, beam_matrix, magnitude):
        """变异束流分布 - 智能变异策略"""
        rows, cols = beam_matrix.shape
        
        # 创建变异副本
        new_beam = beam_matrix.copy()
        
        # 基于优化区域选择变异中心
        valid_indices = np.where(self.optimization_mask)
        
        # 只在优化区域内进行变异
        for i in range(len(valid_indices[0])):
            if np.random.rand() < 0.3:  # 30%的概率变异这个点
                x = valid_indices[0][i]
                y = valid_indices[1][i]
                
                # 变异幅度 (中心大边缘小)
                dist_from_center = self.distance_from_center[x, y]
                dist_factor = np.exp(-dist_from_center / 5.0)  # 根据距离衰减
                mutation = np.random.normal(0, magnitude) * dist_factor
                
                # 应用变异
                new_beam[x, y] = max(0, new_beam[x, y] + mutation)
        
        # 应用物理约束
        return self.enforce_radial_constraints(new_beam)

    def optimize_step(self, current_matrix, abs_error, magnitude):
        """执行单步优化"""
        # 创建候选方案
        candidates = [current_matrix.copy()]  # 保留当前方案
        for _ in range(4):
            candidate = self.mutate_beam(current_matrix, magnitude)
            candidates.append(candidate)
        
        # 评估所有候选方案
        best_candidate = None
        best_abs_error = abs_error
        best_errs = (0, 0)
        errors = []
        
        for candidate in candidates:
            try:
                # 模拟束沿X和Y移动时的轮廓
                sim_x = self.simulate_etching(candidate, "x")  # 束沿Y移动 -> X轮廓
                sim_y = self.simulate_etching(candidate, "y")  # 束沿X移动 -> Y轮廓
                
                # 计算误差
                abs_error, rel_err_x, rel_err_y = self.calculate_error(sim_x, sim_y)
                errors.append(abs_error)
                
                if abs_error < best_abs_error:
                    best_abs_error = abs_error
                    best_errs = (rel_err_x, rel_err_y)
                    best_candidate = candidate
            except Exception as e:
                self.log(f"评估失败: {str(e)}")
                errors.append(1.0)  # 高误差
            
        # 记录评估结果
        errors_str = ", ".join([f"{err:.5f}" for err in errors])
        self.log(f"变异评估: [{errors_str}]")
        
        if best_candidate is None:
            best_candidate = candidates[0]  # 返回原始
            self.log("无改进")
        else:
            improvement = abs_error - best_abs_error
            self.log(f"找到改进 {improvement:.5f}")
        
        return best_candidate, best_abs_error, best_errs

    def run_optimization(self, max_iterations=100, target_rel_error=10.0):
        """运行优化过程 - 稳健的收敛策略"""
        self.log(f"\n开始优化过程 (目标相对误差: {target_rel_error}%)")
        
        current_matrix = self.optimized_beam.copy()
        
        # 初始评估
        sim_x = self.simulate_etching(current_matrix, "x")  # 束沿Y移动 -> X轮廓
        sim_y = self.simulate_etching(current_matrix, "y")  # 束沿X移动 -> Y轮廓
        abs_error0, rel_err_x0, rel_err_y0 = self.calculate_error(sim_x, sim_y)
        self.log(f"初始误差: 绝对误差={abs_error0:.4f} (束X动时Y向误差: {rel_err_x0:.1f}%, 束Y动时X向误差: {rel_err_y0:.1f}%)")
        
        # 添加初始状态到历史记录
        self.history["iteration"].append(0)
        self.history["abs_error"].append(abs_error0)
        self.history["rel_err_x"].append(rel_err_x0)
        self.history["rel_err_y"].append(rel_err_y0)
        
        # 优化循环
        start_time = time.time()
        best_matrix = current_matrix.copy()
        best_abs_error = abs_error0
        best_rel_errs = (rel_err_x0, rel_err_y0)
        stagnation_count = 0
        
        for iteration in range(1, max_iterations + 1):
            # 动态调整变异幅度
            if iteration < 10:
                magnitude = 0.15
            elif iteration < 30:
                magnitude = 0.10
            elif iteration < 50:
                magnitude = 0.07
            else:
                magnitude = max(0.03, 0.10 * (1 - iteration/max_iterations))
            
            # 执行优化迭代
            self.log(f"迭代 {iteration}: 变异幅度={magnitude:.3f}")
            new_matrix, new_abs_error, (new_rel_err_x, new_rel_err_y) = self.optimize_step(
                current_matrix, best_abs_error, magnitude
            )
            
            # 检查改进
            if new_abs_error < best_abs_error:
                improvement = best_abs_error - new_abs_error
                best_abs_error = new_abs_error
                best_matrix = new_matrix.copy()
                best_rel_errs = (new_rel_err_x, new_rel_err_y)
                stagnation_count = 0
                
                self.log(f"改进 Δ={improvement:.5f}, 新绝对误差={best_abs_error:.4f}")
                self.log(f"相对误差: 束X动时Y向 {best_rel_errs[0]:.1f}%, 束Y动时X向 {best_rel_errs[1]:.1f}%")
                
                # 记录历史
                self.history["iteration"].append(iteration)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(new_rel_err_x)
                self.history["rel_err_y"].append(new_rel_err_y)
            else:
                stagnation_count += 1
                self.log(f"无改进，维持误差: {best_abs_error:.4f}")
            
            # 更新当前矩阵
            current_matrix = new_matrix
            
            # 每20次记录一次进度
            if iteration % 20 == 0:
                elapsed = time.time() - start_time
                self.log(f"迭代 {iteration}进度: 束X动时Y向误差: {best_rel_errs[0]:.1f}%, 束Y动时X向误差: {best_rel_errs[1]:.1f}% (用时 {elapsed:.1f}秒)")
            
            # 检查收敛 (满足其中一项即可)
            if new_rel_err_x < target_rel_error and new_rel_err_y < target_rel_error:
                self.log(f"在{iteration}代达成目标误差!")
                break
                
            # 早停检测
            if stagnation_count > 15:
                self.log(f"在{iteration}代检测到连续{stagnation_count}次无改进，结束优化")
                break
            
            # 保存每50代的中间结果
            if iteration % 50 == 0:
                self.log(f"保存迭代 {iteration} 的中间结果...")
                np.savetxt(f"beam_intermediate_{iteration}.csv", current_matrix * self.max_val, delimiter=",")
        
        # 最终处理
        self.optimized_beam = best_matrix
        
        # 最终评估
        sim_x = self.simulate_etching(self.optimized_beam, "x")
        sim_y = self.simulate_etching(self.optimized_beam, "y")
        final_abs_error, final_rel_err_x, final_rel_err_y = self.calculate_error(sim_x, sim_y)
        
        # 保存最终结果
        np.savetxt("optimized_beam_distribution.csv", self.optimized_beam * self.max_val, delimiter=",")
        
        elapsed_time = time.time() - start_time
        self.log(f"\n优化完成! 总迭代次数: {iteration}")
        self.log(f"最终绝对误差: {final_abs_error:.4f} (初始={abs_error0:.4f})")
        self.log(f"最终相对误差: ")
        self.log(f"  束X移动时Y向误差: {final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%)")
        self.log(f"  束Y移动时X向误差: {final_rel_err_y:.1f}% (初始={rel_err_y0:.1f}%)")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        
        return self.optimized_beam, final_rel_err_x, final_rel_err_y

    def visualize_results(self):
        """可视化优化结果"""
        try:
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle("离子束刻蚀效率优化结果", fontsize=16)
            
            # 原始束流分布
            ax1 = plt.subplot(231)
            im1 = ax1.imshow(self.initial_beam, cmap="viridis", extent=[-15, 15, -15, 15])
            ax1.set_title("初始束流分布")
            plt.colorbar(im1, ax=ax1, label="刻蚀速率 (nm/s)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            
            # 优化后束流分布
            ax2 = plt.subplot(232)
            optimized_beam_full = self.optimized_beam * self.max_val
            im2 = ax2.imshow(optimized_beam_full, cmap="viridis", extent=[-15, 15, -15, 15])
            ax2.set_title("优化后束流分布")
            plt.colorbar(im2, ax=ax2, label="刻蚀速率 (nm/s)")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            
            # 差异分布
            ax3 = plt.subplot(233)
            diff = abs(optimized_beam_full - self.initial_beam)
            im3 = ax3.imshow(diff, cmap="hot", extent=[-15, 15, -15, 15])
            ax3.set_title("束流分布变化")
            plt.colorbar(im3, ax=ax3, label="变化量 (nm/s)")
            ax3.set_xlabel("X (mm)")
            ax3.set_ylabel("Y (mm)")
            
            # 束沿Y轴移动时的X方向轮廓对比
            ax4 = plt.subplot(234)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")
            
            # 实验数据来自束沿Y移动时测量的X方向截面
            ax4.scatter(
                self.beam_traced_y_axis[:, 0], 
                self.beam_traced_y_axis[:, 1]/np.max(self.beam_traced_y_axis[:, 1]), 
                c="g", s=30, label="实验数据 (束沿Y)"
            )
            ax4.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            ax4.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            ax4.set_title("束沿Y移动时的X轴截面")
            ax4.set_xlabel("X位置 (mm)")
            ax4.set_ylabel("归一化刻蚀深度")
            ax4.grid(True)
            ax4.legend()
            
            # 束沿X轴移动时的Y方向轮廓对比
            ax5 = plt.subplot(235)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")
            
            # 实验数据来自束沿X移动时测量的Y方向截面
            ax5.scatter(
                self.beam_traced_x_axis[:, 0], 
                self.beam_traced_x_axis[:, 1]/np.max(self.beam_traced_x_axis[:, 1]), 
                c="g", s=30, label="实验数据 (束沿X)"
            )
            ax5.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            ax5.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            ax5.set_title("束沿X移动时的Y轴截面")
            ax5.set_xlabel("Y位置 (mm)")
            ax5.set_ylabel("归一化刻蚀深度")
            ax5.grid(True)
            ax5.legend()
            
            # 误差收敛曲线
            ax6 = plt.subplot(236)
            if len(self.history["iteration"]) > 1:
                # 左侧Y轴 - 绝对误差
                ax6.plot(self.history["iteration"], self.history["abs_error"], "k-", label="绝对误差")
                ax6.set_xlabel("迭代次数")
                ax6.set_ylabel("绝对误差", color='k')
                ax6.tick_params(axis='y', labelcolor='k')
                
                # 右侧Y轴 - 相对误差
                ax6b = ax6.twinx()
                ax6b.plot(self.history["iteration"], self.history["rel_err_x"], "r--", label="束X动时Y向误差")
                ax6b.plot(self.history["iteration"], self.history["rel_err_y"], "g--", label="束Y动时X向误差")
                ax6b.set_ylabel("相对误差 (%)", color='b')
                ax6b.tick_params(axis='y', labelcolor='b')
                
                # 合并图例
                lines, labels = ax6.get_legend_handles_labels()
                lines2, labels2 = ax6b.get_legend_handles_labels()
                ax6.legend(lines + lines2, labels + labels2, loc='best')
                
                ax6.set_title("误差收敛曲线")
                ax6.grid(True)
            
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(top=0.92)
            plt.savefig("beam_optimization_results.png")
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
    print("非高斯离子束束流分布优化".center(70))
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
        
        # 第一阶段优化: 中等目标
        optimizer.log("\n===== 第一阶段优化 (目标: 25%) =====")
        result, err_x, err_y = optimizer.run_optimization(
            max_iterations=100,
            target_rel_error=25.0
        )
        
        # 第二阶段精炼
        optimizer.log("\n===== 第二阶段精炼 (目标: 18%) =====")
        result, err_x, err_y = optimizer.run_optimization(
            max_iterations=80,
            target_rel_error=18.0
        )
        
        # 第三阶段优化: 高精度
        optimizer.log("\n===== 第三阶段优化 (目标: 15%) =====")
        result, err_x, err_y = optimizer.run_optimization(
            max_iterations=50,
            target_rel_error=15.0
        )
        
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