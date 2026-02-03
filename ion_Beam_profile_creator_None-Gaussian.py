import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import scipy
from scipy.interpolate import RegularGridInterpolator

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
    def __init__(self, x_profile_path, y_profile_path, initial_guess_path, grid_bound=15.0, grid_points=31):
        """初始化优化器"""
        # 创建日志文件（UTF-8编码解决乱码问题）
        self.log_file = open("beam_optimization_log.txt", "w", encoding="utf-8")
        self.log("离子束刻蚀效率优化引擎启动")
        self.log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        
        # 加载初始猜测
        self.load_initial_beam(initial_guess_path)
        
        # 加载实验数据
        self.x_exp_data = self.load_experimental_data(x_profile_path)
        self.y_exp_data = self.load_experimental_data(y_profile_path)
        
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
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                
            # 判断分隔符类型
            if len(first_line.split(',')) > 1:
                delimiter = ','
            elif len(first_line.split(';')) > 1:
                delimiter = ';'
            elif len(first_line.split('\t')) > 1:
                delimiter = '\t'
            else:
                delimiter = None
                
            data = np.loadtxt(file_path, delimiter=delimiter)
            
            # 确保数据有位置和值两列
            if data.shape[1] != 2:
                self.log(f"警告: 数据形状为{data.shape}，可能格式不符")
                
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
        """
        强制执行三项径向约束：
        1. 从中心点沿任意方向单调递减
        2. 中心10mm半径内禁止零值点
        3. 极值点必须是中心点或中心点附近
        """
        # 寻找峰值中心
        center_idx = np.array(np.unravel_index(np.argmax(beam_matrix), beam_matrix.shape))
        center_pos = (self.grid[center_idx[0]], self.grid[center_idx[1]])
        
        # 约束1: 确保中心点位于物理中心附近 (|x|<1mm, |y|<1mm)
        center_distance = np.sqrt(center_pos[0]**2 + center_pos[1]**2)
        if center_distance > 2.0:  # 中心偏离超过2mm
            # 找到最近的中心区点作为新中心
            center_region_mask = (np.abs(self.grid) < 1.0) & (np.abs(self.grid) < 1.0)
            center_region_indices = np.where(center_region_mask)
            
            if center_region_indices[0].size > 0:
                # 在中心区找到最大值点
                center_region_values = beam_matrix[center_region_mask]
                max_in_center = np.max(center_region_values)
                
                # 将当前最大点设为找到的居中点
                beam_matrix[center_idx[0], center_idx[1]] = max_in_center
                self.log(f"修正中心偏离: {center_pos} -> (0,0), 值={max_in_center:.4f}")
        
        # 重新确定中心点
        center_idx = np.array(np.unravel_index(np.argmax(beam_matrix), beam_matrix.shape))
        center_pos = (self.grid[center_idx[0]], self.grid[center_idx[1]])
        max_val = beam_matrix[center_idx]
        
        # 约束2: 10mm半径内禁止零值点
        # 创建距离矩阵
        xx, yy = np.meshgrid(self.grid, self.grid, indexing="ij")
        r = np.sqrt((xx - center_pos[0])**2 + (yy - center_pos[1])**2)
        
        # 10mm半径区域
        inner_region = (r <= 10.0)
        zero_points = np.where(beam_matrix[inner_region] < 0.001)
        
        if zero_points[0].size > 0:
            # 找到最小非零值作为基线
            non_zero_values = beam_matrix[inner_region][beam_matrix[inner_region] > 0.001]
            min_val = np.min(non_zero_values) if non_zero_values.size > 0 else 0.01
            
            # 设置所有零值点为基础值（至少为最大值的1%）
            beam_matrix[inner_region] = np.maximum(beam_matrix[inner_region], min_val * 0.8)
            self.log(f"禁止零值点: 设置 {zero_points[0].size} 个点至少为 {min_val*0.8:.4f}")
        
        # 约束3: 沿8个方向辐射递减
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        modified_points = 0
        
        for angle in angles:
            # 生成射线方向
            dx, dy = np.cos(angle), np.sin(angle)
            
            # 生成射线上的点
            steps = np.arange(0, 11, 1)  # 10mm辐射
            x_steps = center_idx[0] + dx * steps / self.grid_spacing
            y_steps = center_idx[1] + dy * steps / self.grid_spacing
            
            # 有效网格点
            valid_points = (x_steps >= 0) & (x_steps < self.grid_points) & \
                           (y_steps >= 0) & (y_steps < self.grid_points)
            x_indices = x_steps[valid_points].astype(int)
            y_indices = y_steps[valid_points].astype(int)
            
            # 射线上的值
            ray_values = beam_matrix[x_indices, y_indices]
            ray_distances = steps[valid_points]
            
            # 强制单调递减
            max_so_far = ray_values[0]
            for i in range(1, len(ray_values)):
                if ray_distances[i] > 10.0:  # 10mm外不强制
                    continue
                    
                # 不允许超过当前最大值
                if ray_values[i] > max_so_far:
                    new_val = min(max_so_far * 0.98, ray_values[i])
                    beam_matrix[x_indices[i], y_indices[i]] = new_val
                    max_so_far = new_val
                    modified_points += 1
                
                # 确保递减 (每次至少减少1%)
                ray_values[i] = min(ray_values[i], max_so_far * 0.99)
                max_so_far = max_so_far * 0.99
            
        if modified_points > 0:
            self.log(f"径向约束: 调整了 {modified_points} 个点")
        
        return beam_matrix

    def simulate_etching(self, beam_matrix, direction):
        """模拟指定方向的刻蚀轮廓"""
        interpolator = self.create_interpolator(beam_matrix)
        
        profile = np.zeros_like(self.grid)
        scan_points = len(self.grid)
        
        if direction == "x":
            # X方向扫描 (沿Y方向移动)
            for i in range(scan_points):
                x_pos = self.grid[i]
                # 沿Y方向的路径点
                path_points = np.array([(x_pos, y) for y in self.grid])
                etch_rates = interpolator(path_points)
                dwell_time = self.grid_spacing / self.scan_velocity
                profile[i] = np.trapz(etch_rates, dx=self.grid_spacing)
        else:  # Y方向
            # Y方向扫描 (沿X方向移动)
            for j in range(scan_points):
                y_pos = self.grid[j]
                # 沿X方向的路径点
                path_points = np.array([(x, y_pos) for x in self.grid])
                etch_rates = interpolator(path_points)
                dwell_time = self.grid_spacing / self.scan_velocity
                profile[j] = np.trapz(etch_rates, dx=self.grid_spacing)
        
        # 归一化
        max_profile = np.max(profile)
        return profile / max_profile if max_profile > 0 else profile

    def calculate_error(self, sim_x, sim_y):
        """计算与实验数据的误差"""
        # 归一化实验数据
        exp_x = self.x_exp_data[:, 1] / np.max(self.x_exp_data[:, 1])
        exp_y = self.y_exp_data[:, 1] / np.max(self.y_exp_data[:, 1])
        
        # 插值到相同位置
        try:
            interp_sim_x = np.interp(self.x_exp_data[:, 0], self.grid, sim_x)
            interp_sim_y = np.interp(self.y_exp_data[:, 0], self.grid, sim_y)
        except Exception as e:
            self.log(f"插值错误: {str(e)}")
            return 1.0, 100.0, 100.0
        
        # 计算绝对误差
        err_x = np.mean(np.abs(interp_sim_x - exp_x))
        err_y = np.mean(np.abs(interp_sim_y - exp_y))
        
        # 计算相对误差（仅在有信号的地方）
        signal_mask_x = exp_x > 0.05 * np.max(exp_x)
        signal_mask_y = exp_y > 0.05 * np.max(exp_y)
        
        rel_err_x = np.mean(np.abs(interp_sim_x[signal_mask_x] - exp_x[signal_mask_x]) / exp_x[signal_mask_x]) * 100
        rel_err_y = np.mean(np.abs(interp_sim_y[signal_mask_y] - exp_y[signal_mask_y]) / exp_y[signal_mask_y]) * 100
        
        abs_error = (err_x + err_y) / 2
        return abs_error, rel_err_x, rel_err_y

    def mutate_beam(self, beam_matrix, magnitude):
        """变异束流分布 - 更智能的变异策略"""
        rows, cols = beam_matrix.shape
        new_beam = beam_matrix.copy()
        
        # 随机选择变异中心 (60%概率在优化区内)
        if np.random.rand() < 0.6:
            # 在优化区内选择中心点
            valid_indices = np.where(self.optimization_mask)
            idx = np.random.randint(0, len(valid_indices[0]))
            cx, cy = valid_indices[0][idx], valid_indices[1][idx]
        else:
            # 随机选择任意点
            cx = np.random.randint(0, rows)
            cy = np.random.randint(0, cols)
        
        # 变异范围 (基于距离中心点的距离)
        xx, yy = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2) / max(rows, cols) * 30
        
        # 变异幅度 (中心大边缘小)
        mutation = np.random.normal(0, magnitude, size=(rows, cols)) * np.exp(-dist)
        
        # 应用变异
        new_beam += mutation
        
        # 确保非负和约束
        new_beam = np.maximum(new_beam, 0)
        new_beam[~self.optimization_mask] = 0  # 外部区域设为0
        
        return new_beam

    def optimize_step(self, beam_matrix, abs_error, magnitude):
        """执行单步优化 - 更高效的变异策略"""
        candidates = []
        
        # 生成4个候选变体
        for _ in range(4):
            candidate = self.mutate_beam(beam_matrix, magnitude)
            candidate = self.enforce_radial_constraints(candidate)  # 应用所有约束
            candidates.append(candidate)
        
        # 最佳候选
        best_candidate = None
        best_abs_error = abs_error
        best_errs = (0, 0)
        improvements = []
        
        for candidate in candidates:
            sim_x = self.simulate_etching(candidate, "x")
            sim_y = self.simulate_etching(candidate, "y")
            abs_error, rel_err_x, rel_err_y = self.calculate_error(sim_x, sim_y)
            
            improvements.append(("abs_error", abs_error))
            
            if abs_error < best_abs_error:
                best_abs_error = abs_error
                best_errs = (rel_err_x, rel_err_y)
                best_candidate = candidate
        
        # 记录改进情况
        improvements_str = ", ".join([f"{val:.5f}" for name, val in improvements])
        self.log(f"变异评估: [{improvements_str}]")
        
        if best_candidate is None:
            # 随机选择一个略有不同的候选进行探索
            best_candidate = candidates[np.random.randint(0, len(candidates))]
            best_abs_error = abs_error  # 保持原始误差
            self.log("无改进，但进行了探索变异")
        
        return best_candidate, best_abs_error, best_errs

    def run_optimization(self, max_iterations=300, target_rel_error=10.0):
        """运行优化过程 - 改进的收敛策略"""
        self.log(f"\n开始优化过程 (目标相对误差: {target_rel_error}%)")
        
        current_matrix = self.optimized_beam.copy()
        
        # 初始评估
        sim_x = self.simulate_etching(current_matrix, "x")
        sim_y = self.simulate_etching(current_matrix, "y")
        abs_error0, rel_err_x0, rel_err_y0 = self.calculate_error(sim_x, sim_y)
        self.log(f"初始误差: 绝对误差={abs_error0:.4f} (X: {rel_err_x0:.1f}%, Y: {rel_err_y0:.1f}%)")
        
        # 优化循环
        start_time = time.time()
        iteration = 0
        abs_error = abs_error0
        best_abs_error = abs_error
        best_matrix = current_matrix.copy()
        best_rel_errs = (rel_err_x0, rel_err_y0)
        improvements = []
        stagnation_count = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 动态调整变异幅度
            if iteration < 10:
                magnitude = 0.2  # 初始大范围探索
            elif iteration < 50:
                magnitude = 0.1
            elif iteration < 100:
                magnitude = 0.05
            else:
                magnitude = max(0.02, 0.05 * (1 - iteration/300))
            
            # 执行优化迭代
            new_matrix, new_abs_error, (new_rel_err_x, new_rel_err_y) = self.optimize_step(
                current_matrix, abs_error, magnitude
            )
            
            # 检查改进
            if new_abs_error < best_abs_error:
                improvement = best_abs_error - new_abs_error
                best_abs_error = new_abs_error
                best_matrix = new_matrix.copy()
                best_rel_errs = (new_rel_err_x, new_rel_err_y)
                stagnation_count = 0
                
                self.log(f"Iter{iteration}: 改进 Δ={improvement:.5f}, 绝对误差={best_abs_error:.4f}, 相对误差(X/Y)={new_rel_err_x:.1f}%/{new_rel_err_y:.1f}%")
                
                # 记录历史
                self.history["iteration"].append(iteration)
                self.history["abs_error"].append(best_abs_error)
                self.history["rel_err_x"].append(new_rel_err_x)
                self.history["rel_err_y"].append(new_rel_err_y)
                
                improvements.append(improvement)
            else:
                stagnation_count += 1
                # 每20次记录一次进度
                if iteration % 20 == 0:
                    elapsed = time.time() - start_time
                    self.log(f"迭代 {iteration}: 当前误差 X={new_rel_err_x:.1f}%/Y={new_rel_err_y:.1f}% (用时 {elapsed:.1f}秒)")
            
            # 更新当前矩阵
            current_matrix = new_matrix
            
            # 检查收敛 (满足其中一项即可)
            if new_rel_err_x < target_rel_error and new_rel_err_y < target_rel_error:
                self.log(f"在{iteration}代达成目标误差!")
                break
                
            # 早停检测 (-1%误差持续5代)
            if len(improvements) > 5 and np.mean(improvements[-5:]) < 0.01:
                self.log(f"在{iteration}代检测到早停条件")
                break
                
            # 无改进过久
            if stagnation_count > 20:
                self.log(f"在{iteration}代检测到停滞，结束优化")
                break
            
            # 保存中间结果
            if iteration % 50 == 0:
                self.log(f"保存迭代 {iteration} 的中间结果...")
                np.savetxt(f"beam_interim_{iteration}.csv", current_matrix * self.max_val, delimiter=",")
        
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
        self.log(f"最终误差: 绝对={final_abs_error:.4f} (初始={abs_error0:.4f}), X={final_rel_err_x:.1f}% (初始={rel_err_x0:.1f}%), Y={final_rel_err_y:.1f}% (初始={rel_err_y0:.1f}%)")
        self.log(f"总耗时: {elapsed_time:.1f}秒")
        
        return self.optimized_beam, final_rel_err_x, final_rel_err_y

    def visualize_results(self):
        """可视化优化结果"""
        try:
            plt.figure(figsize=(15, 10))
            
            # 原始束流分布
            plt.subplot(231)
            plt.imshow(self.initial_beam, cmap="viridis", extent=[-15, 15, -15, 15])
            plt.title("初始束流分布")
            plt.colorbar(label="刻蚀速率 (nm/s)")
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            
            # 优化后束流分布
            plt.subplot(232)
            plt.imshow(self.optimized_beam * self.max_val, cmap="viridis", extent=[-15, 15, -15, 15])
            plt.title("优化后束流分布")
            plt.colorbar(label="刻蚀速率 (nm/s)")
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            
            # X方向对比
            plt.subplot(233)
            sim_x_initial = self.simulate_etching(self.initial_beam / self.max_val, "x")
            sim_x_optim = self.simulate_etching(self.optimized_beam, "x")
            plt.plot(self.grid, sim_x_initial, "b--", label="初始模拟")
            plt.plot(self.grid, sim_x_optim, "r-", label="优化后模拟")
            plt.scatter(self.x_exp_data[:, 0], self.x_exp_data[:, 1]/np.max(self.x_exp_data[:, 1]), 
                       c="g", s=30, label="实验数据")
            plt.title("X方向刻蚀轮廓对比")
            plt.xlabel("位置 (mm)")
            plt.ylabel("归一化刻蚀深度")
            plt.grid(True)
            plt.legend()
            
            # Y方向对比
            plt.subplot(234)
            sim_y_initial = self.simulate_etching(self.initial_beam / self.max_val, "y")
            sim_y_optim = self.simulate_etching(self.optimized_beam, "y")
            plt.plot(self.grid, sim_y_initial, "b--", label="初始模拟")
            plt.plot(self.grid, sim_y_optim, "r-", label="优化后模拟")
            plt.scatter(self.y_exp_data[:, 0], self.y_exp_data[:, 1]/np.max(self.y_exp_data[:, 1]), 
                       c="g", s=30, label="实验数据")
            plt.title("Y方向刻蚀轮廓对比")
            plt.xlabel("位置 (mm)")
            plt.ylabel("归一化刻蚀深度")
            plt.grid(True)
            plt.legend()
            
            # 误差收敛曲线
            plt.subplot(235)
            if len(self.history["iteration"]) > 1:
                plt.plot(self.history["iteration"], self.history["abs_error"], "k-", label="绝对误差")
                plt.legend()
                plt.xlabel("迭代次数")
                plt.ylabel("绝对误差")
                plt.title("误差收敛曲线")
                plt.grid(True)
            
            # 相对误差曲线
            plt.subplot(236)
            if len(self.history["iteration"]) > 1:
                plt.plot(self.history["iteration"], self.history["rel_err_x"], "r-", label="X方向")
                plt.plot(self.history["iteration"], self.history["rel_err_y"], "g-", label="Y方向")
                plt.axhline(y=10, color='b', linestyle='--', label="目标误差")
                plt.legend()
                plt.xlabel("迭代次数")
                plt.ylabel("相对误差 %")
                plt.title("相对误差变化")
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("beam_optimization_results.png")
            self.log("优化结果可视化已保存")
            plt.close()
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
        "x_profile": "x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv",
        "y_profile": "y_crosssection_trimmed_amount_profile_of_Movement_on_X-axis.csv",
        "initial_beam": "beamprofile.csv"
    }
    
    print("非高斯离子束束流分布优化".center(60, "="))
    
    missing = [f for f in files.values() if not os.path.exists(f)]
    if missing:
        print("错误: 以下文件不存在:")
        for f in missing:
            print(f"  - {f}")
        print("请检查文件路径后重试!")
        sys.exit(1)
    
    # 创建优化器
    try:
        optimizer = BeamEfficiencyOptimizer(
            files["x_profile"],
            files["y_profile"],
            files["initial_beam"]
        )
        
        # 第一阶段优化：中等精度
        result, err_x, err_y = optimizer.run_optimization(
            max_iterations=150,
            target_rel_error=20.0  # 目标20%误差
        )
        
        # 第二阶段优化：高精度
        if min(err_x, err_y) > 10.0:
            optimizer.log("\n开始第二阶段优化...")
            result, err_x, err_y = optimizer.run_optimization(
                max_iterations=100,
                target_rel_error=10.0  # 更严格的目标
            )
        
        # 第三阶段优化：微调
        if min(err_x, err_y) > 8.0:
            optimizer.log("\n开始第三阶段微调...")
            result, err_x, err_y = optimizer.run_optimization(
                max_iterations=50,
                target_rel_error=5.0  # 最终目标
            )
        
        # 可视化
        optimizer.visualize_results()
        
        # 最终报告
        optimizer.finalize()
        
        print("\n" + "="*60)
        print(f"优化完成! 最终误差: X方向={err_x:.1f}%, Y方向={err_y:.1f}%")
        print(f"结果文件: optimized_beam_distribution.csv, beam_optimization_results.png")
        print("="*60)
    except Exception as e:
        print(f"优化过程中出错: {str(e)}")
        if 'optimizer' in locals() and hasattr(optimizer, 'log_file'):
            optimizer.log_file.close()

if __name__ == "__main__":
    main()
