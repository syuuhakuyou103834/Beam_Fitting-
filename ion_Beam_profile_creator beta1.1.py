import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from scipy.integrate import trapezoid
import warnings
import sys

warnings.filterwarnings('ignore', category=RuntimeWarning)  # 抑制优化警告

# ================= 优化的中文字体支持配置 =================
def setup_chinese_fonts():
    """稳健的中文字体检测方法"""
    try:
        # 尝试几种常见字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
        available_fonts = []
        for font in chinese_fonts:
            try:
                fm.findfont(font)
                available_fonts.append(font)
            except:
                pass
        
        if available_fonts:
            plt.rcParams['font.sans-serif'] = available_fonts
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用字体: {available_fonts[0]}")
            return True
        
        # 如果全部失败，尝试获取系统字体
        all_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'SC' in f.name]
        if all_fonts:
            plt.rcParams['font.sans-serif'] = all_fonts[:1]  # 只选一个
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用系统字体: {all_fonts[0]}")
            return True
    except:
        pass
    
    print("警告：无法加载中文字体，图表将使用英文")
    return False

# 调用字体设置函数
font_available = setup_chinese_fonts()

# ================= 增强的束流重构系统 =================
class IonBeamReconstructor:
    def __init__(self, x_profile_path, y_profile_path):
        """初始化束流重构系统"""
        self.x_profile_path = x_profile_path
        self.y_profile_path = y_profile_path
        self.load_and_preprocess_data()
        self.create_high_resolution_grid()
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置优化参数"""
        self.optimal_params = None
        self.beam_model = None
        self.recon_error = 0.0
        self.rotation_angle = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.iteration_count = 0
        self.angle_history = []  # 存储角度优化历史
        self.error_history = []  # 存储误差历史
        self.optimization_stages = []  # 存储优化阶段信息
        self.iteration_count_per_stage = []  # 每个阶段的迭代次数
    
    def load_and_preprocess_data(self):
        """加载并预处理实验数据，保持原始分辨率"""
        # 加载数据
        print(f"读取实验数据: [X扫描: {os.path.basename(self.x_profile_path)}, Y扫描: {os.path.basename(self.y_profile_path)}]")
        x_data = np.loadtxt(self.x_profile_path, delimiter=',', skiprows=1)
        y_data = np.loadtxt(self.y_profile_path, delimiter=',', skiprows=1)
        
        # 保留原始分辨率
        self.x_pos = x_data[:, 0]
        self.y_pos = y_data[:, 0]
        
        # 归一化处理 (积分为1)
        self.x_data = x_data[:, 1] / trapezoid(x_data[:, 1], self.x_pos)
        self.y_data = y_data[:, 1] / trapezoid(y_data[:, 1], self.y_pos)
        
        # 打印数据摘要
        print(f"X扫描数据点: {len(self.x_pos)}, 范围: {np.min(self.x_pos):.2f} 到 {np.max(self.x_pos):.2f} mm")
        print(f"Y扫描数据点: {len(self.y_pos)}, 范围: {np.min(self.y_pos):.2f} 到 {np.max(self.y_pos):.2f} mm")
    
    def create_high_resolution_grid(self):
        """创建高分辨率计算网格"""
        # 确定网格步长 (使用原始数据的最小步长)
        dx = np.min(np.diff(np.unique(self.x_pos)))
        dy = np.min(np.diff(np.unique(self.y_pos)))
        if dx <= 0 or np.isnan(dx) or dy <= 0 or np.isnan(dy):
            # 使用默认步长
            dx = 0.25
            dy = 0.25
            print(f"使用默认步长: {dx}×{dy} mm")
        else:
            print(f"使用数据步长: {dx:.4f}×{dy:.4f} mm")
            
        # 计算网格范围
        x_min, x_max = np.min(self.x_pos), np.max(self.x_pos)
        y_min, y_max = np.min(self.y_pos), np.max(self.y_pos)
        padding = max(x_max-x_min, y_max-y_min) * 0.15  # 15%扩展
        
        # 创建网格
        grid_x = np.arange(x_min - padding, x_max + padding + dx, dx)
        grid_y = np.arange(y_min - padding, y_max + padding + dy, dy)
        self.xx, self.yy = np.meshgrid(grid_x, grid_y)
        
        # 打印网格信息
        print(f"创建高分辨率网格: {len(grid_x)}×{len(grid_y)}点")
    
    def double_gaussian_model(self, params):
        """双高斯束流模型"""
        x0, y0 = params[0], params[1]
        sig_x1, sig_y1, amp1 = params[2], params[3], params[4]
        sig_x2, sig_y2, amp2 = params[5], params[6], params[7]
        theta = np.deg2rad(params[8])
        
        # 更新参数
        self.x_offset, self.y_offset = x0, y0
        self.rotation_angle = params[8]
        
        # 创建偏移坐标并旋转
        xx_offset = self.xx - x0
        yy_offset = self.yy - y0
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated = np.dot(rotation_matrix, np.stack([xx_offset.ravel(), yy_offset.ravel()]))
        x_rot = rotated[0].reshape(xx_offset.shape)
        y_rot = rotated[1].reshape(yy_offset.shape)
        
        # 计算高斯分量
        g1 = amp1 * np.exp(-x_rot**2/(2*sig_x1**2) - y_rot**2/(2*sig_y1**2))
        g2 = amp2 * np.exp(-x_rot**2/(2*sig_x2**2) - y_rot**2/(2*sig_y2**2))
        
        return g1 + g2
    
    def model_mismatch(self, params):
        """计算模型误差"""
        # 生成模型预测
        beam_profile = self.double_gaussian_model(params)
        
        # 计算预测剖面
        pred_x = np.zeros_like(self.x_pos)
        pred_y = np.zeros_like(self.y_pos)
        
        # 找到最接近实际位置的网格列
        for i, x_val in enumerate(self.x_pos):
            col_idx = np.argmin(np.abs(self.xx[0, :] - x_val))
            pred_x[i] = trapezoid(beam_profile[:, col_idx], self.yy[:, col_idx])
        
        # 找到最接近实际位置的网格行
        for j, y_val in enumerate(self.y_pos):
            row_idx = np.argmin(np.abs(self.yy[:, 0] - y_val))
            pred_y[j] = trapezoid(beam_profile[row_idx, :], self.xx[row_idx, :])
        
        # 计算误差
        error_x = np.sqrt(np.mean((pred_x - self.x_data)**2))
        error_y = np.sqrt(np.mean((pred_y - self.y_data)**2))
        
        # 计算总误差
        total_error = error_x + error_y
        
        # 记录当前迭代信息
        self.iteration_count += 1
        current_angle = params[8]
        
        # 只有在第一次或角度变化时才记录
        if (len(self.angle_history) == 0 or 
            current_angle != self.angle_history[-1] or 
            total_error < min(self.error_history) * 0.95):
            
            self.angle_history.append(current_angle)
            self.error_history.append(total_error)
            
            if self.iteration_count <= 20 or self.iteration_count % 10 == 0 or total_error < min(self.error_history):
                print(f"Iter {self.iteration_count:03d}: θ={current_angle:7.2f}°, Error={total_error:.6f}")
        
        return total_error
    
    def reconstruct_beam(self):
        """执行束流重构 - 增强版本"""
        print("\n===== 开始束流重构 =====")
        print("使用两阶段优化策略...")
        
        # 重置迭代计数器
        self.iteration_count = 0
        self.angle_history = []
        self.error_history = []
        
        # ===== 第一阶段：全局粗略优化 =====
        print("\n阶段 1: 全局粗略优化")
        self.optimization_stages.append("全局优化 (差分进化)")
        
        # 设置参数边界
        bounds = [
            (-5, 5), (-5, 5),         # x0, y0
            (0.1, 15), (0.1, 15), (0.01, 2),  # sig_x1, sig_y1, amp1
            (0.1, 10), (0.1, 10), (0.01, 1.5),  # sig_x2, sig_y2, amp2
            (-60, 60)                  # theta
        ]
        
        # 创建初始参数猜测
        initial_guess = [
            0.0, 0.0,  # x0, y0
            np.ptp(self.x_pos)/4, np.ptp(self.y_pos)/4, 1.0,  # 第一个高斯
            np.ptp(self.x_pos)/8, np.ptp(self.y_pos)/8, 0.5,  # 第二个高斯
            0.0  # theta
        ]
        
        # 尝试全局优化
        print("使用差分进化算法进行全局优化...")
        try:
            global_result = differential_evolution(
                lambda params: self.model_mismatch(params),
                bounds=bounds,
                strategy='best1bin',
                maxiter=50,
                popsize=12,
                tol=0.01,
                seed=42,
                disp=True,
                workers=1  # Windows兼容
            )
            initial_guess = global_result.x
            print(f"全局优化完成，初始误差: {global_result.fun:.6f}")
            self.iteration_count_per_stage.append(len(self.error_history))
            self.optimization_stages.append(f"全局优化结束 θ={initial_guess[8]:.1f}°")
        except Exception as e:
            print(f"全局优化失败: {str(e)}, 使用初始参数")
            self.iteration_count_per_stage.append(0)
            self.optimization_stages.append("全局优化失败，使用初始参数")
        
        # ===== 第二阶段：局部精确优化 =====
        print("\n阶段 2: 局部精确优化")
        self.optimization_stages.append("局部优化 (SLSQP)")
        
        # 使用更紧密的边界进行局部优化
        if len(self.angle_history) > 0 and len(bounds) == 9:
            last_angle = initial_guess[8]
            angle_bounds = (max(-60, last_angle-10), min(60, last_angle+10))
        else:
            angle_bounds = (-45, 45)
        
        phase2_bounds = [
            (initial_guess[0]-1, initial_guess[0]+1), 
            (initial_guess[1]-1, initial_guess[1]+1),
            (max(0.05, initial_guess[2]*0.5), min(20, initial_guess[2]*1.5)),
            (max(0.05, initial_guess[3]*0.5), min(20, initial_guess[3]*1.5)),
            (max(0.01, initial_guess[4]*0.5), min(2.5, initial_guess[4]*1.5)),
            (max(0.05, initial_guess[5]*0.5), min(15, initial_guess[5]*1.5)),
            (max(0.05, initial_guess[6]*0.5), min(15, initial_guess[6]*1.5)),
            (max(0.01, initial_guess[7]*0.5), min(2.0, initial_guess[7]*1.5)),
            angle_bounds
        ]
        
        # 执行局部优化
        start_errors = len(self.error_history)
        result = minimize(
            lambda params: self.model_mismatch(params),
            initial_guess,
            method='SLSQP',
            bounds=phase2_bounds,
            options={
                'maxiter': 100,
                'ftol': 1e-6,
                'disp': True
            }
        )
        
        # 保存优化结果
        phase2_iters = len(self.error_history) - start_errors
        self.iteration_count_per_stage.append(phase2_iters)
        self.optimal_params = result.x
        self.beam_model = self.double_gaussian_model(result.x)
        self.optimization_stages.append(f"优化完成 θ={self.rotation_angle:.1f}°")
        
        # 计算最终误差
        self.calculate_final_error()
        
        # 打印优化总结
        print("\n" + "="*60)
        print("重构完成! 旋转角度优化路径总结:")
        for i, stage in enumerate(self.optimization_stages):
            iter_count = self.iteration_count_per_stage[i] if i < len(self.iteration_count_per_stage) else 0
            print(f" - 阶段 {i+1}: {stage} ({iter_count}次迭代)")
        
        if len(self.angle_history) > 0:
            print(f"\n旋转角度从初始值优化到: {self.angle_history[0]:.2f}° → {self.angle_history[-1]:.2f}°")
            print(f"误差从 {self.error_history[0]:.4f} 减少到 {self.error_history[-1]:.4f}")
        else:
            print("\n没有优化的角度历史数据")
        
        print(f"综合拟合误差: {self.recon_error*100:.4f}%")
        print("="*60)
        
        return self.beam_model
    
    def calculate_final_error(self):
        """计算最终误差"""
        # 计算预测剖面
        pred_x = np.zeros_like(self.x_pos)
        pred_y = np.zeros_like(self.y_pos)
        
        for i, x_val in enumerate(self.x_pos):
            col_idx = np.argmin(np.abs(self.xx[0, :] - x_val))
            pred_x[i] = trapezoid(self.beam_model[:, col_idx], self.yy[:, col_idx])
        
        for j, y_val in enumerate(self.y_pos):
            row_idx = np.argmin(np.abs(self.yy[:, 0] - y_val))
            pred_y[j] = trapezoid(self.beam_model[row_idx, :], self.xx[row_idx, :])
        
        # 计算归一化RMS误差
        x_range = np.max(self.x_data) - np.min(self.x_data) or 1.0
        y_range = np.max(self.y_data) - np.min(self.y_data) or 1.0
        
        err_x = np.sqrt(np.mean((pred_x - self.x_data)**2)) / x_range
        err_y = np.sqrt(np.mean((pred_y - self.y_data)**2)) / y_range
        
        self.recon_error = 0.5 * (err_x + err_y)
    
    def export_beam_csv(self, output_path):
        """输出高分辨率束流分布为CSV文件"""
        if self.beam_model is None:
            raise ValueError("未执行束流重构操作")
        
        # 归一化束流分布
        beam_normalized = self.beam_model / np.max(self.beam_model)
        
        # 获取网格坐标
        x_coords = self.xx[0, :]
        y_coords = self.yy[:, 0]
        
        # 准备标题
        header = f"离子束能量分布 (优化角度: {self.rotation_angle:.4f}°)\n"
        header += f"中心偏移: X={self.x_offset:.4f} mm, Y={self.y_offset:.4f} mm\n"
        header += f"重构误差: {self.recon_error*100:.4f}%\n"
        header += f"X坐标范围: {np.min(x_coords):.2f} 到 {np.max(x_coords):.2f} mm\n"
        header += f"Y坐标范围: {np.min(y_coords):.2f} 到 {np.max(y_coords):.2f} mm\n"
        header += "\n优化路径总结:\n"
        for stage in self.optimization_stages:
            header += f"# {stage}\n"
        
        # 保存数据
        np.savetxt(output_path, beam_normalized, delimiter=",", header=header, comments='# ')
        print(f"高分辨率束流分布已保存至: {output_path}")
        
        # 创建优化路径的CSV
        if len(self.angle_history) > 0:
            path_path = output_path.replace(".csv", "_optimization_path.csv")
            path_data = np.column_stack([
                np.arange(len(self.angle_history)),
                self.angle_history,
                self.error_history
            ])
            path_header = "迭代步,旋转角度(度),总误差"
            np.savetxt(path_path, path_data, delimiter=",", header=path_header, fmt=['%d', '%.6f', '%.6f'])
            print(f"旋转角度优化路径已保存至: {path_path}")
        
        return output_path
    
    def generate_plots(self, output_prefix="beam_reconstruction"):
        """生成增强版验证图表，包含详细的优化轨迹"""
        if self.beam_model is None:
            raise ValueError("未执行束流重构操作")
        
        # 计算预测剖面
        pred_x = np.zeros_like(self.x_pos)
        pred_y = np.zeros_like(self.y_pos)
        
        for i, x_val in enumerate(self.x_pos):
            col_idx = np.argmin(np.abs(self.xx[0, :] - x_val))
            pred_x[i] = trapezoid(self.beam_model[:, col_idx], self.yy[:, col_idx])
        
        for j, y_val in enumerate(self.y_pos):
            row_idx = np.argmin(np.abs(self.yy[:, 0] - y_val))
            pred_y[j] = trapezoid(self.beam_model[row_idx, :], self.xx[row_idx, :])
        
        # 计算误差
        x_error = np.abs(pred_x - self.x_data) / np.max(self.x_data) * 100
        y_error = np.abs(pred_y - self.y_data) / np.max(self.y_data) * 100
        
        # 创建优化的增强可视化
        fig = plt.figure(figsize=(18, 22))
        if not font_available:
            plt.rcParams['font.sans-serif'] = ['Arial']
        
        # 创建网格布局
        gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 1.5])
        
        # 1. 束流热力图
        ax1 = fig.add_subplot(gs[0, :])
        im = ax1.imshow(
            self.beam_model,
            extent=[np.min(self.xx), np.max(self.xx), np.min(self.yy), np.max(self.yy)],
            cmap='viridis',
            origin='lower',
            aspect='auto'
        )
        # 标记中心和坐标系旋转
        ax1.plot(self.x_offset, self.y_offset, 'rx', markersize=12, markeredgewidth=3)
        
        # 绘制旋转后的坐标系
        rotation_angle = self.rotation_angle
        length = min(5, (np.max(self.yy) - np.min(self.yy)) / 4)
        x_end = self.x_offset + length * np.cos(np.deg2rad(rotation_angle))
        y_end = self.y_offset + length * np.sin(np.deg2rad(rotation_angle))
        ax1.plot([self.x_offset, x_end], [self.y_offset, y_end], 'w--', linewidth=1.5, alpha=0.8)
        ax1.plot([self.x_offset, self.x_offset+length/2], [self.y_offset, self.y_offset], 'c--', linewidth=1.5, alpha=0.8)
        ax1.plot([self.x_offset, self.x_offset], [self.y_offset, self.y_offset+length/2], 'm--', linewidth=1.5, alpha=0.8)
        
        ax1.set_title(f"束流能量分布 (旋转角: {rotation_angle:.2f}°)", fontsize=14)
        ax1.set_xlabel("X位置 (mm)", fontsize=12)
        ax1.set_ylabel("Y位置 (mm)", fontsize=12)
        plt.colorbar(im, ax=ax1, label="相对强度")
        
        # 2. X方向剖面
        ax2 = fig.add_subplot(gs[1, 0])
        # 光滑处理
        fine_points = 300
        x_fine = np.linspace(min(self.x_pos), max(self.x_pos), fine_points)
        
        # 使用插值平滑曲线
        pred_x_smooth = np.interp(x_fine, self.x_pos, pred_x)
        x_data_smooth = np.interp(x_fine, self.x_pos, self.x_data)
        
        if font_available:
            exp_label = "实验数据"
            model_label = "模型预测"
        else:
            exp_label = "Experimental"
            model_label = "Model Prediction"
        
        ax2.plot(self.x_pos, self.x_data, 'bo', markersize=3, label=exp_label)
        ax2.plot(x_fine, pred_x_smooth, 'r-', linewidth=2, label=model_label)
        ax2.fill_between(x_fine, 
                         x_data_smooth - np.abs(pred_x_smooth - x_data_smooth),
                         x_data_smooth + np.abs(pred_x_smooth - x_data_smooth),
                         color='gray', alpha=0.2)
        ax2.set_title("X方向剖面拟合", fontsize=12)
        ax2.set_xlabel("X位置 (mm)", fontsize=10)
        ax2.set_ylabel("归一化强度", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        # 添加误差文本
        max_x_err = np.max(x_error)
        avg_x_err = np.mean(x_error)
        if font_available:
            err_text = f"最大误差: {max_x_err:.2f}%\n平均误差: {avg_x_err:.2f}%"
        else:
            err_text = f"Max error: {max_x_err:.2f}%\nAvg error: {avg_x_err:.2f}%"
        
        ax2.text(0.02, 0.95, err_text, 
                transform=ax2.transAxes, ha='left', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # 3. Y方向剖面
        ax3 = fig.add_subplot(gs[1, 1])
        y_fine = np.linspace(min(self.y_pos), max(self.y_pos), fine_points)
        pred_y_smooth = np.interp(y_fine, self.y_pos, pred_y)
        y_data_smooth = np.interp(y_fine, self.y_pos, self.y_data)
        
        ax3.plot(self.y_pos, self.y_data, 'go', markersize=3, label=exp_label)
        ax3.plot(y_fine, pred_y_smooth, 'm-', linewidth=2, label=model_label)
        ax3.fill_between(y_fine, 
                         y_data_smooth - np.abs(pred_y_smooth - y_data_smooth),
                         y_data_smooth + np.abs(pred_y_smooth - y_data_smooth),
                         color='gray', alpha=0.2)
        ax3.set_title("Y方向剖面拟合", fontsize=12)
        ax3.set_xlabel("Y位置 (mm)", fontsize=10)
        ax3.set_ylabel("归一化强度", fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(loc='best', fontsize=9)
        
        # 添加误差文本
        max_y_err = np.max(y_error)
        avg_y_err = np.mean(y_error)
        if font_available:
            err_text = f"最大误差: {max_y_err:.2f}%\n平均误差: {avg_y_err:.2f}%"
        else:
            err_text = f"Max error: {max_y_err:.2f}%\nAvg error: {avg_y_err:.2f}%"
        
        ax3.text(0.02, 0.95, err_text, 
                transform=ax3.transAxes, ha='left', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # 4. 旋转角度优化轨迹
        ax4 = fig.add_subplot(gs[2, :])
        
        if len(self.angle_history) >= 2:
            iterations = np.arange(len(self.angle_history))
            
            ax4.plot(iterations, self.angle_history, 'b-o', linewidth=1.5, markersize=4)
            
            if font_available:
                title = "旋转角度优化轨迹"
                xlabel = "迭代步数"
                ylabel = "旋转角度 (度)"
            else:
                title = "Rotation Angle Optimization Path"
                xlabel = "Iteration"
                ylabel = "Rotation Angle (deg)"
                
            ax4.set_title(title, fontsize=14)
            ax4.set_xlabel(xlabel, fontsize=12)
            ax4.set_ylabel(ylabel, fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.6)
            
            # 标记优化阶段（如果存在）
            if self.iteration_count_per_stage and len(self.iteration_count_per_stage) >= 2:
                stages = np.cumsum([0] + self.iteration_count_per_stage[:2])
                colors = ['#FF5733', '#33FF57', '#3377FF', '#F3FF33']
                
                for i, stage_start in enumerate(stages[:-1]):
                    stage_end = stages[i+1]
                    if i > 0:  # 跳过第一个点（起点）
                        ax4.axvline(x=stage_start, color=colors[i%4], linestyle='--', alpha=0.7)
                    
                    if stage_end < len(self.angle_history) and i < len(self.optimization_stages):
                        mid_point = (stage_start + stage_end) / 2
                        ax4.text(mid_point, min(self.angle_history) + 0.3 * np.ptp(self.angle_history), 
                                self.optimization_stages[i], fontsize=9, 
                                ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # 添加初始值和最终值
            ax4.plot(0, self.angle_history[0], 'ro', markersize=8, label=("初始角度" if font_available else "Initial"))
            ax4.plot(len(self.angle_history)-1, self.angle_history[-1], 'gs', markersize=10, label=("最终角度" if font_available else "Final"))
            ax4.legend(loc='best')
            
            # 在轨迹上标记关键点
            ax4_text = f"{self.angle_history[0]:.1f}°" if font_available else f"{self.angle_history[0]:.1f} deg"
            ax4.text(0, self.angle_history[0] + 0.1 * np.ptp(self.angle_history), ax4_text, fontsize=9)
            
            ax4_text = f"{self.angle_history[-1]:.1f}°" if font_available else f"{self.angle_history[-1]:.1f} deg"
            ax4.text(len(self.angle_history)-1, self.angle_history[-1] + 0.1 * np.ptp(self.angle_history), ax4_text, fontsize=9)
        else:
            if font_available:
                no_data_text = "无角度优化数据"
            else:
                no_data_text = "No Angle Optimization Data"
            
            ax4.text(0.5, 0.5, no_data_text, ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.axis('off')
        
        # 5. 误差优化轨迹
        ax5 = fig.add_subplot(gs[3, 0])
        if len(self.error_history) >= 2:
            iterations = np.arange(len(self.error_history))
            ax5.plot(iterations, self.error_history, 'r-s', linewidth=1.5, markersize=4)
            
            if font_available:
                title = "模型误差优化轨迹"
                xlabel = "迭代步数"
                ylabel = "总误差 (RMS)"
            else:
                title = "Error Optimization Path"
                xlabel = "Iteration"
                ylabel = "Total Error (RMS)"
                
            ax5.set_title(title, fontsize=14)
            ax5.set_xlabel(xlabel, fontsize=12)
            ax5.set_ylabel(ylabel, fontsize=12)
            ax5.grid(True, linestyle='--', alpha=0.6)
            # ax5.set_yscale('log')  # 如果误差变化范围大，可以使用对数坐标
            
            # 添加初始值和最终值
            init_label = "初始误差" if font_available else "Initial Error"
            final_label = "最终误差" if font_available else "Final Error"
            ax5.plot(0, self.error_history[0], 'ro', markersize=8, label=init_label)
            ax5.plot(len(self.error_history)-1, self.error_history[-1], 'gs', markersize=10, label=final_label)
            ax5.legend(loc='best')
        else:
            if font_available:
                no_data_text = "无误差优化数据"
            else:
                no_data_text = "No Error Optimization Data"
            
            ax5.text(0.5, 0.5, no_data_text, ha='center', va='center', transform=ax5.transAxes, fontsize=14)
            ax5.axis('off')
        
        # 6. 参数摘要
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.axis('off')
        
        if self.optimal_params is not None and len(self.optimal_params) >= 9:
            if font_available:
                title = "重构参数摘要"
                angle_text = f"旋转角度: {self.rotation_angle:.2f}°"
                center_text = f"中心偏移: X={self.x_offset:.4f} mm, Y={self.y_offset:.4f} mm"
                error_text = f"综合误差: {self.recon_error*100:.4f}%"
                comp1_title = "高斯分量1"
                comp1_text = f"• σX = {self.optimal_params[2]:.4f} mm\n• σY = {self.optimal_params[3]:.4f} mm\n• 幅度 = {self.optimal_params[4]:.4f}"
                comp2_title = "高斯分量2"
                comp2_text = f"• σX = {self.optimal_params[5]:.4f} mm\n• σY = {self.optimal_params[6]:.4f} mm\n• 幅度 = {self.optimal_params[7]:.4f}"
            else:
                title = "Reconstruction Summary"
                angle_text = f"Rotation Angle: {self.rotation_angle:.2f}°"
                center_text = f"Center Offset: X={self.x_offset:.4f} mm, Y={self.y_offset:.4f} mm"
                error_text = f"Total Error: {self.recon_error*100:.4f}%"
                comp1_title = "Gaussian Component 1"
                comp1_text = f"• σX = {self.optimal_params[2]:.4f} mm\n• σY = {self.optimal_params[3]:.4f} mm\n• Amplitude = {self.optimal_params[4]:.4f}"
                comp2_title = "Gaussian Component 2"
                comp2_text = f"• σX = {self.optimal_params[5]:.4f} mm\n• σY = {self.optimal_params[6]:.4f} mm\n• Amplitude = {self.optimal_params[7]:.4f}"
            
            param_text = (
                f"{title}\n\n"
                f"{angle_text}\n"
                f"{center_text}\n"
                f"{error_text}\n\n"
                f"{comp1_title}\n{comp1_text}\n\n"
                f"{comp2_title}\n{comp2_text}"
            )
        elif self.optimal_params is not None:
            param_text = "参数不完整" if font_available else "Incomplete parameters"
        else:
            param_text = "无重构参数" if font_available else "No reconstruction parameters"
        
        ax6.text(0.05, 0.95, param_text, 
                fontsize=11, 
                verticalalignment='top', 
                bbox=dict(facecolor='#f7f7f7', alpha=0.9, boxstyle='round,pad=1'))
        
        # 添加全局标题
        title_text = (f"离子束重建详细分析 (优化角度: {self.rotation_angle:.2f}°)\n重构误差: {self.recon_error*100:.4f}%" 
                     if font_available else
                     f"Ion Beam Reconstruction Analysis (Rotation: {self.rotation_angle:.2f}°)\nError: {self.recon_error*100:.4f}%")
        
        fig.suptitle(title_text, 
                   fontsize=18, fontweight='bold', y=0.99)
        
        # 调整布局
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.92, hspace=0.3)
        
        # 保存图表
        output_path = f"{output_prefix}_analysis.png"
        plt.savefig(output_path, dpi=180, bbox_inches='tight')
        print(f"增强版质量分析图表已保存至: {output_path}")
        plt.close(fig)
        
        return output_path

# ======================= 主程序入口 ========================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 20 + ("高级离子束能量分布重建系统" if font_available else "Advanced Ion Beam Reconstruction System"))
    print("="*70)
    
    # 初始化重建器
    reconstructor = IonBeamReconstructor(
        "x_crosssection trimmed amount profile of Movement on Y-axis.csv",
        "y_crosssection trimmed amount profile of Movement on X-axis.csv"
    )
    
    # 执行束流重建
    try:
        beam_profile = reconstructor.reconstruct_beam()
        
        # 输出结果
        csv_path = reconstructor.export_beam_csv("reconstructed_beam_highres.csv")
        
        # 生成增强版分析图表
        plot_path = reconstructor.generate_plots()
        
        # 打印最终结果
        print("\n" + "="*70)
        print(("重建成功完成!" if font_available else "Reconstruction completed successfully!") + 
             f" 最终错误率: {reconstructor.recon_error*100:.4f}%")
        
        if reconstructor.angle_history:
            angle_change = f"旋转角度: {reconstructor.rotation_angle:.2f}° (从 {reconstructor.angle_history[0]:.2f}° 优化)"
            print(angle_change + f" - 优化路径保存至CSV文件")
        else:
            print(("旋转角度: " if font_available else "Rotation Angle: ") + f"{reconstructor.rotation_angle:.2f}°")
        
        print(("束流数据文件: " if font_available else "Beam data file: ") + csv_path)
        print(("分析图表文件: " if font_available else "Analysis chart: ") + plot_path)
        print("="*70)
    except Exception as e:
        print(("\n!!! 重构过程中发生错误: " if font_available else "\n!!! Error during reconstruction: ") + str(e))
        import traceback
        traceback.print_exc()
        print("="*70)
