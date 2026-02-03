import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from scipy.integrate import trapezoid
from scipy.stats import wasserstein_distance
import warnings
import pickle
from functools import partial

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
        all_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name]
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

# ================= 高稳定性束流重构系统 =================
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
        self.model_type = "double_gaussian"  # 默认模型
        self.angle_history = []
    
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
    
    def create_high_resolution_grid(self):
        """创建高分辨率计算网格"""
        # 确定网格步长 (使用原始数据的最小步长)
        dx = np.min(np.diff(np.unique(self.x_pos)))
        dy = np.min(np.diff(np.unique(self.y_pos)))
        if dx == 0 or dy == 0:
            dx, dy = 0.25, 0.25  # 默认值
        
        # 计算网格范围
        x_min, x_max = np.min(self.x_pos), np.max(self.x_pos)
        y_min, y_max = np.min(self.y_pos), np.max(self.y_pos)
        padding = max(x_max-x_min, y_max-y_min) * 0.15  # 15%扩展
        
        # 创建网格
        grid_x = np.arange(x_min - padding, x_max + padding + dx, dx)
        grid_y = np.arange(y_min - padding, y_max + padding + dy, dy)
        self.xx, self.yy = np.meshgrid(grid_x, grid_y)
        
        print(f"创建高分辨率网格: {len(grid_x)}×{len(grid_y)}点, 步长: {dx:.4f}×{dy:.4f} mm")
    
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
    
    def model_mismatch(self, params, model_type="double"):
        """计算模型误差"""
        # 生成模型预测
        beam_profile = self.double_gaussian_model(params)
        
        # 计算预测剖面
        pred_x = np.zeros_like(self.x_pos)
        pred_y = np.zeros_like(self.y_pos)
        
        for i, x_val in enumerate(self.x_pos):
            col_idx = np.argmin(np.abs(self.xx[0, :] - x_val))
            pred_x[i] = trapezoid(beam_profile[:, col_idx], self.yy[:, col_idx])
        
        for j, y_val in enumerate(self.y_pos):
            row_idx = np.argmin(np.abs(self.yy[:, 0] - y_val))
            pred_y[j] = trapezoid(beam_profile[row_idx, :], self.xx[row_idx, :])
        
        # 计算误差
        error_x = np.sqrt(np.mean((pred_x - self.x_data)**2))
        error_y = np.sqrt(np.mean((pred_y - self.y_data)**2))
        
        # 角度惩罚项
        angle_weight = 1 + 0.2 * min(abs(params[8]), 45) / 45
        
        return angle_weight * (error_x + error_y)
    
    def reconstruct_beam(self):
        """执行束流重构"""
        print("\n===== 开始束流重构 =====")
        
        # 设置参数边界
        bounds = [
            (-5, 5), (-5, 5),         # x0, y0
            (0.2, 20), (0.2, 20), (0.01, 3),  # sig_x1, sig_y1, amp1
            (0.2, 15), (0.2, 15), (0.01, 2),  # sig_x2, sig_y2, amp2
            (-45, 45)                  # theta
        ]
        
        # 创建初始参数猜测
        initial_guess = [
            0.0, 0.0,  # x0, y0
            np.ptp(self.x_data)/2, np.ptp(self.y_data)/2, 1.0,  # 第一个高斯
            np.ptp(self.x_data)/4, np.ptp(self.y_data)/4, 0.5,  # 第二个高斯
            0.0  # theta
        ]
        
        # 第一阶段：全局粗略优化
        print("进行全局粗略优化...")
        try:
            global_result = differential_evolution(
                lambda params: self.model_mismatch(params, "double"),
                bounds=bounds,
                strategy='best1bin',
                maxiter=50,
                popsize=10,
                tol=0.1,
                workers=1,  # Windows下设为1
                seed=42,
                disp=True
            )
            initial_guess = global_result.x
            print(f"全局优化完成，误差: {global_result.fun:.6f}")
        except Exception as e:
            print(f"全局优化失败: {str(e)}, 使用默认初始参数")
        
        # 第二阶段：局部精确优化
        print("\n进行局部精确优化...")
        angle_history = []  # 存储角度变化历史
        
        def callback(xk, convergence):
            angle_value = xk[8]
            angle_history.append(angle_value)
            if np.isfinite(convergence) and convergence < 0.01:
                return True  # 提前停止
                
        result = minimize(
            lambda params: self.model_mismatch(params, "double"),
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            callback=lambda xk: callback(xk, 0.0),  # 修改回调以接受额外参数
            options={
                'maxiter': 500,
                'ftol': 1e-6,
                'disp': True
            }
        )
        
        # 保存结果
        self.optimal_params = result.x
        self.beam_model = self.double_gaussian_model(result.x)
        self.angle_history = angle_history
        
        # 计算最终误差
        self.calculate_final_error()
        
        # 打印结果
        print("\n" + "="*60)
        print(f"重构完成! 旋转角度: {self.rotation_angle:.4f}°")
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
        header += f"X坐标范围: {np.min(x_coords):.2f} to {np.max(x_coords):.2f} mm\n"
        header += f"Y坐标范围: {np.min(y_coords):.2f} to {np.max(y_coords):.2f} mm\n"
        
        # 保存数据
        np.savetxt(output_path, beam_normalized, delimiter=",", header=header, comments='# ')
        print(f"高分辨率束流分布已保存至: {output_path}")
        
        return output_path
    
    def generate_plots(self, output_prefix="beam_reconstruction"):
        """生成高级验证图表"""
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
        x_abs_error = np.abs(pred_x - self.x_data)
        y_abs_error = np.abs(pred_y - self.y_data)
        
        # 创建可视化
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        if not font_available:
            plt.rcParams['font.sans-serif'] = ['Arial']
        title_font = 16
        subtitle_font = 14
        
        # 1. 束流热力图
        ax1 = axes[0, 0]
        im = ax1.imshow(
            self.beam_model,
            extent=[np.min(self.xx), np.max(self.xx), np.min(self.yy), np.max(self.yy)],
            cmap='viridis',
            origin='lower',
            aspect='auto'
        )
        ax1.plot(self.x_offset, self.y_offset, 'rx', markersize=10, markeredgewidth=2)
        ax1.set_title("束流能量分布热图", fontsize=subtitle_font)
        ax1.set_xlabel("X位置 (mm)", fontsize=12)
        ax1.set_ylabel("Y位置 (mm)", fontsize=12)
        plt.colorbar(im, ax=ax1, label="相对强度")
        
        # 2. X方向剖面
        ax2 = axes[0, 1]
        ax2.plot(self.x_pos, self.x_data, 'b-', linewidth=2, label="实验数据")
        # 光滑处理
        x_fine = np.linspace(min(self.x_pos), max(self.x_pos), 500)
        pred_smooth = np.interp(x_fine, self.x_pos, pred_x)
        ax2.plot(x_fine, pred_smooth, 'r--', linewidth=2, label="模型预测")
        ax2.set_title("X方向剖面", fontsize=subtitle_font)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        
        # 3. Y方向剖面
        ax3 = axes[1, 0]
        ax3.plot(self.y_pos, self.y_data, 'g-', linewidth=2, label="实验数据")
        y_fine = np.linspace(min(self.y_pos), max(self.y_pos), 500)
        pred_smooth = np.interp(y_fine, self.y_pos, pred_y)
        ax3.plot(y_fine, pred_smooth, 'm--', linewidth=2, label="模型预测")
        ax3.set_title("Y方向剖面", fontsize=subtitle_font)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
        
        # 4. 角度优化轨迹
        ax4 = axes[1, 1]
        if len(self.angle_history) > 1:
            ax4.plot(range(len(self.angle_history)), self.angle_history, 'b-o', linewidth=1.5, markersize=4)
            ax4.set_title("角度优化轨迹", fontsize=subtitle_font)
            ax4.set_xlabel("迭代步数", fontsize=12)
            ax4.set_ylabel("旋转角度 (度)", fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.6)
            ax4.text(0.7, 0.1, f"初始: {self.angle_history[0]:.1f}°\n最终: {self.angle_history[-1]:.1f}°",
                     transform=ax4.transAxes, fontsize=12)
        else:
            ax4.text(0.5, 0.5, "无角度优化数据", ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.axis('off')
        
        # 5. 误差分析
        ax5 = axes[2, 0]
        bar_width = 0.35
        ax5.bar(0, np.max(x_abs_error), bar_width, color='b', label='X方向')
        ax5.bar(bar_width, np.max(y_abs_error), bar_width, color='g', label='Y方向')
        ax5.set_xticks([bar_width/2, bar_width*1.5])
        ax5.set_xticklabels(['最大绝对误差', ''])
        ax5.set_title("绝对误差分析", fontsize=subtitle_font)
        ax5.legend()
        
        # 6. 参数摘要
        ax6 = axes[2, 1]
        if self.optimal_params is not None:
            param_text = (
                "重构参数: \n"
                f"· 旋转角度 = {self.rotation_angle:.2f}°\n"
                f"· 中心偏移 = ({self.x_offset:.2f}, {self.y_offset:.2f}) mm\n"
                f"· 综合误差 = {self.recon_error*100:.2f}%\n\n"
                "高斯分量1: \n"
                f"· σX = {self.optimal_params[2]:.2f}\n"
                f"· σY = {self.optimal_params[3]:.2f}\n"
                f"· 幅度 = {self.optimal_params[4]:.2f}\n\n"
                "高斯分量2: \n"
                f"· σX = {self.optimal_params[5]:.2f}\n"
                f"· σY = {self.optimal_params[6]:.2f}\n"
                f"· 幅度 = {self.optimal_params[7]:.2f}"
            )
        else:
            param_text = "无重构参数"
        
        ax6.text(0.05, 0.9, param_text, fontsize=12, verticalalignment='top')
        ax6.axis('off')
        
        # 标题和调整布局
        fig.suptitle(f"离子束重建分析报告 (旋转角度: {self.rotation_angle:.2f}°)\n",
                   fontsize=title_font, y=0.98)
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.93)
        
        # 保存图表
        output_path = f"{output_prefix}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"质量分析图表已保存至: {output_path}")
        plt.close(fig)
        
        return output_path


# ======================= 主程序 ========================
if __name__ == "__main__":
    print("\n********** 高稳定性离子束能量分布重建系统 **********")
    
    # 初始化重建器
    reconstructor = IonBeamReconstructor(
        "x_crosssection trimmed amount profile of Movement on Y-axis.csv",
        "y_crosssection trimmed amount profile of Movement on X-axis.csv"
    )
    
    # 执行束流重建
    beam_profile = reconstructor.reconstruct_beam()
    
    # 输出结果
    csv_path = reconstructor.export_beam_csv("reconstructed_beam_highres.csv")
    plot_path = reconstructor.generate_plots()
    
    # 打印摘要
    print("\n重建完成!")
    print(f"重构误差: {reconstructor.recon_error*100:.2f}%")
    print(f"旋转角度: {reconstructor.rotation_angle:.2f}°")
    print(f"束流数据已保存至: {csv_path}")
    print(f"分析图表已保存至: {plot_path}")
    print("="*60)
