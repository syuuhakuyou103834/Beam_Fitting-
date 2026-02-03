import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from scipy.integrate import trapezoid
import warnings
import sys
import traceback
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings('ignore', category=RuntimeWarning)

def setup_chinese_fonts():
    """稳健的中文字体检测方法"""
    try:
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
        
        all_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'SC' in f.name]
        if all_fonts:
            plt.rcParams['font.sans-serif'] = all_fonts[:1]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用系统字体: {all_fonts[0]}")
            return True
    except:
        pass
    
    print("警告：无法加载中文字体，图表将使用英文")
    return False

font_available = setup_chinese_fonts()

class IonBeamReconstructor:
    def __init__(self, x_profile_path, y_profile_path, model_type='improved_asymmetric', preset_angle=None):
        self.x_profile_path = x_profile_path
        self.y_profile_path = y_profile_path
        self.model_type = model_type
        self.preset_angle = preset_angle  # 新增：允许用户预设角度
        self.load_and_preprocess_data()
        self.create_high_resolution_grid()
        self.reset_parameters()
        self.load_original_profile_data()
    
    def reset_parameters(self):
        self.optimal_params = None
        self.beam_model = None
        self.recon_error = 0.0
        self.rotation_angle = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.iteration_count = 0
        self.angle_history = []
        self.error_history = []
        self.optimization_stages = []
        self.iteration_count_per_stage = []
    
    def load_and_preprocess_data(self):
        print(f"读取实验数据: [X扫描: {os.path.basename(self.x_profile_path)}, Y扫描: {os.path.basename(self.y_profile_path)}]")
        x_data = np.loadtxt(self.x_profile_path, delimiter=',', skiprows=1)
        y_data = np.loadtxt(self.y_profile_path, delimiter=',', skiprows=1)
        
        # 检查并删除无效值
        x_data = x_data[~np.isnan(x_data).any(axis=1)]
        y_data = y_data[~np.isnan(y_data).any(axis=1)]
        
        self.x_pos = x_data[:, 0]
        self.y_pos = y_data[:, 0]
        
        # 归一化处理（使用线性平滑防止零值）
        x_raw = x_data[:, 1]
        x_raw[x_raw < 0] = 0  # 确保无非负值
        y_raw = y_data[:, 1]
        y_raw[y_raw < 0] = 0  # 确保无非负值
        
        # 添加基线值防止除零错误
        baseline_x = max(1e-6, np.max(x_raw)*0.01)
        baseline_y = max(1e-6, np.max(y_raw)*0.01)
        
        self.x_data = x_raw + baseline_x
        self.x_data = self.x_data / trapezoid(self.x_data, self.x_pos)
        
        self.y_data = y_raw + baseline_y
        self.y_data = self.y_data / trapezoid(self.y_data, self.y_pos)
        
        print(f"X扫描数据点: {len(self.x_pos)}, 范围: {np.min(self.x_pos):.2f} 到 {np.max(self.x_pos):.2f} mm")
        print(f"Y扫描数据点: {len(self.y_pos)}, 范围: {np.min(self.y_pos):.2f} 到 {np.max(self.y_pos):.2f} mm")

    def load_original_profile_data(self):
        x_data_raw = np.loadtxt(self.x_profile_path, delimiter=',', skiprows=1)
        self.x_pos_raw = x_data_raw[:, 0]
        self.x_data_raw = x_data_raw[:, 1]
        
        y_data_raw = np.loadtxt(self.y_profile_path, delimiter=',', skiprows=1)
        self.y_pos_raw = y_data_raw[:, 0]
        self.y_data_raw = y_data_raw[:, 1]
        
        # 确保原始数据非负
        self.x_data_raw[self.x_data_raw < 0] = 0
        self.y_data_raw[self.y_data_raw < 0] = 0
        
        self.scan_velocity = 30.0
        print(f"设置扫描速度为: {self.scan_velocity} mm/s")
        
        y_scan_length = max(self.y_pos_raw) - min(self.y_pos_raw)
        self.x_time_factor = y_scan_length / self.scan_velocity
        
        x_scan_length = max(self.x_pos_raw) - min(self.x_pos_raw)
        self.y_time_factor = x_scan_length / self.scan_velocity
    
    def create_high_resolution_grid(self):
        # 使用固定的0.25mm分辨率而不是从数据推断
        dx = 0.25
        dy = 0.25
        print(f"使用固定步长: {dx}×{dy} mm")
            
        x_min, x_max = np.min(self.x_pos), np.max(self.x_pos)
        y_min, y_max = np.min(self.y_pos), np.max(self.y_pos)
        padding = max(x_max-x_min, y_max-y_min) * 0.2  # 增加边界填充
        
        grid_x = np.arange(x_min - padding, x_max + padding + dx, dx)
        grid_y = np.arange(y_min - padding, y_max + padding + dy, dy)
        self.xx, self.yy = np.meshgrid(grid_x, grid_y)
        
        print(f"创建高分辨率网格: {len(grid_x)}×{len(grid_y)}点")
    
    # ======================= 改进的模型函数 =======================
    def improved_asymmetric_gaussian_model(self, params):
        """
        改进的非对称高斯模型
        使用更可靠的数学表达式确保非负性和稳定性
        """
        x0, y0 = params[0], params[1]
        sig_x, sig_y, amp = params[2], params[3], params[4]
        skew_x, skew_y, theta = params[5], params[6], params[7]
        
        # 将参数限制在合理范围内
        sig_x = max(0.1, sig_x)
        sig_y = max(0.1, sig_y)
        amp = max(1e-3, amp)
        
        # 旋转角度处理
        theta_rad = np.deg2rad(theta)
        rot_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])
        
        # 创建偏移网格
        xx_offset = self.xx - x0
        yy_offset = self.yy - y0
        
        # 旋转坐标
        coords = np.stack([xx_offset.ravel(), yy_offset.ravel()])
        rotated_coords = np.dot(rot_matrix, coords)
        x_rot = rotated_coords[0].reshape(xx_offset.shape)
        y_rot = rotated_coords[1].reshape(yy_offset.shape)
        
        # 改进的非对称项 - 使用指数函数的组合确保非负
        # 对于正偏斜：放大右侧，抑制左侧
        # 对于负偏斜：放大左侧，抑制右侧
        skew_factor_x = np.exp(skew_x * x_rot / (sig_x + 1e-6))
        skew_factor_y = np.exp(skew_y * y_rot / (sig_y + 1e-6))
        
        # 高斯核心 - 确保非负
        exponent = -0.5 * ((x_rot/sig_x)**2 + (y_rot/sig_y)**2) / (1.0 + 0.1*(skew_factor_x + skew_factor_y))
        intensity = amp * np.exp(exponent) * skew_factor_x * skew_factor_y
        
        # 应用非负约束
        return np.maximum(0, intensity)
    
    def get_beam_model(self, params):
        if self.model_type == 'asymmetric':
            return self.improved_asymmetric_gaussian_model(params)
        elif self.model_type == 'improved_asymmetric':
            return self.improved_asymmetric_gaussian_model(params)
        elif self.model_type == 'double_gaussian':
            return self.double_gaussian_model(params)
        elif self.model_type == 'gmm':  # 新增 GMM 支持
            return self.gmm_model(params)
        else:
            # 默认回退到双高斯
            return self.double_gaussian_model(params)
    
    def double_gaussian_model(self, params):
        """保留的双高斯模型"""
        x0, y0 = params[0], params[1]
        sig_x1, sig_y1, amp1 = params[2], params[3], params[4]
        sig_x2, sig_y2, amp2 = params[5], params[6], params[7]
        theta = np.deg2rad(params[8])
        
        self.x_offset, self.y_offset = x0, y0
        self.rotation_angle = params[8]
        
        xx_offset = self.xx - x0
        yy_offset = self.yy - y0
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated = np.dot(rotation_matrix, np.stack([xx_offset.ravel(), yy_offset.ravel()]))
        x_rot = rotated[0].reshape(xx_offset.shape)
        y_rot = rotated[1].reshape(yy_offset.shape)
        
        g1 = amp1 * np.exp(-x_rot**2/(2*sig_x1**2) - y_rot**2/(2*sig_y1**2))
        g2 = amp2 * np.exp(-x_rot**2/(2*sig_x2**2) - y_rot**2/(2*sig_y2**2))
        
        return np.maximum(0, g1 + g2)  # 确保非负


    def gmm_model(self, params):
        """
        三高斯混合模型 (GMM) - 10个参数
        参数顺序: [x0, y0, sig_x1, sig_y1, amp1, sig_x2, sig_y2, amp2, sig_x3, sig_y3, amp3, theta]
        """
        # 中心位置偏移
        x0, y0 = params[0], params[1]
        
        # 高斯分量1
        sig_x1, sig_y1, amp1 = params[2], params[3], params[4]
        # 高斯分量2
        sig_x2, sig_y2, amp2 = params[5], params[6], params[7]
        # 高斯分量3
        sig_x3, sig_y3, amp3 = params[8], params[9], params[10]
        
        # 旋转角度（最后一个参数）
        theta = np.deg2rad(params[11])
        
        # 设置中心偏移
        self.x_offset, self.y_offset = x0, y0
        self.rotation_angle = params[11]
        
        # 创建偏移网格
        xx_offset = self.xx - x0
        yy_offset = self.yy - y0
        
        # 旋转坐标
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated = np.dot(rotation_matrix, np.stack([xx_offset.ravel(), yy_offset.ravel()]))
        x_rot = rotated[0].reshape(xx_offset.shape)
        y_rot = rotated[1].reshape(yy_offset.shape)
        
        # 计算三个高斯分量
        g1 = amp1 * np.exp(-x_rot**2/(2*sig_x1**2) - y_rot**2/(2*sig_y1**2))
        g2 = amp2 * np.exp(-x_rot**2/(2*sig_x2**2) - y_rot**2/(2*sig_y2**2))
        g3 = amp3 * np.exp(-x_rot**2/(2*sig_x3**2) - y_rot**2/(2*sig_y3**2))
        
        # 叠加三个分量
        return g1 + g2 + g3
    




    # ======================= 优化核心改进 =======================
    def model_mismatch(self, params):
        beam_profile = self.get_beam_model(params)
        
        pred_x = np.zeros_like(self.x_pos)
        pred_y = np.zeros_like(self.y_pos)
        
        # 使用积分方法计算预测剖面
        for i, x_val in enumerate(self.x_pos):
            col_idx = np.argmin(np.abs(self.xx[0, :] - x_val))
            pred_x[i] = trapezoid(beam_profile[:, col_idx], self.yy[:, col_idx])
        
        for j, y_val in enumerate(self.y_pos):
            row_idx = np.argmin(np.abs(self.yy[:, 0] - y_val))
            pred_y[j] = trapezoid(beam_profile[row_idx, :], self.xx[row_idx, :])
        
        # 改进的误差计算：引入加权绝对误差
        x_weights = np.sqrt(self.x_data) + 0.01
        y_weights = np.sqrt(self.y_data) + 0.01
        
        error_x = np.sum(x_weights * np.abs(pred_x - self.x_data))
        error_y = np.sum(y_weights * np.abs(pred_y - self.y_data))
        
        # 总误差
        total_error = error_x + error_y
        
        # 记录当前迭代信息
        self.iteration_count += 1
        
        # 根据不同模型获取角度索引
        if self.model_type in ['asymmetric', 'improved_asymmetric']:
            angle_idx = 7
        elif self.model_type == 'double_gaussian':
            angle_idx = 8
        elif self.model_type == 'gmm':  # 添加GMM模型的支持
            angle_idx = 11
        else:  # 默认值
            angle_idx = 8
            
        current_angle = params[angle_idx]
        
        # 记录优化过程
        self.angle_history.append(current_angle)
        self.error_history.append(total_error)
        
        if self.iteration_count <= 20 or self.iteration_count % 10 == 0 or total_error < min(self.error_history):
            print(f"Iter {self.iteration_count:03d}: θ={current_angle:7.2f}°, Error={total_error:.6f}")
        
        return total_error
    
    def model_mismatch_with_fwhm(self, params):
        """
        带FWHM约束的目标函数
        1. 计算模型拟合误差
        2. 添加FWHM偏差惩罚项
        """
        # 原始拟合误差计算
        base_error = self.model_mismatch(params)
        
        if self.model_type == 'gmm':
            # 从参数中提取主高斯分量
            sig_x1, sig_y1 = params[2], params[3]
            
            # 计算理论FWHM (假设主高斯主导)
            fwhm_x = 2.3548 * sig_x1
            fwhm_y = 2.3548 * sig_y1
            avg_fwhm = (fwhm_x + fwhm_y) / 2
            
            # 计算实测平均FWHM (实验值)
            half_max_x = np.max(self.x_data) * 0.5
            x_indices_above = np.where(self.x_data > half_max_x)[0]
            exp_fwhm_x = abs(self.x_pos[max(x_indices_above)] - self.x_pos[min(x_indices_above)])
            
            half_max_y = np.max(self.y_data) * 0.5
            y_indices_above = np.where(self.y_data > half_max_y)[0]
            exp_fwhm_y = abs(self.y_pos[max(y_indices_above)] - self.y_pos[min(y_indices_above)])
            
            exp_avg_fwhm = (exp_fwhm_x + exp_fwhm_y) / 2
            
            # FWHM偏差惩罚 (相对误差平方)
            fwhm_error = ((avg_fwhm - exp_avg_fwhm) / exp_avg_fwhm) ** 2
            
            # 添加惩罚到总误差
            return base_error + fwhm_error * 10  # 惩罚因子10
        else:
            return base_error
    
    def _setup_rotated_grids(self, theta_deg):
        """创建给定角度的旋转网格"""
        # 创建旋转矩阵
        theta_rad = np.radians(theta_deg)
        rotation_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])
        
        # 创建未偏移的网格
        xx_offset = self.xx
        yy_offset = self.yy
        
        # 应用旋转到网格
        coords = np.stack([xx_offset.ravel(), yy_offset.ravel()])
        rotated_coords = np.dot(rotation_matrix, coords)
        xx_rot = rotated_coords[0].reshape(xx_offset.shape)
        yy_rot = rotated_coords[1].reshape(yy_offset.shape)
        
        return xx_rot, yy_rot

    def _rotated_bivariate_error_func(self, theta_rad):
        """
        计算给定旋转角度下的重构误差（仅旋转，其他参数固定）
        返回: 误差值 (float)
        """
        try:
            # 获取旋转角度（单位：度）
            theta_deg = np.degrees(theta_rad[0])
            
            # 基于模型类型设置参数
            if self.model_type in ['asymmetric', 'improved_asymmetric']:
                params = [
                    0.0, 0.0,                           # x0, y0
                    2.0, 2.0, 1.0,                      # σ_x, σ_y, amp
                    0.0, 0.0,                           # skew_x, skew_y
                    theta_deg                            # theta
                ]
            elif self.model_type == 'double_gaussian':
                params = [
                    0.0, 0.0,                           # x0, y0
                    2.0, 2.0, 0.8,                      # σ_x1, σ_y1, amp1
                    1.0, 1.0, 0.2,                      # σ_x2, σ_y2, amp2
                    theta_deg                            # theta
                ]
            elif self.model_type == 'gmm':  # 新增GMM支持 (三高斯混合模型)
                params = [
                    0.0, 0.0,                           # x0, y0
                    2.0, 2.0, 0.6,                      # σ_x1, σ_y1, amp1
                    1.5, 1.5, 0.3,                      # σ_x2, σ_y2, amp2
                    1.2, 1.2, 0.1,                      # σ_x3, σ_y3, amp3
                    theta_deg                            # theta
                ]
            else:
                # 默认使用双高斯
                params = [
                    0.0, 0.0,                           # x0, y0
                    2.0, 2.0, 0.8,                      # σ_x1, σ_y1, amp1
                    1.0, 1.0, 0.2,                      # σ极x2, σ_y2, amp2
                    theta_deg                            # theta
                ]
            
            # 使用给定参数创建束流模型
            beam_model = self.get_beam_model(params)
            
            # 计算预测X剖面
            pred_x = np.zeros_like(self.x_pos)
            for i, x_val in enumerate(self.x_pos):
                col_idx = np.argmin(np.abs(self.xx[0, :] - x_val))
                pred_x[i] = trapezoid(beam_model[:, col_idx], self.yy[:, col_idx])
            
            # 计算预测Y剖面
            pred_y = np.zeros_like(self.y_pos)
            for j, y_val in enumerate(self.y_pos):
                row_idx = np.argmin(np.abs(self.yy[:, 0] - y_val))
                pred_y[j] = trapezoid(beam_model[row_idx, :], self.xx[row_idx, :])
            
            # 归一化预测和数据
            pred_x_norm = pred_x / np.max(pred_x) if np.max(pred_x) > 0 else pred_x
            pred_y_norm = pred_y / np.max(pred_y) if np.max(pred_y) > 0 else pred_y
            
            data_x_norm = self.x_data / np.max(self.x_data)
            data_y_norm = self.y_data / np.max(self.y_data)
            
            # 计算误差
            error_x = np.mean(np.abs(pred_x_norm - data_x_norm))
            error_y = np.mean(np.abs(pred_y_norm - data_y_norm))
            
            # 返回加权总误差
            return 0.5 * error_x + 0.5 * error_y
        
        except Exception as e:
            print(f"全局扫描中计算角度{theta_deg:.1f}°时出错: {str(e)}")
            return float('inf')  # 返回一个大数表示无效

    
    def reconstruct_beam(self):
        print("\n===== 束流重构开始 =====")
        print(f"使用模型: {self.model_type}")
        print("采用分阶段优化策略...")
        
        self.iteration_count = 0
        self.angle_history = []
        self.error_history = []
        
        # 实验数据FWHM测量
        def estimate_fwhm(x, y):
            """从一维数据中测量半高宽"""
            half_max = max(y) * 0.5
            indices_above = np.where(y > half_max)[0]
            if len(indices_above) < 2:
                return None
            return abs(x[max(indices_above)] - x[min(indices_above)])
        
        x_fwhm = estimate_fwhm(self.x_pos, self.x_data)
        y_fwhm = estimate_fwhm(self.y_pos, self.y_data)
        self.exp_avg_fwhm = (x_fwhm + y_fwhm) / 2 if (x_fwhm and y_fwhm) else 4.0  # 默认4mm
        if x_fwhm:
            print(f"实验测量FWHM: X={x_fwhm:.2f}mm, Y={y_fwhm:.2f}mm, 平均={self.exp_avg_fwhm:.2f}mm")
        else:
            print(f"警告: 无法测量FWHM, 使用默认值 {self.exp_avg_fwhm:.1f}mm")
        
        # 根据模型类型设置边界和初始猜测
        if self.model_type in ['asymmetric', 'improved_asymmetric']:
            print("使用改进的非对称高斯模型 (8个参数)")
            bounds = [
                (-3, 3), (-3, 3),                  # x0, y0
                (0.3, 8), (0.3, 8), (0.1, 2),      # σ_x, σ_y, amp
                (-1.5, 1.5), (-1.5, 1.5),          # skew_x, skew_y
                (-30, 30)                          # theta
            ]
            
            # 基于FWHM设置初始σ
            sig_initial = max(1.0, self.exp_avg_fwhm / 2.35)
            
            initial_guess = [
                0.0, 0.0,                          # x0, y0
                sig_initial, sig_initial,           # σ_x, σ_y
                1.0,                               # amp
                0.1, 0.1,                         # 轻微偏斜
                0.0                                # theta
            ]
            
            simplified_params = initial_guess.copy()
            simplified_params[5] = 1.0  # sig_x
            simplified_params[6] = 1.0  # sig_y

        elif self.model_type == 'double_gaussian':
            print("使用双高斯模型 (9个参数)")
            bounds = [
                (-3, 3), (-3, 3),                 # x0, y0
                (0.3, 8), (0.3, 8), (0.1, 1.5),   # σ_x1, σ_y1, amp1
                (0.3, 6), (0.3, 6), (0.05, 0.8),  # σ_x2, σ_y2, amp2
                (-30, 30)                          # theta
            ]
            
            sig_initial = max(1.0, self.exp_avg_fwhm / 2.35)
            
            initial_guess = [
                0.0, 0.0,
                sig_initial, sig_initial,          # σ_x1, σ_y1
                0.8,                               # amp1
                max(0.5, sig_initial * 0.75),      # σ_x2
                max(0.5, sig_initial * 0.75),      # σ_y2
                0.2,                               # amp2
                0.0                                # theta
            ]
            
            simplified_params = initial_guess.copy()

        elif self.model_type == 'gmm':  # 新增 GMM 支持
            print("使用三高斯混合模型 (12个参数)")
            sig_initial = max(1.0, self.exp_avg_fwhm / 2.35)
            
            # 设置带FWHM约束的边界
            bounds = [
                (-3, 3), (-3, 3),                 # x0, y0
                (sig_initial*0.5, sig_initial*1.5),  # σ_x1范围基于FWHM
                (sig_initial*0.5, sig_initial*1.5),  # σ_y1范围基于FWHM
                (0.1, 1.5),                       # amp1
                (0.3, sig_initial*1.3),           # σ_x2 (小于主高斯)
                (0.3, sig_initial*1.3),           # σ_y2 (小于主高斯)
                (0.05, 1.0),                      # amp2
                (0.3, sig_initial),               # σ_x3 (小于次高斯)
                (0.3, sig_initial),               # σ_y3 (小于次高斯)
                (0.05, 0.8),                      # amp3
                (-30, 30)                         # theta
            ]
            
            # 基于FWHM设置初始值
            initial_guess = [
                0.0, 0.0,                          # x0, y0
                sig_initial * 1.0,                 # σ_x1
                sig_initial * 0.95,                # σ_y1
                0.70,                              # amp1 (主分量)
                sig_initial * 0.8,                 # σ_x2
                sig_initial * 0.75,                # σ_y2
                0.20,                              # amp2
                sig_initial * 0.6,                 # σ_x3
                sig_initial * 0.5,                 # σ_y3
                0.10,                              # amp3
                0.0                                # theta
            ]
            
            # 简化优化使用相同的初始猜测
            simplified_params = initial_guess.copy()

        # 检查是否有预设角度
        if self.preset_angle is not None:
            print(f"使用预设角度进行优化: {self.preset_angle}°")
            best_theta = self.preset_angle
            self.optimization_stages = [f"阶段0: 使用预设角度 θ={best_theta}°"]
        
        else:
        # ==== 阶段0: 全局粗搜索 ====
            print("阶段0: 全局粗搜索 (-90° 到 90°)")
            
            # 设置全局探索范围
            sample_angles = np.linspace(-85, 85, 36)  # 5°步长采样
            best_theta = 0.0
            best_error = float('inf')
            
            # ... 原有全局扫描代码 ...
            self.optimization_stages = [f"阶段0: 全局搜索 (5°网格) → θ={best_theta:.1f}°"]
        
        # 如果指定了预设角度，修改其边界条件
        if self.preset_angle is not None and self.model_type == 'gmm':
            # 设置角度为预设值，允许在预设角度±5°范围内优化
            angle_low = self.preset_angle - 5
            angle_high = self.preset_angle + 5
            bounds[11] = (angle_low, angle_high)
            
            # 更新初始猜测中的角度值
            initial_guess[11] = self.preset_angle
            simplified_params[11] = self.preset_angle
            
            print(f"限定角度优化范围: {angle_low:.1f}° 到 {angle_high:.1f}°")

        # ===== 阶段1: 随机初始化优化 (多起点探索) =====
        print("\n阶段1: 随机初始化优化 (探索全局最优)")
        self.optimization_stages.append("随机初始化优化 (多起点探索)")
        
        candidate_solutions = []
        num_restarts = 5  # 尝试5个不同的随机起点
        self.iteration_count = 0  # 重置迭代计数
        
        # 确保使用带FWHM约束的目标函数
        self.error_history = []
        self.angle_history = []
        
        # 全局搜索后的角度作为基础
        base_params = simplified_params.copy()
        if self.model_type in ['asymmetric', 'improved_asymmetric']:
            base_params[7] = best_theta
        elif self.model_type == 'double_gaussian':
            base_params[8] = best_theta
        elif self.model_type == 'gmm':
            base_params[11] = best_theta
            
        for restart in range(num_restarts):
            try:
                # 生成随机初始参数 (在允许范围内)
                random_init = base_params.copy()
                
                for i, (low, high) in enumerate(bounds):
                    # theta参数保持全局搜索的结果
                    if (self.model_type == 'gmm' and i == 11) or \
                       (self.model_type == 'double_gaussian' and i == 8) or \
                       (self.model_type in ['asymmetric', 'improved_asymmetric'] and i == 7):
                        continue
                    
                    # 其他参数随机变化±20%  
                    if i < len(random_init):
                        param_range = high - low
                        perturbation = np.random.uniform(-0.2, 0.2) * param_range
                        new_value = random_init[i] + perturbation
                        # 确保新值在边界内
                        if new_value < low:
                            perturbation = low - random_init[i]  # 使用最小可行扰动
                        elif new_value > high:
                            perturbation = high - random_init[i]
            
                        random_init[i] = np.clip(new_value, low, high)
                
                print(f"随机起点{restart+1}: 角度={random_init[-1] if self.model_type!='gmm' else random_init[11]:.1f}°, σx1={random_init[2]:.2f}mm")
                
                # 使用较小范围的边界进行初步优化
                stage1_bounds = [(max(low, v*0.9), min(high, v*1.1)) for (low, high), v in zip(bounds, random_init)]
                
                # 使用L-BFGS-B方法进行快速初步优化
                result_local = minimize(
                    self.model_mismatch_with_fwhm,  # 使用带FWHM约束的目标函数
                    random_init,
                    method='L-BFGS-B',
                    bounds=stage1_bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-4,
                        'disp': False,
                        'iprint': 0  # 抑制输出
                    }
                )
                
                if result_local.success and result_local.fun < float('inf'):
                    candidate_solutions.append((result_local.x, result_local.fun))
                    print(f"随机起点{restart+1}: 误差={result_local.fun:.4f}, 角度={result_local.x[11] if self.model_type=='gmm' else result_local.x[8]:.1f}°")
                else:
                    print(f"随机起点{restart+1}: 优化失败")
            except Exception as e:
                print(f"随机起点{restart+1}优化出错: {str(e)}")
        
        # 选择最佳候选解
        if candidate_solutions:
            candidate_solutions.sort(key=lambda x: x[1])  # 按误差升序排序
            best_candidate = candidate_solutions[0][0]
            print(f"阶段1最佳解: 误差={candidate_solutions[0][1]:.4f}")
            optimized_simple = best_candidate
            
            # 报告最佳解的FWHM
            if self.model_type == 'gmm':
                sig_x1 = best_candidate[2]
                fwhm = 2.3548 * sig_x1
                print(f"最佳初始解FWHM: 理论={fwhm:.2f}mm, 实验={self.exp_avg_fwhm:.2f}mm")
        else:
            print("警告: 所有随机起点优化失败，使用默认初始值")
            optimized_simple = base_params
        
        # ===== 阶段2: 完整参数优化 =====
        print("\n阶段2: 完整参数优化")
        self.optimization_stages.append("完整优化 (SLSQP)")
        
        # 基于随机优化的结果，构建完整优化的初始值
        if self.model_type in ['asymmetric', 'improved_asymmetric']:
            full_initial_guess = [
                optimized_simple[0], optimized_simple[1],   # x0, y0
                optimized_simple[2], optimized_simple[3], optimized_simple[4],  # σ_x, σ_y, amp
                optimized_simple[5], optimized_simple[6],   # skew_x, skew_y
                optimized_simple[7]                         # theta
            ]
        elif self.model_type == 'double_gaussian':
            full_initial_guess = [
                optimized_simple[0], optimized_simple[1],   # x0, y0
                optimized_simple[2], optimized_simple[3], optimized_simple[4],  # 第一个高斯
                optimized_simple[5], optimized_simple[6], optimized_simple[7], # 第二个高斯
                optimized_simple[8]                         # theta
            ]
        elif self.model_type == 'gmm':  # GMM模型
            full_initial_guess = optimized_simple.copy()  # 直接使用全部12个参数
        else:  # 默认使用双高斯
            full_initial_guess = [
                optimized_simple[0], optimized_simple[1],
                optimized_simple[2], optimized_simple[3], optimized_simple[4],
                optimized_simple[5], optimized_simple[6], optimized_simple[7],
                optimized_simple[8]
            ]
        
        # 添加FWHM约束到边界
        constrained_bounds = bounds.copy()
        if self.model_type == 'gmm':
            # 主高斯标准差FWHM约束 (确保在实验值±1mm范围内)
            sig_low = max(0.3, (self.exp_avg_fwhm - 1.0) / 2.3548)
            sig_high = min(8.0, (self.exp_avg_fwhm + 1.0) / 2.3548)
            constrained_bounds[2] = (sig_low, sig_high)  # σ_x1
            constrained_bounds[3] = (sig_low, sig_high)  # σ_y1
            
            print(f"添加FWHM约束: σ范围[{sig_low:.2f}-{sig_high:.2f}] -> FWHM范围[{(sig_low*2.3548):.2f}-{(sig_high*2.3548):.2f}]mm")
            print(f"实验FWHM: {self.exp_avg_fwhm:.2f}mm")
        
        # 执行局部优化，使用更严格的边界
        start_errors = len(self.error_history)
        result = minimize(
            self.model_mismatch_with_fwhm,  # 使用带FWHM约束的目标函数
            full_initial_guess,
            method='SLSQP',
            bounds=constrained_bounds,
            options={
                'maxiter': 200,
                'ftol': 1e-6,
                'disp': True,
                'iprint': 1
            }
        )
        
        # 保存优化结果
        stage2_iters = len(self.error_history) - start_errors
        self.iteration_count_per_stage.append(stage2_iters)
        self.optimal_params = result.x
        
        # 计算最终角度
        if self.model_type in ['asymmetric', 'improved_asymmetric']:
            self.rotation_angle = result.x[7]
        elif self.model_type == 'double_gaussian':
            self.rotation_angle = result.x[8]
        elif self.model_type == 'gmm':  # GMM模型的角度位置
            self.rotation_angle = result.x[11]
            
        # 最终输出FWHM信息
        if self.model_type == 'gmm':
            sig_x1 = result.x[2]
            sig_y1 = result.x[3]
            sim_fwhm_x = 2.3548 * sig_x1
            sim_fwhm_y = 2.3548 * sig_y1
            sim_avg_fwhm = (sim_fwhm_x + sim_fwhm_y) / 2
            print(f"最终FWHM: 模型={sim_avg_fwhm:.2f}mm, 实验={self.exp_avg_fwhm:.2f}mm")
            
        self.optimization_stages.append(f"优化完成 θ={self.rotation_angle:.1f}°")
        
        # 如果中心偏移过大，进行修正
        max_center_offset = 1.0
        if self.optimal_params[0] < -max_center_offset or self.optimal_params[0] > max_center_offset:
            print(f"警告: X中心偏移过大 ({self.optimal_params[0]:.2f}mm), 正在修正...")
            self.optimal_params[0] = 0.0  # 强制为0
        
        if self.optimal_params[1] < -max_center_offset or self.optimal_params[1] > max_center_offset:
            print(f"警告: Y中心偏移过大 ({self.optimal_params[1]:.2f}mm), 正在修正...")
            self.optimal_params[1] = 0.0  # 强制为0
        
        # 生成最终束流模型
        self.x_offset = self.optimal_params[0]
        self.y_offset = self.optimal_params[1]
        self.beam_model = self.get_beam_model(self.optimal_params)
        
        # 计算最终误差
        self.calculate_final_error()
        
        # 打印优化总结
        print("\n" + "="*60)
        print("重构完成! 旋转角度优化路径总结:")
        for i, stage in enumerate(self.optimization_stages):
            if i < len(self.iteration_count_per_stage):
                print(f" - 阶段 {i+1}: {stage} ({self.iteration_count_per_stage[i]}次迭代)")
            else:
                print(f" - 阶段 {i+1}: {stage}")
        
        if len(self.angle_history) > 0:
            print(f"\n旋转角度优化轨迹: {self.angle_history[0]:.2f}° → {self.angle_history[-1]:.2f}°")
            print(f"误差从 {self.error_history[0]:.4f} 减少到 {self.error_history[-1]:.4f}")
        else:
            print("\n没有记录角度优化数据")
        
        print(f"综合拟合误差: {self.recon_error*100:.4f}%")
        print("="*60)
        
        return self.beam_model

    
    def calculate_final_error(self):
        """计算最终误差 - 使用加权平均绝对误差"""
        # 计算预测剖面
        pred_x = np.zeros_like(self.x_pos)
        pred_y = np.zeros_like(self.y_pos)
        
        for i, x_val in enumerate(self.x_pos):
            col_idx = np.argmin(np.abs(self.xx[0, :] - x_val))
            pred_x[i] = trapezoid(self.beam_model[:, col_idx], self.yy[:, col_idx])
        
        for j, y_val in enumerate(self.y_pos):
            row_idx = np.argmin(np.abs(self.yy[:, 0] - y_val))
            pred_y[j] = trapezoid(self.beam_model[row_idx, :], self.xx[row_idx, :])
        
        # 计算加权绝对误差
        weights_x = 1.0 / (np.abs(self.x_pos) + 1.0)  # 中心区域权重更高
        weights_y = 1.0 / (np.abs(self.y_pos) + 1.0)
        
        err_x = np.mean(weights_x * np.abs(pred_x - self.x_data)) * 100
        err_y = np.mean(weights_y * np.abs(pred_y - self.y_data)) * 100
        
        self.recon_error = 0.5 * (err_x + err_y) / 100  # 转化为百分比

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
    
    def export_low_res_beam_csv(self, output_path, size=30, resolution=1):
        """
        重写低分辨率刻蚀能力计算方法
        使用更稳健的插值算法和效率因子计算方法
        """
        if self.beam_model is None:
            raise ValueError("未执行束流重构操作")
        
        # 1. 创建网格插值器
        x_points = self.xx[0, :]
        y_points = self.yy[:, 0]
        
        # 创建归一化束流分布 (确保非负)
        beam_norm = self.beam_model / np.max(self.beam_model)
        
        # 使用RegularGridInterpolator进行稳健插值
        interpolator = RegularGridInterpolator(
            (y_points, x_points), 
            beam_norm,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 2. 创建目标网格 (31x31, ±15mm)
        grid_points = np.linspace(-size/2, size/2, int(size/resolution) + 1)
        grid_x, grid_y = np.meshgrid(grid_points, grid_points)
        
        # 3. 插值得到束流分布
        points = np.array([grid_y.ravel(), grid_x.ravel()]).T
        intensities = interpolator(points).reshape(grid_x.shape)
        intensities = np.maximum(0, intensities)  # 确保非负
        
        # 4. 计算刻蚀效率因子 (精确方法)
        # 4.1 计算理论剖面
        theo_x = np.zeros_like(self.x_pos_raw)
        theo_y = np.zeros_like(self.y_pos_raw)
        
        for i, x in enumerate(self.x_pos_raw):
            # 沿Y方向积分
            y_line = self.y_pos_raw.copy()
            x_line = np.full_like(y_line, x)
            points = np.array([y_line, x_line]).T
            int_line = interpolator(points)
            theo_x[i] = trapezoid(int_line, y_line)
        
        for j, y in enumerate(self.y_pos_raw):
            # 沿X方向积分
            x_line = self.x_pos_raw.copy()
            y_line = np.full_like(x_line, y)
            points = np.array([y_line, x_line]).T
            int_line = interpolator(points)
            theo_y[j] = trapezoid(int_line, x_line)
        
        # 4.2 确保理论值非负
        theo_x = np.maximum(0, theo_x)
        theo_y = np.maximum(0, theo_y)
        
        # 4.3 计算效率因子
        # X方向效率因子：实际累积蚀刻量 / 理论累积束流积分
        etch_cum_x = trapezoid(self.x_data_raw, self.x_pos_raw)
        beam_cum_x = trapezoid(theo_x, self.x_pos_raw)
        eff_x = etch_cum_x / beam_cum_x if beam_cum_x > 0 else 0
        
        # Y方向效率因子
        etch_cum_y = trapezoid(self.y_data_raw, self.y_pos_raw)
        beam_cum_y = trapezoid(theo_y, self.y_pos_raw)
        eff_y = etch_cum_y / beam_cum_y if beam_cum_y > 0 else 0
        
        # 平均效率因子
        efficiency = 0.5 * (eff_x + eff_y)
        
        # 5. 计算刻蚀率矩阵
        etch_rate_matrix = intensities * efficiency
        etch_rate_matrix = np.maximum(0, etch_rate_matrix)  # 确保无负值
        
        # 6. 保存文件
        np.savetxt(
            output_path, 
            etch_rate_matrix, 
            delimiter=",",
            header=f"离子束刻蚀能力分布 (优化角度: {self.rotation_angle:.4f}°, 效率因子: {efficiency:.6e} nm/s)",
            comments='#'
        )
        
        # 返回结果
        self.low_res_etch_rate = etch_rate_matrix
        return output_path, efficiency, etch_rate_matrix
    
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
    

    def validate_etch_profiles(self):
        """
        验证模拟扫描结果是否匹配原始数据
        返回:
            fig_path: 验证图表路径
        """
        if not hasattr(self, 'low_res_etch_rate'):
            raise ValueError("请先调用 export_low_res_beam_csv 方法生成低刻蚀率矩阵")
        
        # 创建验证图表
        fig = plt.figure(figsize=(14, 10))
        if not font_available:
            plt.rcParams['font.sans-serif'] = ['Arial']
        
        # 创建网格 (31x31, ±15mm)
        grid_points = np.linspace(-15, 15, 31)
        
        # === X方向验证 ===
        ax1 = plt.subplot(2, 1, 1)
        # 模拟X方向扫描：对每一列沿Y方向积分
        simulated_etch_x = np.zeros(len(grid_points))
        for i in range(len(grid_points)):
            # 积分计算该列的总刻蚀量 (nm) = 刻蚀率 * 时间
            # 沿Y方向的扫描速度是30mm/s，步长1mm，每个点停留时间1/30秒
            etch_per_point = self.low_res_etch_rate[:, i] / self.scan_velocity
            simulated_etch_x[i] = trapezoid(etch_per_point, grid_points)  # 使用 scipy 的 trapezoid
        
        # 归一化模拟结果以便比较
        simulated_etch_x_norm = simulated_etch_x / np.max(simulated_etch_x)
        original_x_norm = self.x_data_raw / np.max(self.x_data_raw)
        
        # 插值以适应原始数据的点数
        interp_x = np.interp(self.x_pos_raw, grid_points, simulated_etch_x_norm)
        
        # 绘制结果对比
        ax1.plot(self.x_pos_raw, original_x_norm, 'b-', label="原始X剖面数据")
        ax1.plot(self.x_pos_raw, interp_x, 'r--', linewidth=2, label="模拟X剖面结果")
        
        # 计算并标注误差 - 防止除以零和INF错误
        abs_error_x = np.mean(np.abs(interp_x - original_x_norm))
        # 添加容差值防止除以零
        epsilon = 1e-10  # 小常数避免除以零
        rel_error_x = np.abs((interp_x - original_x_norm) / (original_x_norm + epsilon))
        valid_rel_errors_x = rel_error_x[np.isfinite(rel_error_x)]  # 移除INF值
        avg_rel_error_x = np.mean(valid_rel_errors_x) * 100 if len(valid_rel_errors_x) > 0 else 0.0
        
        ax1.set_title(f"X方向刻蚀剖面验证 (平均误差: {avg_rel_error_x:.1f}%)", fontsize=12)
        ax1.set_xlabel("X位置 (mm)", fontsize=10)
        ax1.set_ylabel("归一化刻蚀量", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        
        # 添加误差文本
        ax1.text(0.05, 0.05, f"绝对误差: {abs_error_x:.4f}\n相对误差: {avg_rel_error_x:.1f}%", 
                transform=ax1.transAxes, ha='left', bbox=dict(facecolor='white', alpha=0.7))
        
        # === Y方向验证 ===
        ax2 = plt.subplot(2, 1, 2)
        # 模拟Y方向扫描：对每一行沿X方向积分
        simulated_etch_y = np.zeros(len(grid_points))
        for j in range(len(grid_points)):
            # 积分计算该行的总刻蚀量 (nm) = 刻蚀率 * 时间
            etch_per_point = self.low_res_etch_rate[j, :] / self.scan_velocity
            simulated_etch_y[j] = trapezoid(etch_per_point, grid_points)  # 使用 scipy 的 trapezoid
        
        # 归一化模拟结果以便比较
        simulated_etch_y_norm = simulated_etch_y / np.max(simulated_etch_y)
        original_y_norm = self.y_data_raw / np.max(self.y_data_raw)
        
        # 插值以适应原始数据的点数
        interp_y = np.interp(self.y_pos_raw, grid_points, simulated_etch_y_norm)
        
        # 绘制结果对比
        ax2.plot(self.y_pos_raw, original_y_norm, 'g-', label="原始Y剖面数据")
        ax2.plot(self.y_pos_raw, interp_y, 'm--', linewidth=2, label="模拟Y剖面结果")
        
        # 计算并标注误差 - 防止除以零和INF错误
        abs_error_y = np.mean(np.abs(interp_y - original_y_norm))
        # 添加容差值防止除以零
        rel_error_y = np.abs((interp_y - original_y_norm) / (original_y_norm + epsilon))
        valid_rel_errors_y = rel_error_y[np.isfinite(rel_error_y)]  # 移除INF值
        avg_rel_error_y = np.mean(valid_rel_errors_y) * 100 if len(valid_rel_errors_y) > 0 else 0.0
        
        ax2.set_title(f"Y方向刻蚀剖面验证 (平均误差: {avg_rel_error_y:.1f}%)", fontsize=12)
        ax2.set_xlabel("Y位置 (mm)", fontsize=10)
        ax2.set_ylabel("归一化刻蚀量", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        # 添加误差文本
        ax2.text(0.05, 0.05, f"绝对误差: {abs_error_y:.4f}\n相对误差: {avg_rel_error_y:.1f}%", 
                transform=ax2.transAxes, ha='left', bbox=dict(facecolor='white', alpha=0.7))
        
        # === 添加全局标题 ===
        title_text = (f"扫描刻蚀剖面验证 (扫描速度: {self.scan_velocity}mm/s)" 
                     if font_available else
                     f"Scan Etch Profile Validation (Velocity: {self.scan_velocity}mm/s)")
        
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.92)
        
        # 保存验证图表
        fig_path = "etch_profile_validation.png"
        plt.savefig(fig_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        
        # 打印验证结果
        print("\n" + "="*50)
        print(f"扫描刻蚀剖面验证完成:")
        print(f"  - X方向平均相对误差: {avg_rel_error_x:.2f}%")
        print(f"  - Y方向平均相对误差: {avg_rel_error_y:.2f}%")
        print(f"验证图表已保存至: {fig_path}")
        print("="*50)
        
        return fig_path


# ======================= 主程序入口 ========================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 20 + ("高级离子束能量分布重建系统" if font_available else "Advanced Ion Beam Reconstruction System"))
    print("="*70)
    
    # 处理命令行参数
    preset_angle = None
    if len(sys.argv) > 1 and '=' in sys.argv[1]:
        # 解析命令行参数如 angle=30
        for arg in sys.argv[1:]:
            if '=' in arg:
                key, value = arg.split('=')
                if key == 'angle': 
                    try:
                        preset_angle = float(value)
                        print(f"设置预设角度: {preset_angle}°")
                    except ValueError:
                        print("警告: 无效的角度格式，忽略预设角度")
                        preset_angle = None
                    
    # 使用改进的非对称高斯模型
    reconstructor = IonBeamReconstructor(
        "x_crosssection trimmed amount profile of Movement on Y-axis.csv",
        "y_crosssection trimmed amount profile of Movement on X-axis.csv",
        model_type='gmm',  # 使用GMM
        preset_angle=30
    )
    
    # 执行束流重建
    try:
        beam_profile = reconstructor.reconstruct_beam()
        
        if beam_profile is not None and not np.isnan(beam_profile).any():
            print("束流重建成功完成")
        else:
            print("束流重建失败 - 结果包含无效值")
            sys.exit(1)
            
        # 输出结果
        csv_path_highres = reconstructor.export_beam_csv("reconstructed_beam_highres.csv")
        csv_path_lowres, efficiency, etch_rate = reconstructor.export_low_res_beam_csv("ion_beam_etch_rate_31x31.csv")
        
        # 可视化光束刻蚀率
        plt.figure(figsize=(10, 8))
        plt.imshow(etch_rate, cmap='viridis', extent=[-15, 15, -15, 15])
        plt.colorbar(label='刻蚀率 (nm/s)')
        plt.title(f"离子束刻蚀率分布 ({reconstructor.model_type}模型)")
        plt.xlabel("X位置 (mm)")
        plt.ylabel("Y位置 (mm)")
        plt.savefig(f"beam_etch_rate_{reconstructor.model_type}.png", dpi=150, bbox_inches='tight')
        print(f"刻蚀率分布图已保存为 'beam_etch_rate_{reconstructor.model_type}.png'")
        
        # 生成分析图表
        analysis_plot = reconstructor.generate_plots()
        
        # 执行验证
        validation_plot = reconstructor.validate_etch_profiles()
        
        print("\n" + "="*70)
        print("束流重构完成!")
        print(f"高分辨率束流数据: {csv_path_highres}")
        print(f"低分辨率刻蚀率: {csv_path_lowres}")
        print(f"刻蚀效率因子: {efficiency:.6e} nm/s")
        print(f"分析图表: {analysis_plot}")
        print(f"验证图表: {validation_plot}")
        print("="*70)

    except Exception as e:
        print(("\n!!! 重构过程中发生错误: " if font_available else "\n!!! Error during reconstruction: ") + str(e))
        traceback.print_exc()
        print("="*70)
