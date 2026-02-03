import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import time
import traceback
import os

# ================== 中文字体设置 ==================
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：中文字体设置失败，将使用默认字体")

# ================== 参数设置 ==================
v = 30.0  # 离子束移动速度 (mm/s)
beam_size = 31  # 原始束流网格大小
high_res_factor = 4  # 提高分辨率倍数
dy = 0.25  # 高分辨率y方向步长 (mm)
original_size = 31
high_res_size = original_size * high_res_factor - (high_res_factor - 1)  # 121x121
x_range = (-15, 15)
y_range = (-15, 15)

print(f"初始分辨率: {original_size}x{original_size}")
print(f"高分辨率网格: {high_res_size}x{high_res_size}")
print(f"y方向步长: {dy:.4f} mm")

# 为结果输出创建目录
os.makedirs("fitting_progress", exist_ok=True)

# ================== 数据读取与预处理 ==================
# 1. 读取初始beamprofile
try:
    beam_low_res = np.genfromtxt('beamprofile.csv', delimiter=',')
    print(f"原始Beam形状: {beam_low_res.shape}")
    
    # 翻转矩阵行以使y增加方向一致
    beam_low_res = np.flipud(beam_low_res)
    
    # 创建原始坐标网格
    x_orig = np.linspace(x_range[0], x_range[1], original_size)
    y_orig = np.linspace(y_range[0], y_range[1], original_size)  # 严格递增
    
except Exception as e:
    print(f"读取beamprofile文件错误: {e}")
    # 创建默认高斯分布
    x_orig = np.linspace(x_range[0], x_range[1], original_size)
    y_orig = np.linspace(y_range[0], y_range[1], original_size)
    X, Y = np.meshgrid(x_orig, y_orig)
    beam_low_res = 100 * np.exp(-(X**2 + Y**2)/100)
    print("使用默认高斯分布作为初始束流")

# 2. 高精度插值 (31x31 -> 121x121)
x_high = np.linspace(x_range[0], x_range[1], high_res_size)
y_high = np.linspace(y_range[0], y_range[1], high_res_size)  # 严格递增
print(f"高分辨率x范围: {x_high[0]} 到 {x_high[-1]}, 点数: {len(x_high)}")
print(f"高分辨率y范围: {y_high[0]} 到 {y_high[-1]}, 点数: {len(y_high)}")
print(f"y方向步长: {y_high[1]-y_high[0]:.4f} mm (应为0.25)")

# 创建插值函数
try:
    interp_func = RectBivariateSpline(x_orig, y_orig, beam_low_res)
    beam_high_res = interp_func(x_high, y_high, grid=True)
    print(f"插值后Beam形状: {beam_high_res.shape}")
except Exception as e:
    print(f"插值错误: {e}")
    # 使用最近邻插值
    from scipy.interpolate import NearestNDInterpolator
    points = np.array(np.meshgrid(x_orig, y_orig)).T.reshape(-1, 2)
    values = beam_low_res.T.ravel()
    interp_func = NearestNDInterpolator(points, values)
    beam_high_res = interp_func(*np.meshgrid(x_high, y_high, indexing='ij'))
    print(f"使用最近邻插值替代, Beam形状: {beam_high_res.shape}")

# 3. 读取实验刻痕数据
try:
    cross_section_data = np.genfromtxt('x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv', delimiter=',')
    x_exp = cross_section_data[:, 0]
    y_exp = cross_section_data[:, 1]
    print(f"实验数据点数: {len(x_exp)}")
    
    # 基线校正 - 使边缘归零
    edge_samples = np.concatenate([y_exp[:5], y_exp[-5:]])
    baseline = np.mean(edge_samples)
    y_exp_corr = y_exp - baseline
    y_exp_corr = np.maximum(y_exp_corr, 0)
    print(f"基线校正完成 (基线值: {baseline:.4f} nm)")
    
except Exception as e:
    print(f"读取实验数据错误: {e}")
    # 创建模拟实验数据
    x_exp = np.linspace(x_range[0], x_range[1], high_res_size)
    # 计算当前投影
    projection = (dy / v) * np.sum(beam_high_res, axis=1)
    # 添加噪声模拟
    noise = np.random.normal(0, 1, len(projection))
    y_exp_corr = np.maximum(projection + noise, 0)
    print("使用模拟实验数据替代")

# ================== 核心迭代算法 ==================
def calculate_projection(beam_2d, dy, v):
    """计算投影/卷积积分"""
    return (dy / v) * np.sum(beam_2d, axis=1)  # 对y方向求和，得到每个x位置的积分

def radial_constraint(beam, center_idx, direction, max_val=None):
    """确保径向衰减约束"""
    beam_copy = beam.copy()
    center_val = beam_copy[center_idx]
    if max_val is None:
        max_val = center_val  # 中心点值作为最大值
    
    # 检查方向
    dx, dy = direction
    
    # 从中心点开始检查
    for dist in range(1, high_res_size//2):
        x_idx = center_idx[0] + dx * dist
        y_idx = center_idx[1] + dy * dist
        
        # 检查边界
        if x_idx < 0 or x_idx >= high_res_size or y_idx < 0 or y_idx >= high_res_size:
            break
            
        # 计算当前点应有的最大值（基于距离中心点的百分比）
        distance = np.sqrt((dx*dist)**2 + (dy*dist)**2)
        max_allowed = max_val * np.exp(-0.01*distance)  # 指数衰减
        
        # 检查值是否在允许范围内
        current_val = beam_copy[x_idx, y_idx]
        if current_val > max_allowed:
            beam_copy[x_idx, y_idx] = max_allowed  # 强制设为最大允许值
            
    return beam_copy

def apply_all_radial_constraints(beam):
    """应用所有方向的径向约束"""
    center_idx = (high_res_size//2, high_res_size//2)
    center_val = beam[center_idx]
    
    # 定义8个方向进行约束检查
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),  # 上下左右
        (1, 1), (1, -1), (-1, 1), (-1, -1) # 对角线方向
    ]
    
    constrained_beam = beam.copy()
    for direc in directions:
        constrained_beam = radial_constraint(constrained_beam, center_idx, direc, center_val)
    
    return constrained_beam

def smooth_beam(beam, sigma=0.5):
    """应用轻微高斯平滑保持光滑性"""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(beam, sigma, mode='reflect')

def plot_progress(iteration, x_pos, fitted_beam, projection, experimental):
    """绘图显示进度"""
    plt.figure(figsize=(16, 8))
    
    # 束流剖面
    plt.subplot(1, 2, 1)
    x_idx = np.where(np.abs(x_high - x_pos) < 1e-6)[0][0]
    plt.plot(y_high, fitted_beam[x_idx, :])
    plt.title(f'束流在 x={x_pos:.2f}mm 处的y方向剖面')
    plt.xlabel('Y位置 (mm)')
    plt.ylabel('刻蚀效率 (nm/s)')
    plt.grid(True)
    plt.ylim(0, None)  # 确保最低为0
    
    # 投影对比
    plt.subplot(1, 2, 2)
    plt.plot(x_high, projection, 'b-', label='当前投影', linewidth=2)
    plt.plot(x_high, experimental, 'r--', label='实验数据', linewidth=1.5)
    plt.axvline(x=x_pos, color='g', linestyle='--', label=f'当前x={x_pos:.2f}mm', linewidth=1.5)
    plt.title(f'投影与实验对比 (迭代: {iteration})')
    plt.xlabel('X位置 (mm)')
    plt.ylabel('刻蚀深度 (nm)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim(x_high[0], x_high[-1])
    plt.ylim(0, None)  # 确保最低为0
    
    # 添加文本信息
    current_rmse = np.sqrt(np.mean((projection - experimental)**2))
    plt.text(0.05, 0.95, f'RMSE: {current_rmse:.4f} nm', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'fitting_progress/progress_{iteration:03d}.png', dpi=120)
    plt.close()

# ================== 主拟合过程 ==================
start_time = time.time()
print("\n开始迭代优化...")

# 找到中心点索引
center_idx_x = np.argmin(np.abs(x_high))
center_idx_y = np.argmin(np.abs(y_high))
print(f"中心点坐标: X={x_high[center_idx_x]}mm, Y={y_high[center_idx_y]}mm")

# 确保实验数据长度匹配
if len(y_exp_corr) < high_res_size:
    # 如果实验数据点数不足，进行插值
    from scipy.interpolate import interp1d
    interp_exp = interp1d(x_exp, y_exp_corr, kind='linear', fill_value="extrapolate")
    experimental = interp_exp(x_high)
    print(f"实验数据点数不足 ({len(y_exp_corr)} < {high_res_size}), 已插值补全")
elif len(y_exp_corr) > high_res_size:
    # 如果实验数据过多，截断
    experimental = y_exp_corr[:high_res_size]
    print(f"实验数据点数过多 ({len(y_exp_corr)} > {high_res_size}), 已截断")
else:
    experimental = y_exp_corr

# 初始投影计算
initial_projection = calculate_projection(beam_high_res, dy, v)
initial_error = np.sqrt(np.mean((initial_projection - experimental)**2))
print(f"初始RMSE: {initial_error:.4f} nm")

# 开始逐点迭代
num_iterations = 0
converged = False
while not converged:
    previous_rmse = float('inf')
    for i, x_pos in enumerate(x_high):
        # 计算当前投影
        projection = calculate_projection(beam_high_res, dy, v)
        
        # 计算当前x位置的目标积分值
        exp_value = experimental[i]
        current_intensity = projection[i]
        
        # 计算调整因子
        # 避免除零错误
        if current_intensity > 1e-6:
            factor = exp_value / current_intensity
        else:
            # 如果当前积分值过小，使用基于实验值的因子
            if exp_value > 0:
                # 因子为实验值 / (整个积分范围)
                # 计算平均每点应贡献的量
                mean_contribution = exp_value * v / (dy * len(y_high))
                mean_intensity = np.mean(beam_high_res[i, :])
                
                if mean_intensity > 1e-6:
                    factor = mean_contribution / mean_intensity
                else:
                    factor = 1.0
            else:
                factor = 0.001  # 如果实验值为0，强烈减小
        
        # 限制因子范围 - 避免过大跳跃
        factor = np.clip(factor, 0.8, 1.2)
        
        # 应用调整因子到整个x位置的y方向
        beam_high_res[i, :] *= factor
        
        # 每10个点应用一次约束和平滑
        if i % 10 == 9 or i == len(x_high) - 1:
            num_iterations += 1
            
            # 应用径向约束
            beam_high_res = apply_all_radial_constraints(beam_high_res)
            
            # 应用平滑
            beam_high_res = smooth_beam(beam_high_res, sigma=0.5)
            
            # 计算当前误差
            projection = calculate_projection(beam_high_res, dy, v)
            current_rmse = np.sqrt(np.mean((projection - experimental)**2))
            
            print(f"Iter {num_iterations:03d}: x={x_pos:.2f}mm, factor={factor:.4f}, RMSE={current_rmse:.4f} nm")
            
            # 每20个点绘图一次进度
            if num_iterations % 5 == 0 or i == len(x_high) - 1:
                plot_progress(num_iterations, x_pos, beam_high_res, projection, experimental)
            
            # 检查收敛性
            if np.abs(previous_rmse - current_rmse) < 1e-5:
                converged = True
                print(f"收敛于迭代 {num_iterations} (RMSE变化小于阈值)")
                break
            previous_rmse = current_rmse
    
    # 如果完成所有点仍未收敛，再循环一次
    if i == len(x_high) - 1 and not converged:
        print("完成一整个周期但仍未收敛，开始下一个周期...")

# 最终投影计算
final_projection = calculate_projection(beam_high_res, dy, v)
final_rmse = np.sqrt(np.mean((final_projection - experimental)**2))
print(f"\n优化完成 - 最终RMSE: {final_rmse:.4f} nm, 改进率: {((initial_error - final_rmse)/initial_error*100):.2f}%")

# ================== 下采样回原始分辨率 ==================
print("\n下采样回原始分辨率...")
try:
    interp_func = RectBivariateSpline(x_high, y_high, beam_high_res)
    fitted_beam_orig = interp_func(x_orig, y_orig, grid=True)
    print("下采样完成")
except:
    # 若下采样失败，使用原始网格取最接近点
    print("RectBivariateSpline下采样失败，使用原始网格最接近点方法")
    fitted_beam_orig = np.zeros((original_size, original_size))
    for i, x in enumerate(x_orig):
        for j, y in enumerate(y_orig):
            x_idx = np.argmin(np.abs(x_high - x))
            y_idx = np.argmin(np.abs(y_high - y))
            fitted_beam_orig[i, j] = beam_high_res[x_idx, y_idx]

# 确保非负
fitted_beam_orig = np.maximum(fitted_beam_orig, 0)

# 保存结果
np.savetxt("fitted_beamprofile_31x31_optimized.csv", fitted_beam_orig, delimiter=',', fmt='%.6f')
print("优化后的束流分布已保存到: fitted_beamprofile_31x31_optimized.csv")

# ================== 结果可视化 ==================
print("\n生成结果可视化...")
plt.figure(figsize=(18, 12))

# 1. 束流表面图 (优化后)
plt.subplot(2, 2, 1)
X_orig, Y_orig = np.meshgrid(x_orig, y_orig)
contour = plt.contourf(X_orig, Y_orig, fitted_beam_orig.T, 20, cmap='viridis')
plt.colorbar(contour, label='刻蚀效率 (nm/s)')
plt.xlabel('X位置 (mm)')
plt.ylabel('Y位置 (mm)')
plt.title('优化后的束流分布 (31×31)')
plt.axis('equal')

# 2. 束流径向剖面
plt.subplot(2, 2, 2)
r = np.sqrt(X_orig**2 + Y_orig**2)
beam_values = fitted_beam_orig.T.ravel()
r_values = r.ravel()

# 径向分箱统计
r_bins = np.linspace(0, 15, 16)
radial_mean = np.zeros(len(r_bins)-1)
for i in range(len(r_bins)-1):
    mask = (r_values >= r_bins[i]) & (r_values < r_bins[i+1])
    if np.sum(mask) > 0:
        radial_mean[i] = np.mean(beam_values[mask])

r_centers = (r_bins[:-1] + r_bins[1:])/2
plt.plot(r_centers, radial_mean, 'bo-', markersize=5, linewidth=2)
plt.title('径向分布分析')
plt.xlabel('到中心的距离 (mm)')
plt.ylabel('平均刻蚀效率 (nm/s)')
plt.grid(True)

# 3. 投影对比
plt.subplot(2, 2, 3)
plt.plot(x_high, experimental, 'b-', linewidth=2, label='实验数据')
plt.plot(x_high, final_projection, 'r--', linewidth=1.5, label='拟合投影')
plt.title('投影对比')
plt.xlabel('X位置 (mm)')
plt.ylabel('刻蚀深度 (nm)')
plt.legend()
plt.grid(True)
plt.xlim(x_high[0], x_high[-1])

# 4. 误差分析
plt.subplot(2, 2, 4)
diff = experimental - final_projection
plt.plot(x_high, diff, 'g-', linewidth=1.5)
plt.fill_between(x_high, 0, diff, where=diff>0, facecolor='green', alpha=0.3, label='拟合过低区域')
plt.fill_between(x_high, 0, diff, where=diff<0, facecolor='red', alpha=0.3, label='拟合过高区域')
plt.title('差异分析 (实验值 - 拟合值)')
plt.xlabel('X位置 (mm)')
plt.ylabel('差异 (nm)')
plt.grid(True)
plt.legend()
plt.xlim(x_high[0], x_high[-1])

plt.tight_layout()
plt.savefig('final_fitting_results.png', dpi=300)
print("可视化结果已保存为 final_fitting_results.png")

# ================== 最终报告 ==================
opt_time = time.time() - start_time
print("\n===== 优化完成 =====")
print(f"优化耗时: {opt_time:.1f} 秒")
print(f"初始RMSE: {initial_error:.4f} nm")
print(f"最终RMSE: {final_rmse:.4f} nm")
print(f"改进率: {((initial_error - final_rmse)/initial_error*100):.2f}%")
print(f"迭代次数: {num_iterations}")
print("拟合结果已保存为 fitted_beamprofile_31x31_optimized.csv")
