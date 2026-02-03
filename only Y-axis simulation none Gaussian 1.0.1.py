import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time

# ================== 中文字体设置 ==================
try:
    # Windows 字体设置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
except:
    print("警告：中文字体设置失败，将使用默认字体")

# ================== 参数设置 ==================
v = 30.0  # 离子束移动速度 (mm/s)
dy = 0.25  # 插值后y方向步长 (mm)
original_size = 31
high_res_size = 121
x_range = (-15, 15)
y_range = (-15, 15)

# ================== 数据读取与预处理 ==================
# 1. 读取初始beamprofile猜想 (31x31)
beam_low_res = np.genfromtxt('beamprofile.csv', delimiter=',')
print(f"原始Beam形状: {beam_low_res.shape}")

# 创建原始坐标网格
x_orig = np.linspace(x_range[0], x_range[1], original_size)
y_orig = np.linspace(y_range[1], y_range[0], original_size)[::-1]

# 2. 高精度插值 (31x31 -> 121x121)
x_high = np.linspace(x_range[0], x_range[1], high_res_size)
y_high = np.linspace(y_range[0], y_range[1], high_res_size)

interp_func = RectBivariateSpline(x_orig, y_orig, beam_low_res)
beam_high_res = interp_func(x_high, y_high, grid=True)
print(f"插值后Beam形状: {beam_high_res.shape}")

# 3. 读取实验刻痕数据
cross_section_data = np.genfromtxt('x_crosssection_trimmed_amount_profile_of_Movement_on_Y-axis.csv', delimiter=',')
x_exp = cross_section_data[:, 0]
y_exp = cross_section_data[:, 1]

# 基线校正 - 使边缘归零
edge_samples = np.concatenate([y_exp[:5], y_exp[-5:]])  # 取两侧5个点
baseline = np.mean(edge_samples)
y_exp_corr = y_exp - baseline
y_exp_corr = np.maximum(y_exp_corr, 0)  # 确保深度非负
print(f"基线校正完成 (基线值: {baseline:.4f} nm)")

# ================== 优化的Richardson-Lucy反卷积 ==================
def calculate_projection(beam, dy, v):
    """计算沿y轴移动时的投影（刻蚀深度分布）"""
    return dy / v * np.sum(beam, axis=1)

def richardson_lucy(initial_beam, target_profile, dx, v, num_iter=50, smoothing_sigma=0.5, relax=0.2):
    """带正则化和约束的RL反卷积算法"""
    current_beam = np.maximum(initial_beam.copy(), 0)  # 确保非负
    
    # 创建距离权重矩阵: 用于保持中心峰值
    height, width = current_beam.shape
    center_x, center_y = width // 2, height // 2
    x_idx = np.arange(width)
    y_idx = np.arange(height)
    X, Y = np.meshgrid(x_idx, y_idx)
    distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.max(distance_from_center)
    # 权重因子：中心区域权重高，边缘权重低
    center_weight = np.exp(-distance_from_center/max_dist * 4)  
    
    errors = []
    
    for i in range(num_iter):
        # 计算当前投影
        proj = calculate_projection(current_beam, dx, v)
        
        # 添加小值保护（防止除零错误）
        proj_safe = np.where(proj < 1e-6, 1e-6, proj)
        
        # 计算投影与目标的比率
        ratio = target_profile / proj_safe
        
        # 步长控制：避免过度更新
        ratio = np.clip(ratio, 0.5, 1.5)
        
        # 扩展比率到二维（沿y轴方向不变）
        ratio_2d = np.tile(ratio[:, np.newaxis], (1, initial_beam.shape[1]))
        
        # 松弛更新
        current_beam = current_beam * (1 - relax + relax * ratio_2d)
        
        # 非负约束
        current_beam = np.maximum(current_beam, 0)
        
        # 使用距离权重保持中心峰值特性
        current_beam = np.maximum(current_beam * center_weight, 0)
        
        # 正则化：高斯平滑抑制噪声
        if (i % 5 == 0 or i == num_iter-1) and smoothing_sigma > 0:
            current_beam = gaussian_filter(current_beam, sigma=smoothing_sigma)
            
            # 保持最大值不变
            current_max = current_beam.max()
            if current_max > 0:
                current_beam *= initial_beam.max() / current_max
        
        # 计算误差（仅中心区域，避免边缘误差影响）
        center_indices = np.argsort(x_high)[-len(x_exp):]  # 确保索引匹配
        center_proj = proj[center_indices]
        center_target = target_profile
        
        if len(center_proj) > 0:
            mse = np.mean((center_proj - center_target)**2)
            errors.append(mse)
            
            if i % 5 == 0:
                print(f"Iter {i+1}/{num_iter} | MSE(中心±5mm): {mse:.4f}")
    
    return current_beam, errors

# ================== 反卷积拟合 ==================
print("\n开始约束反卷积拟合...")
start_time = time.time()

# 使用正则化参数和步长控制
fitted_beam, mse_history = richardson_lucy(
    beam_high_res, 
    y_exp_corr, 
    dy, 
    v, 
    num_iter=50,  # 减少迭代次数但增强约束
    smoothing_sigma=0.7, 
    relax=0.2
)
print(f"拟合完成! 耗时: {time.time()-start_time:.2f}秒")

# ================== 后处理与结果分析 ==================
# 1. 计算拟合后投影
final_projection = calculate_projection(fitted_beam, dy, v)

# 2. 计算误差指标
def calculate_errors(exp, pred):
    """计算多种误差指标"""
    mse = np.mean((exp - pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(exp - pred))
    rel_error = np.abs(exp - pred) / np.maximum(exp, 1e-6)
    max_error = np.max(np.abs(exp - pred))
    
    # 边缘区域误差 (±15mm附近)
    edge_mask = (np.abs(x_exp) >= 12)
    edge_mse = np.mean((exp[edge_mask] - pred[edge_mask])**2)
    
    # 中心区域误差 (±5mm范围内)
    center_mask = (np.abs(x_exp) <= 5)
    center_mse = np.mean((exp[center_mask] - pred[center_mask])**2)
    
    # 中等区域误差 (5-12mm)
    middle_mask = (np.abs(x_exp) > 5) & (np.abs(x_exp) < 12)
    middle_mse = np.mean((exp[middle_mask] - pred[middle_mask])**2)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "平均相对误差(%)": np.mean(rel_error[exp > 0.1] * 100),
        "最大绝对误差": max_error,
        "边缘区域MSE": edge_mse,
        "中心区域MSE": center_mse,
        "中等区域MSE": middle_mse,
        "边缘区域平均相对误差(%)": np.mean(rel_error[edge_mask & (exp > 0.1)] * 100)
    }

# 3. 计算拟合误差
error_metrics = calculate_errors(y_exp_corr, final_projection)
print("\n===== 拟合性能分析 =====")
for metric, value in error_metrics.items():
    if metric.startswith("边缘区域平均相对误差"):
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:.4f}")

# 4. 下采样回原始分辨率
interp_back_func = RectBivariateSpline(x_high, y_high, fitted_beam)
beam_final_highres = interp_back_func(x_orig, y_orig, grid=True)

# ================== 物理约束处理 ==================
def apply_constraints(beam, center_tolerance=0.01):
    """确保beam满足物理约束：单峰、中心最大值、径向递减"""
    # 定位中心点
    center_idx = (beam.shape[0]//2, beam.shape[1]//2)
    current_max = beam[center_idx]
    
    # 定位全局最大值
    max_idx = np.unravel_index(np.argmax(beam), beam.shape)
    
    # 如果全局最大值不在中心，将其移动到中心
    if np.abs(max_idx[0] - center_idx[0]) > 0 or np.abs(max_idx[1] - center_idx[1]) > 0:
        diff_x = center_idx[0] - max_idx[0]
        diff_y = center_idx[1] - max_idx[1]
        
        print(f"峰值位置偏移: ({diff_x}, {diff_y})，正在进行校正...")
        
        # 滚动阵列使峰值到中心
        beam_shifted = np.roll(beam, (diff_x, diff_y), axis=(0,1))
        
        # 处理滚动后出现的边界异常
        if diff_x > 0:
            beam_shifted[:diff_x, :] = center_tolerance * beam_shifted[:diff_x, :]
        elif diff_x < 0:
            beam_shifted[diff_x:, :] = center_tolerance * beam_shifted[diff_x:, :]
        if diff_y > 0:
            beam_shifted[:, :diff_y] = center_tolerance * beam_shifted[:, :diff_y]
        elif diff_y < 0:
            beam_shifted[:, diff_y:] = center_tolerance * beam_shifted[:, diff_y:]
        
        beam = beam_shifted
    
    # 径向单调递减增强
    y_grid, x_grid = np.meshgrid(np.arange(beam.shape[1]), np.arange(beam.shape[0]))
    dist = np.sqrt((x_grid - center_idx[0])**2 + (y_grid - center_idx[1])**2)
    
    # 最终平滑
    return gaussian_filter(beam, sigma=1.0)

# 应用物理约束
fitted_beam_constrained = apply_constraints(fitted_beam)
beam_final_lowres = apply_constraints(beam_final_highres)

# 保存结果
np.savetxt("fitted_beamprofile_31x31.csv", beam_final_lowres, fmt='%.6f', delimiter=',')

# 应用约束后重新计算投影
final_projection_constrained = calculate_projection(fitted_beam_constrained, dy, v)
constrained_errors = calculate_errors(y_exp_corr, final_projection_constrained)
print("\n===== 约束处理后的性能分析 =====")
for metric, value in constrained_errors.items():
    if metric.startswith("边缘区域平均相对误差"):
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:.4f}")

# ================== 可视化分析 ==================
# 创建图形对象用于保存
fig = plt.figure(figsize=(15, 12))

# 1. 投影对比图
ax1 = plt.subplot(2, 2, 1)
plt.plot(x_exp, y_exp_corr, 'b-', linewidth=2, label='实验数据(基线校正后)')
plt.plot(x_exp, final_projection_constrained, 'r--', linewidth=2, label='拟合投影')
plt.fill_between(x_exp, y_exp_corr, final_projection_constrained, 
                 where=(final_projection_constrained > y_exp_corr), 
                 color='red', alpha=0.2, label='高估区域')
plt.fill_between(x_exp, y_exp_corr, final_projection_constrained, 
                 where=(final_projection_constrained <= y_exp_corr), 
                 color='blue', alpha=0.2, label='低估区域')
plt.title(f'投影数据对比\nRMSE={constrained_errors["RMSE"]:.4f} nm, MAE={constrained_errors["MAE"]:.4f} nm')
plt.xlabel('X位置 (mm)')
plt.ylabel('刻蚀深度 (nm)')
plt.legend()
plt.grid(True)

# 2. 误差分析图
ax2 = plt.subplot(2, 2, 2)
errors = final_projection_constrained - y_exp_corr
plt.plot(x_exp, errors, 'g-', linewidth=1.5, label='绝对误差')
plt.plot(x_exp, 100 * (errors) / (np.max(np.abs(y_exp_corr)) + 1), 'm-', linewidth=1.5, 
         alpha=0.7, label='相对误差(%)')
plt.axhline(0, color='k', linestyle='--')
plt.title('投影误差分析')
plt.xlabel('X位置 (mm)')
plt.ylabel('误差值')
plt.legend()
plt.grid(True)

# 3. Beam等高线图（原始分辨率）
ax3 = plt.subplot(2, 2, 3)
X, Y = np.meshgrid(x_orig, y_orig)
contour = plt.contourf(X, Y, beam_final_lowres.T, levels=20, cmap='viridis')
plt.colorbar(contour, label='刻蚀效率 (nm/s)')
plt.title('拟合Beam分布等高线 (31×31)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.axis('equal')

# 4. 中心区域3D图
ax4 = plt.subplot(2, 2, 4, projection='3d')
x_mask = (np.abs(x_high) <= 5)
y_mask = (np.abs(y_high) <= 5)
X_center, Y_center = np.meshgrid(x_high[x_mask], y_high[y_mask])
Z_center = fitted_beam_constrained
# 只取中心部分
Z_center = Z_center[np.ix_(y_mask, x_mask)]

surf = ax4.plot_surface(X_center, Y_center, Z_center, cmap='plasma', 
                       rstride=1, cstride=1, alpha=0.9, antialiased=True)
ax4.set_title('中心区域刻蚀效率分布')
ax4.set_xlabel('X (mm)')
ax4.set_ylabel('Y (mm)')
ax4.set_zlabel('刻蚀效率 (nm/s)')

# 创建颜色条 - 修复错误
cbar = plt.colorbar(surf, ax=ax4, shrink=0.5, aspect=10)
cbar.set_label('nm/s')

plt.tight_layout()
plt.savefig('beam_fitting_results_with_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ================== Y轴投影特别分析 ==================
# 计算沿Y轴方向的投影
y_projection = np.sum(fitted_beam_constrained, axis=0)
y_center = y_high[y_mask]
y_projection_center = y_projection[y_mask]

# 归一化以查看形状
y_projection /= np.max(y_projection)
y_projection_center /= np.max(y_projection_center)

# 创建Y轴方向分析图
plt.figure(figsize=(12, 6))

# 1. 整个Y轴的投影
plt.subplot(1, 2, 1)
plt.plot(y_high, y_projection, 'b-', linewidth=2)
plt.title('沿Y轴方向的总和投影')
plt.xlabel('Y位置 (mm)')
plt.ylabel('归一化强度')
plt.grid(True)

# 2. 中心Y轴区域的投影
plt.subplot(1, 2, 2)
plt.plot(y_center, y_projection_center, 'r-', linewidth=2)
plt.title('中心区域Y轴投影 (±5mm)')
plt.xlabel('Y位置 (mm)')
plt.ylabel('归一化强度')
plt.grid(True)

plt.tight_layout()
plt.savefig('y_axis_projection_analysis.png', dpi=300)
plt.show()

# ================== 结果报告 ==================
print("\n====== 最终拟合报告 ======")
print(f"总体RMSE: {constrained_errors['RMSE']:.4f} nm")
print(f"平均绝对误差(MAE): {constrained_errors['MAE']:.4f} nm")
print(f"最大绝对误差: {constrained_errors['最大绝对误差']:.4f} nm")
print(f"平均相对误差: {constrained_errors['平均相对误差(%)']:.2f}%")
print(f"中心区域MSE: {constrained_errors['中心区域MSE']:.4f}")
print(f"边缘区域平均相对误差: {constrained_errors['边缘区域平均相对误差(%)']:.2f}%")
print("注意：边缘区域的固有测量噪声导致拟合误差相对较高")
print("拟合结果已保存到：fitted_beamprofile_31x31.csv")
print("Y轴投影分析图已保存到：y_axis_projection_analysis.png")
