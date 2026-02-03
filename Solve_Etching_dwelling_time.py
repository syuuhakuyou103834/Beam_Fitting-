import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import os
import time
import logging
import csv
from matplotlib.patches import Circle

class IonBeamProcessor:
    def __init__(self, grid_size=160.0, resolution=1.0, wafer_diameter=150.0):
        """
        初始化离子束刻蚀处理器
        
        参数:
        grid_size: 网格尺寸 (mm)
        resolution: 空间分辨率 (mm/pixel)
        wafer_diameter: 晶圆直径 (mm)
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.wafer_diameter = wafer_diameter
        
        # 计算网格尺寸
        self.n_pixels = int(grid_size / resolution)
        self.grid_points = np.linspace(-grid_size/2, grid_size/2, self.n_pixels)
        self.X, self.Y = np.meshgrid(self.grid_points, self.grid_points)
        
        # 创建晶圆掩模
        self.wafer_radius = wafer_diameter / 2
        self.wafer_mask = self.create_wafer_mask(extend=True)
        
        # 创建径向距离图 (用于平滑外推)
        self.r_distance = np.sqrt(self.X**2 + self.Y**2)
        
        # 设置最小/最大速度限制
        self.min_speed = 0.01  # mm/s
        self.max_speed = 500.0  # mm/s
        self.base_etch_rate = 1.0  # nm/s (将被归一化)
        
        self.log(f"初始完成: {self.n_pixels}x{self.n_pixels} 网格, 分辨率 {resolution} mm/pixel")
    
    def log(self, message):
        """简单的日志记录函数"""
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
        
    def create_wafer_mask(self, extend=False):
        """创建晶圆区域的掩模
        extend: 是否将掩模略微扩展到晶圆边界之外
        """
        r = np.sqrt(self.X**2 + self.Y**2)
        if extend:
            # 将掩模略微扩展到晶圆外部1像素边界
            mask = r <= (self.wafer_radius + self.resolution)
        else:
            mask = r <= self.wafer_radius
        return mask
    
    def load_etching_data(self, file_path):
        """
        加载刻蚀量数据并进行先进的外推插值
        
        参数:
        file_path: 刻蚀量数据文件路径
        
        返回:
        160x160网格上的刻蚀量分布
        """
        self.log(f"加载刻蚀数据: {file_path}")
        df = pd.read_csv(file_path)
        x = df['X'].values
        y = df['Y'].values
        etching = df['Thickness(nm)'].values
        
        # 1. 计算径向基函数插值
        rbf = scipy.interpolate.Rbf(x, y, etching, 
                                   function='multiquadric', 
                                   smooth=0.1,  # 增加平滑度
                                   epsilon=2.0)  # 控制径向距离的影响
        
        # 2. 准备网格上的初始插值
        etching_grid = rbf(self.X, self.Y)
        
        # 3. 使用高斯平滑处理边界效应
        etching_smoothed = gaussian_filter(etching_grid, sigma=1.0)
        
        # 4. 创建内部区域掩模 (比晶圆略小3mm)
        inner_radius = self.wafer_radius - 3
        inner_mask = np.sqrt(self.X**2 + self.Y**2) <= inner_radius
        
        # 5. 组合结果：内部使用原始插值，过渡区域使用平滑值
        # 创建混合权重 (在边缘处0，内部1)
        blend_weight = np.clip((inner_radius - self.r_distance) / 3, 0, 1)
        
        # 应用混合
        etching_final = (
            blend_weight * etching_grid + 
            (1 - blend_weight) * etching_smoothed
        )
        
        # 6. 确保最小值不为零 (避免速度无限大)
        min_value = np.max([etching_final.min(), 0.01])
        etching_final[etching_final < 0.1 * min_value] = min_value
        
        self.log(f"刻蚀数据插值完成，范围: {etching_final.min():.2f}-{etching_final.max():.2f} nm")
        self.etching_map = etching_final
        return etching_final
    
    def load_beam_profile(self, file_path):
        """
        加载离子束强度分布
        
        参数:
        file_path: 离子束数据文件路径
        
        返回:
        160x160网格上的离子束强度分布
        """
        self.log(f"加载离子束数据: {file_path}")
        # 加载CSV文件
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # 跳过空行
                    # 将空字符串转换为0
                    processed_row = []
                    for val in row:
                        if val.strip() == '':
                            processed_row.append(0.0)
                        else:
                            try:
                                processed_row.append(float(val.strip()))
                            except ValueError:
                                processed_row.append(0.0)
                    data.append(processed_row)
        
        # 转换为NumPy数组
        max_width = max(len(row) for row in data) if data else 0
        padded_data = []
        for row in data:
            if len(row) < max_width:
                row += [0.0] * (max_width - len(row))
            padded_data.append(row)
        
        beam_original = np.array(padded_data)
        self.log(f"原始离子束尺寸: {beam_original.shape}")
        
        # 归一化强度
        beam_original = beam_original / beam_original.max()
        
        # 将小尺寸的离子束投影到大网格上
        beam_grid = np.zeros_like(self.X)
        
        # 计算位置偏移
        beam_h, beam_w = beam_original.shape
        start_x = (self.n_pixels - beam_w) // 2
        start_y = (self.n_pixels - beam_h) // 2
        end_x = start_x + beam_w
        end_y = start_y + beam_h
        
        # 放置离子束
        beam_grid[start_y:end_y, start_x:end_x] = beam_original
        
        # 应用轻微高斯模糊以减少FFT振铃效应
        beam_grid = gaussian_filter(beam_grid, sigma=0.5)
        
        # 归一化光束分布
        self.beam_profile = beam_grid / beam_grid.sum()
        
        self.log("离子束处理完成")
        return self.beam_profile
    
    def calculate_dwell_time(self, regularization=1e-3):
        """
        修复象限颠倒问题的停留时间计算
        """
        self.log("开始反卷积计算停留时间(修复象限)...")
        
        # 1. 获取刻蚀分布和光束分布
        E = self.etching_map
        I = self.beam_profile
        
        # 2. 添加零填充减少边界效应
        pad_size = self.n_pixels // 2
        E_padded = np.pad(E, pad_size, mode='constant', constant_values=0)
        I_padded = np.pad(I, pad_size, mode='constant', constant_values=0)
        
        # 3. 正确的相位处理(关键修复)
        # 将零点移到频谱中心
        E_shifted = ifftshift(E_padded)
        I_shifted = ifftshift(I_padded)
        
        # 4. FFT计算
        F_E = fft2(E_shifted)
        F_I = fft2(I_shifted)
        
        # 5. 计算功率谱
        I_power = np.abs(F_I)**2
        
        # 6. 维纳滤波器
        epsilon = 1.0 / regularization
        wiener_filter = np.conjugate(F_I) / (I_power + epsilon)
        
        # 7. 反卷积
        F_D = F_E * wiener_filter
        
        # 8. 逆傅里叶变换
        D_shifted = ifft2(F_D)
        
        # 9. 移回空间坐标(关键修复)
        dwell_padded = fftshift(D_shifted)
        
        # 10. 移除填充
        dwell_time = np.real(dwell_padded[pad_size:-pad_size, pad_size:-pad_size])
        
        # 11. 设置最小值保证物理合理性
        dwell_min = 0.1 / self.max_speed
        dwell_time = np.maximum(dwell_time, dwell_min)
        
        self.dwell_time = dwell_time
        self.log(f"停留时间计算完成，范围: {dwell_time.min():.4f}-{dwell_time.max():.4f} 秒")
        return dwell_time
    
    def calculate_velocity_map(self):
        """
        根据停留时间计算速度分布
        
        返回:
        160x160网格上的速度分布
        """
        self.log("计算速度分布...")
        with np.errstate(divide='ignore'):
            # 避免除零错误
            velocity = np.divide(1.0, self.dwell_time, 
                                out=np.full_like(self.dwell_time, self.max_speed),
                                where=self.dwell_time > 1e-6)
        
        # 应用物理限制
        velocity = np.clip(velocity, self.min_speed, self.max_speed)
        
        # 在晶圆边缘过渡到最大速度
        edge_mask = ~self.wafer_mask
        velocity[edge_mask] = self.max_speed
        
        # 在边界处创建平滑过渡
        transition_zone = 2  # mm
        transition_mask = np.logical_and(
            self.r_distance > self.wafer_radius - transition_zone,
            self.r_distance <= self.wafer_radius
        )
        
        # 在过渡区域线性插值
        weight = (self.r_distance[transition_mask] - (self.wafer_radius - transition_zone)) / transition_zone
        velocity[transition_mask] = weight * velocity[transition_mask] + (1 - weight) * self.max_speed
        
        self.velocity_map = velocity
        self.log(f"速度计算完成，范围: {velocity.min():.4f}-{velocity.max():.4f} mm/s")
        return velocity
    
    def generate_trajectory_recipe(self, filename="stage_recipe.csv"):
        """
        生成载台运动轨迹Recipe
        
        参数:
        filename: 输出文件名
        """
        self.log(f"生成载台轨迹: {filename}")
        
        # 提取有效点 (整个160x160网格)
        points = []
        for i in range(self.n_pixels):
            for j in range(self.n_pixels):
                x = self.X[i, j]
                y = self.Y[i, j]
                speed = self.velocity_map[i, j]
                
                # 应用速度限制
                speed = max(self.min_speed, min(self.max_speed, speed))
                points.append((x, y, speed))
        
        # 按Y（主）、X（副）排序 - 典型光栅扫描顺序
        points.sort(key=lambda p: (-p[1], p[0]))
        
        # 保存到CSV
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["X(mm)", "Y(mm)", "Velocity(mm/s)"])
                for (x, y, speed) in points:
                    writer.writerow([f"{x:.3f}", f"{y:.3f}", f"{speed:.6f}"])
            self.log(f"轨迹文件保存成功，共 {len(points)} 个点")
        except Exception as e:
            self.log(f"保存轨迹失败: {str(e)}")
    
    def plot_results(self):
        """绘制所有计算结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 刻蚀量分布
        plt.subplot(221)
        im = plt.imshow(self.etching_map, 
                       extent=[-self.grid_size/2, self.grid_size/2, 
                               -self.grid_size/2, self.grid_size/2],
                       cmap='viridis', origin='lower')
        plt.colorbar(im, label='蚀刻深度 (nm)')
        plt.title("目标刻蚀量分布")
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        # 2. 离子束强度分布
        plt.subplot(222)
        im_beam = plt.imshow(self.beam_profile, 
                            extent=[-self.grid_size/2, self.grid_size/2, 
                                    -self.grid_size/2, self.grid_size/2],
                            cmap='inferno', origin='lower')
        plt.colorbar(im_beam, label='强度')
        plt.title("离子束强度分布")
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        # 3. 停留时间分布
        plt.subplot(223)
        im_dwell = plt.imshow(self.dwell_time, 
                             extent=[-self.grid_size/2, self.grid_size/2, 
                                     -self.grid_size/2, self.grid_size/2],
                             cmap='plasma', origin='lower')
        plt.colorbar(im_dwell, label='停留时间 (秒)')
        plt.title("停留时间分布")
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        # 4. 速度分布
        plt.subplot(224)
        im_vel = plt.imshow(self.velocity_map, 
                           extent=[-self.grid_size/2, self.grid_size/2, 
                                   -self.grid_size/2, self.grid_size/2],
                           cmap='jet', origin='lower', 
                           vmin=self.min_speed, vmax=self.max_speed)
        plt.colorbar(im_vel, label='速度 (mm/s)')
        plt.title("载台速度分布")
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        plt.tight_layout()
        plt.savefig("etching_results.png", dpi=300)
        plt.close()
        self.log("结果图表已保存为 etching_results.png")

# 主程序
def main():
    # 配置参数 - 按您的需求修改
    GRID_SIZE = 160.0  # 网格尺寸 (mm)
    RESOLUTION = 1.0   # 空间分辨率 (mm/pixel)
    WAFER_DIAMETER = 150.0  # 晶圆直径 (mm)
    
    # 创建处理器实例
    processor = IonBeamProcessor(
        grid_size=GRID_SIZE,
        resolution=RESOLUTION,
        wafer_diameter=WAFER_DIAMETER
    )
    
    try:
        # 加载数据文件 - 修改为您的文件路径
        etching_file = "Eching_amount_Map.csv"
        beam_file = "ion_beam_profile.csv"
        
        # 处理数据
        processor.load_etching_data(etching_file)
        processor.load_beam_profile(beam_file)
        processor.calculate_dwell_time()
        processor.calculate_velocity_map()
        
        # 生成结果
        processor.generate_trajectory_recipe()
        processor.plot_results()
        
    except Exception as e:
        processor.log(f"处理出错: {str(e)}")
        import traceback
        processor.log(traceback.format_exc())

if __name__ == "__main__":
    # 设置可视化参数
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['figure.dpi'] = 100
    
    main()
