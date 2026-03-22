"""
快速傅里叶变换 (FFT) 模块

从振动信号生成幅度谱。

注意: FFT 会丢失时间信息，因此我们对每个时间段单独应用它，
得到每个时间段的一维频谱。为了创建二维表示，我们将连续时间段的频谱堆叠起来。
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple

from .base_transform import BaseTransform
from config.config import config


class FFTTransform(BaseTransform):
    """
    用于振动分析的快速傅里叶变换。
    
    通过以下步骤生成类似二维频谱图的表示:
    1. 将信号分段
    2. 计算每个分段的 FFT 幅度谱
    3. 将频谱堆叠为二维图像的列
    
    得到的图像显示频率内容随时间的变化，
    类似于频谱图但使用均匀的时间窗口。
    
    示例:
        >>> fft = FFTTransform()
        >>> signal = np.random.randn(1024)  # 一个分段
        >>> spectrum = fft.transform(signal)
        >>> print(spectrum.shape)  # (64, 64)
    """
    
    def __init__(self, image_size: Tuple[int, int] = None):
        """
        初始化 FFT 变换。
        
        参数:
            image_size: 目标输出大小 (高度=频率 bins, 宽度=时间帧)
        """
        super().__init__(image_size or config.IMAGE_SIZE)
        self.norm = config.FFT_NORM
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号分段应用 FFT 并创建二维表示。
        
        对于单个分段输入，通过以下方式创建二维图像:
        1. 计算 FFT 幅度谱
        2. 重塑为二维 (freq_bins x time_bins)
        
        参数:
            signal: 一维信号分段或多个分段的连接
            
        返回:
            二维幅度谱图像
        """
        # 确保信号长度合适
        if len(signal) < 64:
            signal = np.pad(signal, (0, 64 - len(signal)))
        
        # 计算 FFT
        fft_result = np.fft.rfft(signal, norm=self.norm)
        magnitude = np.abs(fft_result)
        
        # 通过重塑创建二维表示
        # 策略: 将一维频谱重塑为二维网格
        spectrum_2d = self._reshape_to_2d(magnitude)
        
        # 调整大小到目标尺寸
        spectrum_2d = self.resize(spectrum_2d)
        
        # 归一化到 [0, 1]
        spectrum_2d = self._normalize(spectrum_2d)
        
        return spectrum_2d
    
    def _reshape_to_2d(self, spectrum: np.ndarray) -> np.ndarray:
        """
        将一维频谱重塑为近似的二维方阵。
        
        使用零填充或截断以适应方形形状，
        然后可以调整大小到目标尺寸。
        
        参数:
            spectrum: 一维 FFT 幅度谱
            
        返回:
            二维重塑后的频谱
        """
        n = len(spectrum)
        
        # 计算大致方形输出的尺寸
        # 使用 2 的下一个幂或最近的平方数
        size = int(np.ceil(np.sqrt(n)))
        
        # 填充或截断以适应
        padded = np.zeros(size * size)
        padded[:min(n, size * size)] = spectrum[:min(n, size * size)]
        
        # 重塑为二维
        return padded.reshape(size, size)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """将图像归一化到 [0, 1] 范围。"""
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        
        return np.zeros_like(image)
    
    def get_name(self) -> str:
        """返回变换名称。"""
        return "FFT"
    
    def get_frequency_axis(self, fs: float = None) -> np.ndarray:
        """
        获取用于绘图的频率轴值。
        
        参数:
            fs: 采样频率。默认为 config.TARGET_SR。
            
        返回:
            频率值数组 (Hz)
        """
        if fs is None:
            fs = config.TARGET_SR
        
        n_fft = config.WINDOW_SIZE
        return np.fft.rfftfreq(n_fft, d=1.0/fs)
