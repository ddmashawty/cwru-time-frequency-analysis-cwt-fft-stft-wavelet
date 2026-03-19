"""
短时傅里叶变换 (STFT) 模块

使用重叠窗口生成时频频谱图。
STFT 比标准 FFT 提供更好的时频局部化。
"""

import numpy as np
from scipy.signal import stft
from typing import Tuple, Optional

from .base_transform import BaseTransform
from config.config import config


class STFTTransform(BaseTransform):
    """
    用于时频分析的短时傅里叶变换。
    
    STFT 将 FFT 应用于重叠窗口，捕获时间和频率信息。
    结果是二维频谱图，显示频率内容如何随时间演变。
    
    示例:
        >>> stft = STFTTransform()
        >>> signal = np.random.randn(1024)
        >>> spectrogram = stft.transform(signal)
        >>> print(spectrogram.shape)  # (64, 64)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = None,
        nperseg: int = None,
        noverlap: int = None,
        nfft: int = None
    ):
        """
        初始化 STFT 变换。
        
        参数:
            image_size: 目标输出大小
            nperseg: 每个段的长度。默认为 config.STFT_NPERSEG。
            noverlap: 段之间的重叠。默认为 config.STFT_NOVERLAP。
            nfft: FFT 点数。默认为 config.STFT_NFFT。
        """
        super().__init__(image_size or config.IMAGE_SIZE)
        
        self.nperseg = nperseg or config.STFT_NPERSEG
        self.noverlap = noverlap or config.STFT_NOVERLAP
        self.nfft = nfft or config.STFT_NFFT
        self.fs = config.TARGET_SR
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号应用 STFT。
        
        参数:
            signal: 一维时域信号
            
        返回:
            二维幅度频谱图
        """
        # 计算 STFT
        f, t, Zxx = stft(
            signal,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            boundary='zeros'
        )
        
        # 获取幅度 (转换为 dB 刻度以获得更好的可视化效果)
        magnitude = np.abs(Zxx)
        
        # 应用对数缩放以获得更好的动态范围
        magnitude = np.log(magnitude + 1e-10)
        
        # 调整大小到目标尺寸
        spectrogram = self.resize(magnitude)
        
        # 归一化
        spectrogram = self._normalize(spectrogram)
        
        return spectrogram
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """将图像归一化到 [0, 1] 范围。"""
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        
        return np.zeros_like(image)
    
    def get_name(self) -> str:
        """返回变换名称。"""
        return "STFT"
    
    def get_freq_time_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取频率和时间轴值。
        
        返回:
            (频率数组, 时间数组) 的元组
        """
        # 计算时间帧数
        signal_len = config.WINDOW_SIZE
        hop = self.nperseg - self.noverlap
        n_frames = (signal_len - self.noverlap) // hop
        
        freq_axis = np.fft.rfftfreq(self.nfft, d=1.0/self.fs)
        time_axis = np.arange(n_frames) * hop / self.fs
        
        return freq_axis, time_axis
