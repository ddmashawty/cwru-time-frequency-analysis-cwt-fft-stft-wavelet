"""
信号预处理器

处理信号预处理操作:
    - 降采样 (例如 48kHz -> 12kHz)
    - 信号分段与重叠
    - 基础归一化
"""

import numpy as np
from scipy.signal import decimate
from typing import List, Optional

from config.config import config


class SignalPreprocessor:
    """
    振动信号预处理器。
    
    执行降采样和分段，为时频变换准备信号。
    
    示例:
        >>> preprocessor = SignalPreprocessor()
        >>> signal = np.random.randn(48000)  # 48kHz 下的 1 秒信号
        >>> segments = preprocessor.process(signal, original_sr=48000)
        >>> print(f"生成了 {len(segments)} 个分段")
    """
    
    def __init__(self):
        """使用配置参数初始化预处理器。"""
        self.window_size = config.WINDOW_SIZE
        self.stride = config.STRIDE
        self.target_sr = config.TARGET_SR
    
    def process(
        self, 
        signal: np.ndarray, 
        original_sr: int,
        downsample: bool = True
    ) -> List[np.ndarray]:
        """
        完整的预处理流程。
        
        参数:
            signal: 原始振动信号
            original_sr: 原始采样率 (Hz)
            downsample: 是否降采样到目标采样率
            
        返回:
            信号分段列表
        """
        # 步骤 1: 如有需要则进行降采样
        if downsample and original_sr != self.target_sr:
            signal = self._downsample(signal, original_sr)
        
        # 步骤 2: 分割为重叠窗口
        segments = self._segment(signal)
        
        return segments
    
    def _downsample(self, signal: np.ndarray, original_sr: int) -> np.ndarray:
        """
        将信号降采样到目标采样率，并进行抗混叠滤波。
        
        使用区域平均方法（area resampling），保留能量分布，
        避免简单插值导致的条纹模糊。
        
        参数:
            signal: 输入信号
            original_sr: 原始采样率
            
        返回:
            降采样后的信号
        """
        factor = original_sr // self.target_sr
        
        if factor <= 1:
            return signal
        
        # 使用区域平均方法（area resampling）
        # 将信号分组后求平均，保留能量分布
        n_samples = (len(signal) // factor) * factor
        trimmed = signal[:n_samples]
        downsampled = trimmed.reshape(-1, factor).mean(axis=1)
        
        return downsampled
    
    def _segment(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        将信号分割为重叠窗口。
        
        参数:
            signal: 输入信号 (已降采样)
            
        返回:
            分段列表，每个分段长度为 window_size
        """
        segments = []
        n = len(signal)
        
        for start in range(0, n - self.window_size + 1, self.stride):
            segment = signal[start:start + self.window_size]
            
            # 只保留完整长度的分段
            if len(segment) == self.window_size:
                segments.append(segment)
        
        return segments
    
    @staticmethod
    def normalize(signal: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        归一化信号。
        
        参数:
            signal: 输入信号
            method: 归一化方法 ("minmax" 或 "zscore")
            
        返回:
            归一化后的信号
        """
        if method == "minmax":
            min_val = signal.min()
            max_val = signal.max()
            if max_val > min_val:
                return (signal - min_val) / (max_val - min_val)
            return signal
        
        elif method == "zscore":
            mean = signal.mean()
            std = signal.std()
            if std > 0:
                return (signal - mean) / std
            return signal
        
        else:
            raise ValueError(f"未知的归一化方法: {method}")


class ImageNormalizer:
    """
    二维时频表示的归一化器。
    
    处理频谱图/图像的归一化，将其调整到适合可视化和机器学习的标准范围。
    """
    
    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """
        将图像归一化到 0-255 范围 (uint8)。
        
        参数:
            image: 输入二维数组
            
        返回:
            范围 [0, 255] 的 uint8 数组
        """
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
        
        return (normalized * 255).astype(np.uint8)
    
    @staticmethod
    def normalize_to_float(image: np.ndarray) -> np.ndarray:
        """
        将图像归一化到 [0, 1] 范围 (float32)。
        
        参数:
            image: 输入二维数组
            
        返回:
            范围 [0, 1] 的 float32 数组
        """
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            return ((image - min_val) / (max_val - min_val)).astype(np.float32)
        
        return np.zeros_like(image, dtype=np.float32)
    
    @staticmethod
    def log_scale(image: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        对图像应用对数缩放 (对频谱图很有用)。
        
        参数:
            image: 输入二维数组 (应为非负值)
            epsilon: 避免 log(0) 的小值
            
        返回:
            对数缩放后的图像
        """
        return np.log(image + epsilon)
