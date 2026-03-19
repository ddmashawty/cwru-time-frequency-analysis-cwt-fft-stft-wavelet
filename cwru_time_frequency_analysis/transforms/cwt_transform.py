"""
连续小波变换 (CWT) 模块

使用连续小波生成高分辨率时频表示。

CWT 非常适合分析非平稳信号和检测轴承振动中的瞬态特征。
"""

import numpy as np
import pywt
from typing import Tuple

from .base_transform import BaseTransform
from config.config import config


class CWTTransform(BaseTransform):
    """
    用于高分辨率时频分析的连续小波变换。
    
    使用 Morlet 小波 (时间和频率局部化的良好平衡)
    生成显示跨尺度能量分布的尺度图 (scalogram)。
    
    示例:
        >>> cwt = CWTTransform()
        >>> signal = np.random.randn(512)
        >>> scalogram = cwt.transform(signal)
        >>> print(scalogram.shape)  # (64, 64)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = None,
        wavelet: str = None,
        num_scales: int = None
    ):
        """
        初始化 CWT 变换。
        
        参数:
            image_size: 目标输出大小 (尺度, 时间)
            wavelet: 小波名称。默认为 config.CWT_WAVELET。
            num_scales: 尺度数量。默认为 config.CWT_SCALES。
        """
        super().__init__(image_size or config.IMAGE_SIZE)
        
        self.wavelet = wavelet or config.CWT_WAVELET
        self.num_scales = num_scales or config.CWT_SCALES
        self.fs = config.TARGET_SR
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号应用 CWT。
        
        参数:
            signal: 一维时域信号
            
        返回:
            二维尺度图 (CWT 系数的幅度)
        """
        # 定义尺度 (对数间隔对 CWT 很常见)
        scales = np.arange(1, self.num_scales + 1)
        
        # 计算 CWT
        coef, frequencies = pywt.cwt(
            signal,
            scales,
            self.wavelet,
            sampling_period=1.0 / self.fs
        )
        
        # 获取幅度 (尺度图)
        scalogram = np.abs(coef)
        
        # 如有需要调整大小到目标尺寸
        scalogram = self.resize(scalogram)
        
        # 归一化
        scalogram = self._normalize(scalogram)
        
        return scalogram
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1] 范围。"""
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        
        return np.zeros_like(image)
    
    def get_name(self) -> str:
        """返回变换名称。"""
        return "CWT"
    
    def get_scale_frequency_mapping(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取尺度与近似频率之间的映射。
        
        返回:
            (尺度, 对应频率) 的元组
        """
        scales = np.arange(1, self.num_scales + 1)
        
        # 对于 Morlet 小波: f = fc / (scale * delta_t)
        # 其中 fc 是小波的中心频率
        fc = pywt.central_frequency(self.wavelet)
        frequencies = fc / (scales * (1.0 / self.fs))
        
        return scales, frequencies
    
    def get_wavelet_info(self) -> dict:
        """
        获取正在使用的小波信息。
        
        返回:
            包含小波属性的字典
        """
        try:
            wavelet_obj = pywt.ContinuousWavelet(self.wavelet)
            return {
                'name': self.wavelet,
                'center_frequency': pywt.central_frequency(self.wavelet),
                'num_scales': self.num_scales,
                'wavelet_object': wavelet_obj
            }
        except:
            return {
                'name': self.wavelet,
                'num_scales': self.num_scales,
                'note': 'Using pywt built-in wavelet'
            }
