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
    
    使用复Morlet小波 (cmor1.0-2, 中心频率参数B=1.0, 带宽参数C=2)
    生成显示跨尺度能量分布的尺度图 (scalogram)。
    
    示例:
        >>> cwt = CWTTransform()
        >>> signal = np.random.randn(1024)
        >>> scalogram = cwt.transform(signal)
        >>> print(scalogram.shape)  # (64, 64)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = None,
        wavelet: str = None,
        num_scales: int = None,
        sampling_period: float = None
    ):
        """
        初始化 CWT 变换。
        
        参数:
            image_size: 目标输出大小 (尺度, 时间)
            wavelet: 小波名称。默认为 config.CWT_WAVELET。
            num_scales: 尺度数量。默认为 config.CWT_SCALES。
            sampling_period: 采样周期。默认为 config.CWT_DT。
        """
        super().__init__(image_size or config.IMAGE_SIZE)
        
        self.wavelet = wavelet or config.CWT_WAVELET
        self.num_scales = num_scales or config.CWT_SCALES
        self.sampling_period = sampling_period or config.CWT_DT
        self.fs = config.TARGET_SR
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号应用 CWT。
        
        参数:
            signal: 一维时域信号
            
        返回:
            二维尺度图 (CWT 系数的幅度)
        """
        # 设置连续小波变换参数
        sampling_period = self.sampling_period  # 采样周期
        totalscal = self.num_scales             # 总尺度
        wavename = self.wavelet                 # 小波基函数
        
        # 获取小波中心频率
        fc = pywt.central_frequency(wavename)
        
        # 根据总尺度计算参数 cparam
        cparam = 2 * fc * totalscal
        
        # 生成尺度序列 (从大到小)
        scales = cparam / np.arange(totalscal, 0, -1)
        
        # 计算 CWT
        coef, frequencies = pywt.cwt(
            signal,
            scales,
            wavename,
            sampling_period=sampling_period
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
        # 使用与 transform 方法相同的尺度计算方式
        totalscal = self.num_scales
        wavename = self.wavelet
        
        # 获取小波中心频率
        fc = pywt.central_frequency(wavename)
        
        # 计算 cparam 并生成尺度序列
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 0, -1)
        
        # 对于复Morlet小波: f = fc / (scale * delta_t)
        frequencies = fc / (scales * self.sampling_period)
        
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
