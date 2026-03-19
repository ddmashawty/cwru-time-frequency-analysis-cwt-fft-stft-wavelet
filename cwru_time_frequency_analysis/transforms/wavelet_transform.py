"""
离散小波变换 (DWT) 模块

使用 pywt 进行多级小波分解。
从小波系数重建二维表示。

DWT 提供多分辨率分析，捕获高频瞬态和低频趋势。
"""

import numpy as np
import pywt
from typing import Tuple

from .base_transform import BaseTransform
from config.config import config


class WaveletTransform(BaseTransform):
    """
    使用多级分解的离散小波变换。
    
    将信号分解为多个尺度的近似系数和细节系数。
    通过以树状结构排列系数来创建二维表示。
    
    示例:
        >>> dwt = WaveletTransform()
        >>> signal = np.random.randn(512)
        >>> coeffs_image = dwt.transform(signal)
        >>> print(coeffs_image.shape)  # (64, 64)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = None,
        wavelet: str = None,
        level: int = None
    ):
        """
        初始化小波变换。
        
        参数:
            image_size: 目标输出大小
            wavelet: 小波名称。默认为 config.WAVELET_NAME。
            level: 分解层数。默认为 config.WAVELET_LEVEL。
        """
        super().__init__(image_size or config.IMAGE_SIZE)
        
        self.wavelet = wavelet or config.WAVELET_NAME
        self.level = level or config.WAVELET_LEVEL
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        应用小波分解并创建二维表示。
        
        参数:
            signal: 一维时域信号
            
        返回:
            二维小波系数图像
        """
        # 确保信号长度适合小波变换
        signal = self._pad_signal(signal)
        
        # 执行多级分解
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # 从系数创建二维表示
        coeff_image = self._coeffs_to_2d(coeffs)
        
        # 调整大小到目标
        coeff_image = self.resize(coeff_image)
        
        # 归一化
        coeff_image = self._normalize(coeff_image)
        
        return coeff_image
    
    def _pad_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        填充信号以兼容小波分解。
        
        信号长度应能被 2^level 整除。
        
        参数:
            signal: 输入信号
            
        返回:
            填充后的信号
        """
        target_len = 2 ** self.level
        current_len = len(signal)
        
        # 找到 2^level 的下一个倍数
        new_len = ((current_len + target_len - 1) // target_len) * target_len
        
        if new_len > current_len:
            pad_width = new_len - current_len
            signal = np.pad(signal, (0, pad_width), mode='constant')
        
        return signal
    
    def _coeffs_to_2d(self, coeffs: list) -> np.ndarray:
        """
        将小波系数转换为二维图像。
        
        以结构化方式排列系数:
        - 近似系数 (cA) 作为基础
        - 每层细节系数 (cD) 堆叠
        
        参数:
            coeffs: wavedec 返回的系数数组列表
            
        返回:
            二维系数图像
        """
        # 策略: 创建方形图像并填充系数
        # 从近似系数开始
        cA = coeffs[0]
        
        # 计算所需的总大小
        total_size = sum(len(c) for c in coeffs)
        
        # 创建足够大的方形图像
        img_size = int(np.ceil(np.sqrt(total_size)))
        image = np.zeros((img_size, img_size))
        
        # 用系数填充图像
        idx = 0
        for coeff_level in coeffs:
            for val in coeff_level:
                row = idx // img_size
                col = idx % img_size
                if row < img_size:
                    image[row, col] = val
                idx += 1
        
        return image
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1] 范围。"""
        # 取绝对值 (系数可能为负)
        image = np.abs(image)
        
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        
        return np.zeros_like(image)
    
    def get_name(self) -> str:
        """返回变换名称。"""
        return "Wavelet"
    
    def get_coefficient_names(self) -> list:
        """
        获取系数数组的名称。
        
        返回:
            类似 ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1'] 的列表
        """
        names = [f'cA{self.level}']
        for i in range(self.level, 0, -1):
            names.append(f'cD{i}')
        return names
