"""
时频变换的基类。

所有变换类都应继承 BaseTransform 并实现 transform() 方法。
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseTransform(ABC):
    """
    时频变换的抽象基类。
    
    子类必须实现:
        - transform(): 对信号分段执行变换
        - get_name(): 返回变换名称
    """
    
    def __init__(self, image_size: Tuple[int, int] = (64, 64)):
        """
        初始化变换。
        
        参数:
            image_size: 目标输出大小 (高度, 宽度)
        """
        self.image_size = image_size
    
    @abstractmethod
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号分段应用变换。
        
        参数:
            signal: 一维信号样本数组
            
        返回:
            二维数组 (image_size[0], image_size[1])
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回此变换的名称。"""
        pass
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        将图像调整为目标大小。
        
        参数:
            image: 输入二维数组
            
        返回:
            形状为 image_size 的调整大小后的数组
        """
        from scipy.ndimage import zoom
        
        if image.shape == self.image_size:
            return image
        
        zoom_factors = (
            self.image_size[0] / image.shape[0],
            self.image_size[1] / image.shape[1]
        )
        
        return zoom(image, zoom_factors, order=1)
