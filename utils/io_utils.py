"""
用于保存和管理输出文件的 I/O 工具。
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config.config import config, TransformType


def create_output_dirs(root_dir: Optional[str] = None) -> dict:
    """
    为所有变换类型创建输出目录。
    
    参数:
        root_dir: 输出的根目录。默认为 config.OUTPUT_ROOT。
        
    返回:
        TransformType 到目录路径的映射字典
    """
    root = Path(root_dir or config.OUTPUT_ROOT)
    
    dirs = {}
    for transform_type in TransformType:
        dir_path = root / transform_type.value
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs[transform_type] = str(dir_path)
    
    return dirs


def get_output_path(
    filename: str,
    transform_type: TransformType,
    segment_idx: int,
    output_root: Optional[str] = None
) -> str:
    """
    为变换后的分段生成输出文件路径。
    
    参数:
        filename: 基础文件名 (例如 '0.007-Ball.mat')
        transform_type: 应用的变换类型
        segment_idx: 分段索引
        output_root: 根输出目录
        
    返回:
        完整的输出文件路径
    """
    root = Path(output_root or config.OUTPUT_ROOT)
    
    # 为变换类型创建子目录
    transform_dir = root / transform_type.value
    transform_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    base_name = Path(filename).stem
    output_name = f"{base_name}_seg{segment_idx:04d}.png"
    
    return str(transform_dir / output_name)


def save_image(
    image: np.ndarray,
    filepath: str,
    colormap: Optional[str] = None,
    dpi: int = None
) -> bool:
    """
    将二维数组保存为图像文件。
    
    支持灰度和色彩映射可视化。
    
    参数:
        image: 二维 numpy 数组
        filepath: 输出文件路径
        colormap: Matplotlib 色彩映射名称。默认为 config.COLORMAP。
        dpi: 图像 DPI。默认为 config.SAVE_DPI。
        
    返回:
        成功返回 True，否则返回 False
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        cmap = colormap or config.COLORMAP
        dpi = dpi or config.SAVE_DPI
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(image.shape[1]/dpi, image.shape[0]/dpi), dpi=dpi)
        
        # 显示图像
        im = ax.imshow(image, cmap=cmap, aspect='auto', origin='lower')
        
        # 移除坐标轴
        ax.axis('off')
        
        # 调整布局以移除边距
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 保存
        plt.savefig(filepath, dpi=dpi, pad_inches=0)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"使用 matplotlib 保存图像时出错: {e}")
        
        # 备选方案: 使用 PIL 保存为灰度图像
        try:
            # 归一化到 0-255
            img_uint8 = (image * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            img_pil.save(filepath)
            return True
        except Exception as e2:
            print(f"使用 PIL 保存图像时出错: {e2}")
            return False


def save_raw_data(
    data: np.ndarray,
    filepath: str
) -> bool:
    """
    将原始 numpy 数组保存到文件。
    
    参数:
        data: 要保存的 numpy 数组
        filepath: 输出文件路径 (应以 .npy 结尾)
        
    返回:
        成功返回 True
    """
    try:
        np.save(filepath, data)
        return True
    except Exception as e:
        print(f"保存原始数据时出错: {e}")
        return False


def load_raw_data(filepath: str) -> Optional[np.ndarray]:
    """
    从文件加载原始 numpy 数组。
    
    参数:
        filepath: .npy 文件路径
        
    返回:
        numpy 数组，如果加载失败则返回 None
    """
    try:
        return np.load(filepath)
    except Exception as e:
        print(f"加载原始数据时出错: {e}")
        return None
