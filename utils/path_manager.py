"""
输出路径管理器

管理分层输出目录结构，保持与 CWRU 原始数据相同的分类层次。

期望的输出结构:
    outputs/
    ├── fft/
    │   ├── 12DriveEndFault/
    │   │   ├── 1730/
    │   │   │   ├── 0.007-Ball_seg0000.png
    │   │   │   └── ...
    │   │   ├── 1750/
    │   │   └── ...
    │   ├── 12FanEndFault/
    │   └── ...
    ├── stft/
    └── ...
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config.config import config, TransformType
from core.data_loader import FileInfo


@dataclass(frozen=True)
class OutputPath:
    """
    输出路径对象。
    
    包含完整路径和各级目录信息，便于管理和验证。
    
    属性:
        full_path: 完整的输出文件路径
        transform_dir: 变换类型目录 (如 'outputs/fft')
        category_dir: 分类目录 (如 'outputs/fft/12DriveEndFault/1730')
        filename: 输出文件名
    """
    full_path: str
    transform_dir: str
    category_dir: str
    filename: str
    
    def ensure_directories(self) -> None:
        """确保所有父目录存在。"""
        Path(self.category_dir).mkdir(parents=True, exist_ok=True)


class OutputPathManager:
    """
    输出路径管理器。
    
    负责根据输入文件的分类结构生成对应的输出路径。
    保持与 CWRU 原始数据相同的目录层次：
        transform_type / folder / subfolder / filename
    
    示例:
        >>> from core.data_loader import FileInfo
        >>> file_info = FileInfo(
        ...     filepath="/data/CWRU/12DriveEndFault/1730/0.007-Ball.mat",
        ...     filename="0.007-Ball.mat",
        ...     folder="12DriveEndFault",
        ...     subfolder="1730",
        ...     relative_path="12DriveEndFault/1730",
        ...     ...
        ... )
        >>> manager = OutputPathManager()
        >>> output_path = manager.get_output_path(
        ...     file_info, TransformType.FFT, segment_idx=0
        ... )
        >>> print(output_path.full_path)
        outputs/fft/12DriveEndFault/1730/0.007-Ball_seg0000.png
    """
    
    def __init__(self, output_root: Optional[str] = None):
        """
        初始化路径管理器。
        
        参数:
            output_root: 输出根目录。默认为 config.OUTPUT_ROOT。
        """
        self.output_root = Path(output_root or config.OUTPUT_ROOT).resolve()
    
    def get_output_path(
        self,
        file_info: FileInfo,
        transform_type: TransformType,
        segment_idx: int,
        extension: str = ".png"
    ) -> OutputPath:
        """
        为指定的文件片段生成输出路径。
        
        路径结构: {output_root}/{transform_type}/{relative_path}/{filename}_seg{idx}.png
        
        参数:
            file_info: 输入文件的 FileInfo 对象
            transform_type: 应用的变换类型
            segment_idx: 片段索引
            extension: 文件扩展名 (默认为 .png)
            
        返回:
            OutputPath 对象，包含完整路径和目录信息
        """
        # 变换类型目录
        transform_dir = self.output_root / transform_type.value
        
        # 分类目录 (保持原始数据的相对路径结构)
        category_dir = transform_dir / file_info.relative_path
        
        # 文件名
        basename = file_info.get_output_basename(segment_idx)
        filename = f"{basename}{extension}"
        
        # 完整路径
        full_path = category_dir / filename
        
        return OutputPath(
            full_path=str(full_path),
            transform_dir=str(transform_dir),
            category_dir=str(category_dir),
            filename=filename
        )
    
    def get_transform_dir(self, transform_type: TransformType) -> str:
        """
        获取指定变换类型的根目录。
        
        参数:
            transform_type: 变换类型
            
        返回:
            变换类型目录的完整路径
        """
        return str(self.output_root / transform_type.value)
    
    def create_transform_directories(self) -> None:
        """
        创建所有变换类型的根目录。
        
        在处理开始前调用，确保基础目录结构存在。
        """
        for transform_type in TransformType:
            transform_dir = self.output_root / transform_type.value
            transform_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_output_structure(self) -> dict:
        """
        验证输出目录结构的完整性。
        
        返回:
            包含验证结果的字典，包括存在的目录和缺失的目录。
        """
        result = {
            'exists': [],
            'missing': [],
            'total_transforms': len(TransformType)
        }
        
        for transform_type in TransformType:
            transform_dir = self.output_root / transform_type.value
            if transform_dir.exists():
                result['exists'].append(transform_type.value)
            else:
                result['missing'].append(transform_type.value)
        
        return result


class HierarchicalImageSaver:
    """
    分层图像保存器。
    
    封装图像保存逻辑，自动处理目录创建和路径管理。
    
    示例:
        >>> saver = HierarchicalImageSaver()
        >>> file_info = loader.get_all_files()[0]
        >>> image = np.random.rand(64, 64)
        >>> saver.save(image, file_info, TransformType.FFT, segment_idx=0)
    """
    
    def __init__(self, output_root: Optional[str] = None):
        """
        初始化保存器。
        
        参数:
            output_root: 输出根目录。默认为 config.OUTPUT_ROOT。
        """
        self.path_manager = OutputPathManager(output_root)
    
    def save(
        self,
        image,
        file_info: FileInfo,
        transform_type: TransformType,
        segment_idx: int
    ) -> str:
        """
        保存图像到分层目录结构中。
        
        参数:
            image: 要保存的图像数组 (numpy 数组或 PIL Image)
            file_info: 输入文件的 FileInfo 对象
            transform_type: 应用的变换类型
            segment_idx: 片段索引
            
        返回:
            保存的完整文件路径
            
        异常:
            IOError: 如果保存失败
        """
        from .io_utils import save_image
        
        # 获取输出路径
        output_path = self.path_manager.get_output_path(
            file_info, transform_type, segment_idx
        )
        
        # 确保目录存在
        output_path.ensure_directories()
        
        # 保存图像
        success = save_image(image, output_path.full_path)
        
        if not success:
            raise IOError(f"无法保存图像到: {output_path.full_path}")
        
        return output_path.full_path
