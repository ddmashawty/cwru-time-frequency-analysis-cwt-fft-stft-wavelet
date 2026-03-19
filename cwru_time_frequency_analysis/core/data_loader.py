"""
CWRU 数据集加载器

处理 CWRU 轴承数据集 MAT 文件的加载和解析。
支持全部四个数据文件夹：
    - 12DriveEndFault (12kHz 驱动端故障)
    - 12FanEndFault (12kHz 风扇端故障)
    - 48DriveEndFault (48kHz 驱动端故障)
    - NormalBaseline (正常基线)
"""

import os
import re
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from config.config import config


@dataclass(frozen=True)
class FileInfo:
    """
    CWRU 数据文件的元数据。
    
    属性:
        filepath: .mat 文件的完整路径
        filename: 基础文件名
        folder: 父文件夹名称 (例如 '12DriveEndFault')
        subfolder: 子文件夹名称 (例如 '1730' 转速文件夹)
        relative_path: 相对于数据根目录的相对路径 (例如 '12DriveEndFault/1730')
        fault_type: 故障类型 ('Normal', 'Ball', 'InnerRace', 'OuterRace')
        fault_size: 故障直径，单位为 mil (正常时为 0)
        rpm: 转速 (RPM)
        original_sr: 原始采样率 (Hz)
        is_normal: 是否为正常基线数据
        label: 正常为 0，故障为 1
    """
    filepath: str
    filename: str
    folder: str
    subfolder: str
    relative_path: str
    fault_type: str
    fault_size: int
    rpm: int
    original_sr: int
    is_normal: bool
    label: int
    
    def get_output_basename(self, segment_idx: int) -> str:
        """
        生成输出文件的基础名称（不含路径）。
        
        格式: {filename}_seg{segment_idx:04d}
        
        参数:
            segment_idx: 分段索引
            
        返回:
            基础文件名（不含扩展名）
        """
        base = self.filename.replace('.mat', '')
        return f"{base}_seg{segment_idx:04d}"


class CWRUDataLoader:
    """
    CWRU 轴承数据集加载器。
    
    处理全部四个数据文件夹，并从驱动端 (DE) 测量中提取振动信号。
    
    示例:
        >>> loader = CWRUDataLoader()
        >>> files = loader.get_all_files()
        >>> for file_info in files:
        ...     signal = loader.load_signal(file_info)
        ...     print(f"从 {file_info.filename} 加载了 {len(signal)} 个样本")
    """
    
    # 支持的 RPM 值
    RPM_VALUES = {1730, 1750, 1772, 1797}
    
    def __init__(self, data_root: Optional[str] = None):
        """
        初始化加载器。
        
        参数:
            data_root: CWRU 数据集的根目录。
                      默认为 config.DATA_ROOT。
        """
        self.data_root = Path(data_root or config.DATA_ROOT).resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(f"数据根目录未找到: {self.data_root}")
    
    def get_all_files(self) -> List[FileInfo]:
        """
        从全部四个 CWRU 文件夹中获取所有 MAT 文件。
        
        返回:
            按文件路径排序的 FileInfo 对象列表。
        """
        all_files = []
        
        for folder in config.DATA_FOLDERS:
            folder_path = self.data_root / folder
            if not folder_path.exists():
                print(f"警告: 文件夹未找到: {folder_path}")
                continue
            
            pattern = str(folder_path / config.FILE_PATTERN)
            mat_files = glob.glob(pattern, recursive=True)
            
            for filepath in mat_files:
                file_info = self._parse_filepath(filepath)
                if file_info:
                    all_files.append(file_info)
        
        # 按文件路径排序以确保可复现性
        all_files.sort(key=lambda x: x.filepath)
        return all_files
    
    def get_files_by_folder(self, folder_name: str) -> List[FileInfo]:
        """
        从特定文件夹获取所有文件。
        
        参数:
            folder_name: 文件夹名称 (例如 '12DriveEndFault')
            
        返回:
            FileInfo 对象列表。
        """
        all_files = self.get_all_files()
        return [f for f in all_files if f.folder == folder_name]
    
    def _parse_filepath(self, filepath: str) -> Optional[FileInfo]:
        """
        解析文件路径以提取元数据。
        
        参数:
            filepath: .mat 文件的完整路径
            
        返回:
            FileInfo 对象，如果解析失败则返回 None。
        """
        try:
            path = Path(filepath).resolve()
            filename = path.name
            
            # 提取文件夹层级信息
            # 路径结构: data_root/folder/subfolder/filename
            try:
                relative = path.relative_to(self.data_root)
                parts = relative.parts
                folder = parts[0] if len(parts) > 0 else ""
                subfolder = parts[1] if len(parts) > 1 else ""
                # relative_path 统一使用正斜杠 (跨平台兼容)
                relative_path = f"{folder}/{subfolder}" if subfolder else folder
            except ValueError:
                # 如果路径不在 data_root 下，手动解析
                folder = path.parent.parent.name
                subfolder = path.parent.name
                relative_path = f"{folder}/{subfolder}"
            
            # 从父目录提取 RPM
            rpm = self._extract_rpm(filepath)
            
            # 解析故障信息
            is_normal = 'Normal' in filename
            
            if is_normal:
                fault_type = 'Normal'
                fault_size = 0
                original_sr = config.SR_NORMAL_ORIGINAL
            else:
                fault_type = self._extract_fault_type(filename)
                fault_size = self._extract_fault_size(filename)
                
                # 根据文件夹确定采样率
                if '48' in folder:
                    original_sr = config.SR_FAULT_48K
                else:
                    original_sr = config.SR_FAULT_12K
            
            return FileInfo(
                filepath=str(filepath),
                filename=filename,
                folder=folder,
                subfolder=subfolder,
                relative_path=relative_path,
                fault_type=fault_type,
                fault_size=fault_size,
                rpm=rpm,
                original_sr=original_sr,
                is_normal=is_normal,
                label=0 if is_normal else 1
            )
            
        except Exception as e:
            print(f"解析 {filepath} 时出错: {e}")
            return None
    
    def _extract_rpm(self, filepath: str) -> int:
        """从文件路径提取 RPM。"""
        # 转换为正斜杠以保持一致性
        path_parts = filepath.replace('\\', '/').split('/')
        
        for part in path_parts:
            if part.isdigit() and int(part) in self.RPM_VALUES:
                return int(part)
        
        return 1797  # 默认 RPM
    
    def _extract_fault_type(self, filename: str) -> str:
        """从文件名提取故障类型。"""
        if 'Ball' in filename:
            return 'Ball'
        elif 'InnerRace' in filename:
            return 'InnerRace'
        elif 'OuterRace' in filename:
            return 'OuterRace'
        return 'Unknown'
    
    def _extract_fault_size(self, filename: str) -> int:
        """
        从文件名提取故障大小 (mil)。
        
        示例:
            0.007-Ball.mat -> 7 mil
            0.014-InnerRace.mat -> 14 mil
            0.021-OuterRace6.mat -> 21 mil
        """
        match = re.search(r'(\d+\.\d+)', filename)
        if match:
            # 从英寸 (0.007) 转换为 mil (7)
            return int(float(match.group(1)) * 1000)
        return 0
    
    def load_signal(self, file_info: FileInfo) -> np.ndarray:
        """
        从 MAT 文件加载振动信号。
        
        提取驱动端 (DE) 加速度信号。
        
        参数:
            file_info: 包含文件元数据的 FileInfo 对象
            
        返回:
            一维振动信号 numpy 数组
            
        异常:
            ValueError: 如果文件中未找到 DE 信号
            FileNotFoundError: 如果文件不存在
        """
        if not os.path.exists(file_info.filepath):
            raise FileNotFoundError(f"文件未找到: {file_info.filepath}")
        
        # 加载 MAT 文件
        mat_data = loadmat(file_info.filepath)
        
        # 查找 DE (Drive End) 变量
        var_name = self._find_de_variable(mat_data)
        if var_name is None:
            raise ValueError(f"在 {file_info.filepath} 中未找到 DE 信号")
        
        # 提取并展平信号
        signal = mat_data[var_name].flatten()
        
        return signal
    
    def _find_de_variable(self, mat_data: dict) -> Optional[str]:
        """
        在 MAT 数据中查找驱动端加速度变量名。
        
        CWRU 文件通常包含:
            - X###_DE_time: 驱动端加速度计
            - X###_FE_time: 风扇端加速度计
            - X###_BA_time: 基座加速度计
        
        参数:
            mat_data: 来自 scipy.io.loadmat 的字典
            
        返回:
            变量名字符串，如果未找到则返回 None。
        """
        # 优先级: _DE_time > 包含 'DE' > 包含 'time'
        for key in mat_data.keys():
            if '_DE_time' in key and not key.startswith('__'):
                return key
        
        # 备选: 任何包含 'DE' 的键
        for key in mat_data.keys():
            if 'DE' in key and not key.startswith('__'):
                return key
        
        # 最后备选: 任何时间序列变量
        for key in mat_data.keys():
            if 'time' in key.lower() and not key.startswith('__'):
                return key
        
        return None
    
    def get_statistics(self) -> dict:
        """
        获取已加载数据集的统计信息。
        
        返回:
            包含按文件夹、故障类型等分类计数的字典。
        """
        files = self.get_all_files()
        
        stats = {
            'total_files': len(files),
            'by_folder': {},
            'by_fault_type': {},
            'by_rpm': {},
        }
        
        for f in files:
            # 按文件夹计数
            stats['by_folder'][f.folder] = stats['by_folder'].get(f.folder, 0) + 1
            
            # 按故障类型计数
            stats['by_fault_type'][f.fault_type] = stats['by_fault_type'].get(f.fault_type, 0) + 1
            
            # 按 RPM 计数
            stats['by_rpm'][f.rpm] = stats['by_rpm'].get(f.rpm, 0) + 1
        
        return stats
