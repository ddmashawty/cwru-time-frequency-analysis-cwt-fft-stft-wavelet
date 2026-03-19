"""
CWRU 时频分析的配置设置。

本模块包含用于处理 CWRU 轴承数据集的各种时频变换的所有超参数和路径配置。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import os


class TransformType(Enum):
    """支持的时频变换类型。"""
    FFT = "fft"                    # 快速傅里叶变换 (Fast Fourier Transform)
    STFT = "stft"                  # 短时傅里叶变换 (Short-Time Fourier Transform)
    WAVELET = "wavelet"            # 离散小波变换 (Discrete Wavelet Transform)
    CWT = "cwt"                    # 连续小波变换 (Continuous Wavelet Transform)


@dataclass
class Config:
    """
    CWRU 数据处理的配置类。
    
    所有参数都设置为类属性，以便于访问和修改，无需更改代码逻辑。
    """
    
    # ========================================================================
    # 数据路径
    # ========================================================================
    # CWRU 数据集的根目录
    DATA_ROOT: str = r"C:\Users\22734\Desktop\python\CWRU"
    
    # 每种变换类型的输出目录
    OUTPUT_ROOT: str = "outputs"
    
    @classmethod
    def get_output_dir(cls, transform_type: TransformType) -> str:
        """获取特定变换类型的输出目录。"""
        return os.path.join(cls.OUTPUT_ROOT, transform_type.value)
    
    # ========================================================================
    # 采样率设置
    # ========================================================================
    # 原始采样率 (Hz)
    SR_NORMAL_ORIGINAL: int = 48000    # 正常基线数据
    SR_FAULT_12K: int = 12000          # 12k 驱动端/风扇端故障数据
    SR_FAULT_48K: int = 48000          # 48k 驱动端故障数据
    
    # 降采样后的目标采样率
    TARGET_SR: int = 12000
    
    # ========================================================================
    # 信号分段
    # ========================================================================
    # 分段窗口大小 (样本数)
    WINDOW_SIZE: int = 512
    
    # 连续窗口之间的重叠比例 (0.0 - 1.0)
    OVERLAP_RATIO: float = 0.5
    
    # 派生属性：窗口之间的步长
    @property
    def STRIDE(self) -> int:
        """根据窗口大小和重叠比例计算步长。"""
        return int(self.WINDOW_SIZE * (1 - self.OVERLAP_RATIO))
    
    # ========================================================================
    # 输出图像设置
    # ========================================================================
    # 所有变换的目标图像大小 (高度, 宽度)
    IMAGE_SIZE: Tuple[int, int] = (64, 64)
    
    # 保存图像的 DPI
    SAVE_DPI: int = 100
    
    # 可视化使用的色彩映射
    COLORMAP: str = "jet"
    
    # ========================================================================
    # 变换特定设置
    # ========================================================================
    
    # --- FFT 设置 ---
    FFT_NORM: str = "ortho"            # FFT 归一化模式
    
    # --- STFT 设置 ---
    STFT_NPERSEG: int = 256            # 每个段的长度
    STFT_NOVERLAP: int = 128           # 段之间的重叠
    STFT_NFFT: int = 256               # FFT 点数
    
    # --- 离散小波变换设置 ---
    WAVELET_NAME: str = "db4"          # Daubechies 小波
    WAVELET_LEVEL: int = 5             # 分解层数
    
    # --- 连续小波变换设置 ---
    CWT_WAVELET: str = "morl"          # Morlet 小波
    CWT_SCALES: int = 64               # 尺度数量
    
    # ========================================================================
    # 处理设置
    # ========================================================================
    # 随机种子，用于结果可复现
    RANDOM_SEED: int = 42
    
    # 工作进程数 (None = 使用所有核心)
    NUM_WORKERS: int = 4
    
    # 详细输出
    VERBOSE: bool = True
    
    # 文件搜索模式
    FILE_PATTERN: str = "**/*.mat"
    
    # ========================================================================
    # CWRU 数据集结构
    # ========================================================================
    # 要处理的子目录 (全部 4 个文件夹)
    DATA_FOLDERS: Tuple[str, ...] = (
        "12DriveEndFault",      # 12kHz 驱动端故障数据
        "12FanEndFault",        # 12kHz 风扇端故障数据
        "48DriveEndFault",      # 48kHz 驱动端故障数据
        "NormalBaseline",       # 正常基线数据 (48kHz)
    )


# 全局配置实例
config = Config()
