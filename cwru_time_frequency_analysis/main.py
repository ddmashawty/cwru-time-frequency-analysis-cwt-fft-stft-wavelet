"""
CWRU 时频分析流程

使用多种时频变换处理 CWRU 轴承数据集的主入口点。

用法:
    python main.py
    
    # 仅处理特定文件夹
    python main.py --folders 12DriveEndFault NormalBaseline
    
    # 仅使用特定变换
    python main.py --transforms fft cwt
"""

import argparse
import sys
import time
from typing import List, Optional
from pathlib import Path

import numpy as np

# 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config, TransformType
from core.data_loader import CWRUDataLoader
from core.preprocessor import SignalPreprocessor
from transforms import FFTTransform, STFTTransform, WaveletTransform, CWTTransform
from utils.path_manager import OutputPathManager, HierarchicalImageSaver


class ProcessingPipeline:
    """
    使用多种变换处理 CWRU 数据的主流程。
    
    编排整个工作流程:
        1. 加载数据文件
        2. 预处理信号 (下采样、分帧)
        3. 应用时频变换
        4. 按原始数据结构保存结果到分层目录
    """
    
    def __init__(self):
        """初始化流程组件。"""
        self.data_loader = CWRUDataLoader()
        self.preprocessor = SignalPreprocessor()
        self.image_saver = HierarchicalImageSaver()
        self.path_manager = OutputPathManager()
        
        # 初始化变换器
        self.transforms = {
            TransformType.FFT: FFTTransform(),
            TransformType.STFT: STFTTransform(),
            TransformType.WAVELET: WaveletTransform(),
            TransformType.CWT: CWTTransform(),
        }
        
        # 创建输出目录结构
        self.path_manager.create_transform_directories()
        
        # 统计信息
        self.stats = {
            'files_processed': 0,
            'segments_processed': 0,
            'images_saved': 0,
            'errors': [],
        }
    
    def process_file(
        self,
        file_info,
        transform_types: Optional[List[TransformType]] = None
    ) -> bool:
        """
        使用指定变换处理单个数据文件。
        
        参数:
            file_info: 包含文件元数据的 FileInfo 对象
            transform_types: 要应用的变换列表。如果为 None，应用所有变换。
            
        返回:
            如果成功返回 True
        """
        if transform_types is None:
            transform_types = list(TransformType)
        
        try:
            # 加载信号
            if config.VERBOSE:
                print(f"  加载: {file_info.filename} ({file_info.relative_path})")
            
            signal = self.data_loader.load_signal(file_info)
            
            # 预处理 (下采样 + 分帧)
            segments = self.preprocessor.process(signal, file_info.original_sr)
            
            if len(segments) == 0:
                print(f"  警告: 未从 {file_info.filename} 生成分段")
                return False
            
            if config.VERBOSE:
                print(f"    生成了 {len(segments)} 个分段")
            
            # 使用每种变换处理每个分段
            for seg_idx, segment in enumerate(segments):
                for transform_type in transform_types:
                    transform = self.transforms[transform_type]
                    
                    # 应用变换
                    result = transform.transform(segment)
                    
                    # 保存图像 (使用分层目录结构)
                    try:
                        saved_path = self.image_saver.save(
                            result, file_info, transform_type, seg_idx
                        )
                        self.stats['images_saved'] += 1
                        
                        if config.VERBOSE and seg_idx == 0:
                            # 仅在第一个分段时打印路径示例
                            print(f"    [{transform_type.value}] 保存到: .../{Path(saved_path).parent.name}/{Path(saved_path).name}")
                    
                    except IOError as e:
                        error_msg = f"保存失败 {file_info.filename} [{transform_type.value}] 分段 {seg_idx}: {e}"
                        print(f"    错误: {error_msg}")
                        self.stats['errors'].append(error_msg)
            
            self.stats['files_processed'] += 1
            self.stats['segments_processed'] += len(segments)
            
            return True
            
        except Exception as e:
            error_msg = f"处理 {file_info.filename} 时出错: {str(e)}"
            print(f"  {error_msg}")
            self.stats['errors'].append(error_msg)
            return False
    
    def process_folder(
        self,
        folder_name: str,
        transform_types: Optional[List[TransformType]] = None
    ) -> None:
        """
        处理特定文件夹中的所有文件。
        
        参数:
            folder_name: 文件夹名称 (例如 '12DriveEndFault')
            transform_types: 要应用的变换列表
        """
        print(f"\n处理文件夹: {folder_name}")
        print("-" * 60)
        
        files = self.data_loader.get_files_by_folder(folder_name)
        
        if len(files) == 0:
            print(f"  在 {folder_name} 中未找到文件")
            return
        
        print(f"  找到 {len(files)} 个文件")
        
        for i, file_info in enumerate(files):
            if config.VERBOSE:
                print(f"  [{i+1}/{len(files)}] {file_info.filename}")
            
            self.process_file(file_info, transform_types)
    
    def process_all(
        self,
        folders: Optional[List[str]] = None,
        transform_types: Optional[List[TransformType]] = None
    ) -> None:
        """
        处理指定文件夹中的所有文件。
        
        参数:
            folders: 要处理的文件夹名称列表。如果为 None，处理所有文件夹。
            transform_types: 要应用的变换列表。如果为 None，应用所有变换。
        """
        if folders is None:
            folders = list(config.DATA_FOLDERS)
        
        print("=" * 60)
        print("CWRU 时频分析流程")
        print("=" * 60)
        print(f"\n配置:")
        print(f"  数据根目录: {config.DATA_ROOT}")
        print(f"  输出根目录: {config.OUTPUT_ROOT}")
        print(f"  目标采样率: {config.TARGET_SR} Hz")
        print(f"  窗口大小: {config.WINDOW_SIZE}")
        print(f"  重叠率: {config.OVERLAP_RATIO * 100:.0f}%")
        print(f"  图像大小: {config.IMAGE_SIZE}")
        print(f"\n输出目录结构:")
        print(f"  outputs/{{transform_type}}/{{folder}}/{{subfolder}}/{{filename}}_seg{{idx}}.png")
        print(f"\n处理 {len(folders)} 个文件夹:")
        for folder in folders:
            print(f"  - {folder}")
        print(f"\n变换方法:")
        transforms_to_use = transform_types or list(TransformType)
        for t in transforms_to_use:
            print(f"  - {t.value}")
        
        start_time = time.time()
        
        # 处理每个文件夹
        for folder in folders:
            self.process_folder(folder, transform_types)
        
        elapsed = time.time() - start_time
        
        # 验证输出结构
        validation = self.path_manager.validate_output_structure()
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("处理摘要")
        print("=" * 60)
        print(f"处理的文件: {self.stats['files_processed']}")
        print(f"生成的分段: {self.stats['segments_processed']}")
        print(f"保存的图像: {self.stats['images_saved']}")
        print(f"错误: {len(self.stats['errors'])}")
        print(f"耗时: {elapsed:.1f}秒")
        print(f"\n输出目录验证:")
        print(f"  已创建: {validation['exists']}")
        if validation['missing']:
            print(f"  缺失: {validation['missing']}")
        
        if self.stats['errors']:
            print(f"\n前 5 个错误:")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="CWRU 时频分析流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用所有变换处理所有数据
  python main.py
  
  # 仅处理特定文件夹
  python main.py --folders 12DriveEndFault NormalBaseline
  
  # 仅使用特定变换
  python main.py --transforms fft cwt
  
  # 组合使用
  python main.py --folders 12DriveEndFault --transforms stft cwt
        """
    )
    
    parser.add_argument(
        '--folders',
        nargs='+',
        choices=list(config.DATA_FOLDERS),
        help='要处理的特定文件夹 (默认: 全部)'
    )
    
    parser.add_argument(
        '--transforms',
        nargs='+',
        choices=[t.value for t in TransformType],
        help='要应用的特定变换 (默认: 全部)'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default=config.DATA_ROOT,
        help=f'CWRU 数据集的根目录 (默认: {config.DATA_ROOT})'
    )
    
    parser.add_argument(
        '--output-root',
        type=str,
        default=config.OUTPUT_ROOT,
        help=f'输出的根目录 (默认: {config.OUTPUT_ROOT})'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='减少输出详细程度'
    )
    
    return parser.parse_args()


def main():
    """主入口点。"""
    args = parse_arguments()
    
    # 从参数更新配置
    config.DATA_ROOT = args.data_root
    config.OUTPUT_ROOT = args.output_root
    config.VERBOSE = not args.quiet
    
    # 解析变换类型
    transform_types = None
    if args.transforms:
        transform_types = [TransformType(t) for t in args.transforms]
    
    # 创建并运行流程
    pipeline = ProcessingPipeline()
    pipeline.process_all(
        folders=args.folders,
        transform_types=transform_types
    )


if __name__ == "__main__":
    main()
