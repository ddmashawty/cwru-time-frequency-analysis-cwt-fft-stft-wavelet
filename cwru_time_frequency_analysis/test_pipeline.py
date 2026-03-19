"""
CWRU 时频分析流程的测试脚本。

无需处理完整数据集的快速验证。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_imports():
    """测试所有模块导入。"""
    print("[1/5] 测试导入...")
    
    from config.config import Config, TransformType
    from core.data_loader import CWRUDataLoader, FileInfo
    from core.preprocessor import SignalPreprocessor
    from transforms import FFTTransform, STFTTransform, WaveletTransform, CWTTransform
    from utils.io_utils import save_image, create_output_dirs
    
    print("  [OK] 所有模块导入成功")
    return True


def test_config():
    """测试配置。"""
    print("\n[2/5] 测试配置...")
    
    from config.config import config, TransformType
    
    assert config.WINDOW_SIZE == 512
    assert config.TARGET_SR == 12000
    assert len(TransformType) == 4
    
    print(f"  [OK] 配置已加载")
    print(f"       窗口大小: {config.WINDOW_SIZE}")
    print(f"       目标采样率: {config.TARGET_SR} Hz")
    print(f"       变换: {[t.value for t in TransformType]}")
    return True


def test_transforms():
    """测试变换实现。"""
    print("\n[3/5] 测试变换...")
    
    from transforms import FFTTransform, STFTTransform, WaveletTransform, CWTTransform
    from config.config import config
    
    # 生成测试信号 (12kHz 下 1 秒)
    np.random.seed(42)
    signal = np.random.randn(config.WINDOW_SIZE)
    
    transforms = {
        'FFT': FFTTransform(),
        'STFT': STFTTransform(),
        'Wavelet': WaveletTransform(),
        'CWT': CWTTransform(),
    }
    
    for name, transform in transforms.items():
        result = transform.transform(signal)
        assert result.shape == config.IMAGE_SIZE, f"{name}: 输出形状错误"
        assert result.min() >= 0 and result.max() <= 1, f"{name}: 未归一化"
        print(f"  [OK] {name}: {result.shape}, 范围 [{result.min():.3f}, {result.max():.3f}]")
    
    return True


def test_preprocessing():
    """测试信号预处理。"""
    print("\n[4/5] 测试预处理...")
    
    from core.preprocessor import SignalPreprocessor
    
    preprocessor = SignalPreprocessor()
    
    # 测试信号 (48kHz 下 2 秒)
    original_sr = 48000
    signal = np.random.randn(original_sr * 2)
    
    segments = preprocessor.process(signal, original_sr)
    
    expected_segments = (len(signal) // 4 - 512) // 256 + 1
    
    print(f"  [OK] 生成了 {len(segments)} 个分段")
    print(f"       输入: {len(signal)} 个样本 @ {original_sr/1000:.0f}kHz")
    print(f"       输出: {len(segments)} 个分段 x {len(segments[0])} 个样本")
    
    assert len(segments) > 0
    assert all(len(s) == 512 for s in segments)
    
    return True


def test_data_loader():
    """测试数据加载器 (如果数据存在)。"""
    print("\n[5/5] 测试数据加载器...")
    
    from core.data_loader import CWRUDataLoader
    from config.config import config
    import os
    
    if not os.path.exists(config.DATA_ROOT):
        print(f"  [跳过] 在 {config.DATA_ROOT} 未找到数据")
        print(f"         在 config.py 中设置正确路径以运行完整测试")
        return True
    
    try:
        loader = CWRUDataLoader()
        files = loader.get_all_files()
        
        print(f"  [OK] 找到 {len(files)} 个文件")
        
        if len(files) > 0:
            # 测试加载第一个文件
            file_info = files[0]
            signal = loader.load_signal(file_info)
            
            print(f"  [OK] 已加载: {file_info.filename}")
            print(f"       类型: {file_info.fault_type}")
            print(f"       RPM: {file_info.rpm}")
            print(f"       信号长度: {len(signal)}")
        
        return True
        
    except Exception as e:
        print(f"  [错误] {e}")
        return False


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("CWRU 时频分析 - 测试套件")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_transforms,
        test_preprocessing,
        test_data_loader,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [失败] {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("所有测试通过!")
        return 0
    else:
        print(f"测试失败: {sum(not r for r in results)}/{len(results)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
