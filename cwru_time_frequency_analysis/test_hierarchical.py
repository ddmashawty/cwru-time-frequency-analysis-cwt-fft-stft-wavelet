"""
分层输出目录结构测试

验证新的输出路径管理器是否正确地保持了与 CWRU 原始数据相同的分类层次。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import tempfile
import shutil


def test_file_info_with_relative_path():
    """测试 FileInfo 是否正确提取相对路径。"""
    print("\n[1/5] 测试 FileInfo 相对路径...")
    
    from core.data_loader import CWRUDataLoader
    
    loader = CWRUDataLoader()
    files = loader.get_all_files()
    
    if len(files) == 0:
        print("  [跳过] 未找到数据文件")
        return True
    
    # 验证每个文件的 relative_path
    for file_info in files[:5]:  # 只检查前5个
        print(f"  文件: {file_info.filename}")
        print(f"    folder: {file_info.folder}")
        print(f"    subfolder: {file_info.subfolder}")
        print(f"    relative_path: {file_info.relative_path}")
        
        # 验证路径一致性
        expected_path = f"{file_info.folder}/{file_info.subfolder}" if file_info.subfolder else file_info.folder
        assert file_info.relative_path == expected_path, f"路径不匹配: {file_info.relative_path} != {expected_path}"
    
    print("  [OK] 相对路径提取正确")
    return True


def test_output_path_manager():
    """测试 OutputPathManager 的路径生成。"""
    print("\n[2/5] 测试 OutputPathManager...")
    
    from utils.path_manager import OutputPathManager, OutputPath
    from core.data_loader import FileInfo
    from config.config import TransformType
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputPathManager(tmpdir)
        
        # 创建模拟的 FileInfo
        file_info = FileInfo(
            filepath="/data/CWRU/12DriveEndFault/1730/0.007-Ball.mat",
            filename="0.007-Ball.mat",
            folder="12DriveEndFault",
            subfolder="1730",
            relative_path="12DriveEndFault/1730",
            fault_type="Ball",
            fault_size=7,
            rpm=1730,
            original_sr=12000,
            is_normal=False,
            label=1
        )
        
        # 测试路径生成
        output_path = manager.get_output_path(file_info, TransformType.FFT, segment_idx=5)
        
        print(f"  输出路径: {output_path.full_path}")
        print(f"  变换目录: {output_path.transform_dir}")
        print(f"  分类目录: {output_path.category_dir}")
        
        # 验证路径结构
        assert "fft" in output_path.full_path
        assert "12DriveEndFault" in output_path.full_path
        assert "1730" in output_path.full_path
        assert "0.007-Ball_seg0005.png" in output_path.full_path
        
        # 验证目录创建
        output_path.ensure_directories()
        assert Path(output_path.category_dir).exists()
        
        print("  [OK] 路径生成正确")
    
    return True


def test_hierarchical_structure():
    """测试完整的分层目录结构创建。"""
    print("\n[3/5] 测试分层目录结构...")
    
    from utils.path_manager import OutputPathManager
    from config.config import TransformType
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputPathManager(tmpdir)
        
        # 创建所有变换目录
        manager.create_transform_directories()
        
        # 验证结构
        validation = manager.validate_output_structure()
        
        print(f"  已创建目录: {validation['exists']}")
        print(f"  缺失目录: {validation['missing']}")
        
        assert len(validation['exists']) == 4, "应该创建 4 个变换目录"
        assert len(validation['missing']) == 0, "不应该有缺失目录"
        
        print("  [OK] 目录结构创建正确")
    
    return True


def test_image_saver():
    """测试 HierarchicalImageSaver 的保存功能。"""
    print("\n[4/5] 测试分层图像保存...")
    
    import numpy as np
    from utils.path_manager import HierarchicalImageSaver
    from core.data_loader import FileInfo
    from config.config import TransformType
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = HierarchicalImageSaver(tmpdir)
        
        # 创建模拟的 FileInfo
        file_info = FileInfo(
            filepath="/data/CWRU/NormalBaseline/1797/Normal.mat",
            filename="Normal.mat",
            folder="NormalBaseline",
            subfolder="1797",
            relative_path="NormalBaseline/1797",
            fault_type="Normal",
            fault_size=0,
            rpm=1797,
            original_sr=48000,
            is_normal=True,
            label=0
        )
        
        # 创建测试图像
        test_image = np.random.rand(64, 64).astype(np.float32)
        
        # 保存图像
        saved_path = saver.save(test_image, file_info, TransformType.CWT, segment_idx=0)
        
        print(f"  保存路径: {saved_path}")
        
        # 验证文件存在
        assert Path(saved_path).exists(), "保存的文件应该存在"
        
        # 验证路径结构
        path_parts = Path(saved_path).parts
        assert "cwt" in path_parts
        assert "NormalBaseline" in path_parts
        assert "1797" in path_parts
        
        print("  [OK] 图像保存到正确位置")
    
    return True


def test_directory_structure_matches_cwru():
    """测试输出目录结构与 CWRU 原始结构匹配。"""
    print("\n[5/5] 测试目录结构与 CWRU 匹配...")
    
    from utils.path_manager import OutputPathManager
    from core.data_loader import CWRUDataLoader, FileInfo
    from config.config import TransformType
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OutputPathManager(tmpdir)
        loader = CWRUDataLoader()
        
        files = loader.get_all_files()
        if len(files) == 0:
            print("  [跳过] 未找到数据文件")
            return True
        
        # 选择第一个文件测试
        file_info = files[0]
        
        print(f"  测试文件: {file_info.filename}")
        print(f"  原始相对路径: {file_info.relative_path}")
        
        # 为每种变换生成路径
        for transform_type in TransformType:
            output_path = manager.get_output_path(file_info, transform_type, 0)
            
            # 验证路径包含原始分类结构
            assert file_info.folder in output_path.full_path
            assert file_info.subfolder in output_path.full_path
            
            print(f"  [{transform_type.value}] -> .../{file_info.relative_path}/")
        
        print("  [OK] 输出结构与 CWRU 分类一致")
    
    return True


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("分层输出目录结构测试")
    print("=" * 60)
    
    tests = [
        test_file_info_with_relative_path,
        test_output_path_manager,
        test_hierarchical_structure,
        test_image_saver,
        test_directory_structure_matches_cwru,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [失败] {e}")
            import traceback
            traceback.print_exc()
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
