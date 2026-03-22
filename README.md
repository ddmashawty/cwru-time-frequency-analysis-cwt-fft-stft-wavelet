# CWRU 时频分析工具包

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于分析 CWRU (Case Western Reserve University) 轴承数据集的综合性工具包，支持多种时频变换方法。

## 功能特性

- **四种时频变换方法**:
  - **FFT** (快速傅里叶变换): 频域分析
  - **STFT** (短时傅里叶变换): 时频谱图
  - **DWT** (离散小波变换): 多分辨率分析
  - **CWT** (连续小波变换): 高分辨率尺度图

- **完整的 CWRU 数据覆盖**: 处理全部 4 个数据文件夹:
  - `12DriveEndFault` - 12kHz 驱动端故障数据
  - `12FanEndFault` - 12kHz 风扇端故障数据
  - `48DriveEndFault` - 48kHz 驱动端故障数据
  - `NormalBaseline` - 正常基准数据

- **分层目录结构**: 输出图像按照 CWRU 原始数据的分类层次保存 (`变换类型/文件夹/转速/文件名`)
- **模块化架构**: 职责清晰分离，便于扩展

## 项目结构

```
cwru_time_frequency_analysis/
├── config/                 # 配置设置
│   ├── __init__.py
│   └── config.py          # 超参数和路径配置
├── core/                   # 核心处理模块
│   ├── __init__.py
│   ├── data_loader.py     # CWRU 数据加载
│   └── preprocessor.py    # 信号预处理
├── transforms/             # 时频变换模块
│   ├── __init__.py
│   ├── base_transform.py  # 抽象基类
│   ├── fft_transform.py   # FFT 实现
│   ├── stft_transform.py  # STFT 实现
│   ├── wavelet_transform.py  # DWT 实现
│   └── cwt_transform.py   # CWT 实现
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── io_utils.py        # 文件 I/O 操作
│   └── path_manager.py    # 输出路径管理 (分层目录结构)
├── outputs/                # 生成的图像 (自动创建)
│   ├── fft/               # FFT 变换输出
│   │   ├── 12DriveEndFault/
│   │   │   ├── 1730/
│   │   │   │   └── *.png
│   │   │   └── ...
│   │   └── ...
│   ├── stft/              # STFT 变换输出
│   ├── wavelet/           # 小波变换输出
│   └── cwt/               # CWT 变换输出
├── main.py                # 主入口程序
├── requirements.txt       # Python 依赖
└── README.md             # 本文件
```

## 安装配置

### 环境要求

- Python 3.8 或更高版本
- CWRU 轴承数据集 (从 [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter) 下载)

### 安装步骤

1. 克隆本仓库:
```bash
git clone <仓库地址>
cd cwru_time_frequency_analysis
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 配置数据路径:
   - 编辑 `config/config.py` 并设置 `DATA_ROOT` 指向你的 CWRU 数据集位置
   - 或使用命令行参数 `--data-root` 指定路径

## 使用方法

### 基础用法

使用所有变换处理所有数据:

```bash
python main.py
```

### 处理指定文件夹

```bash
python main.py --folders 12DriveEndFault NormalBaseline
```

### 使用指定变换

```bash
python main.py --transforms fft cwt
```

### 自定义数据路径

```bash
python main.py --data-root "/path/to/CWRU" --output-root "/path/to/outputs"
```

### 静默模式

减少输出信息:

```bash
python main.py --quiet
```

## 配置参数

所有超参数都集中在 `config/config.py` 中管理:

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `WINDOW_SIZE` | 4096 | 分帧长度 (采样点) |
| `OVERLAP_RATIO` | 0.75 | 帧间重叠率 |
| `STRIDE` | 1024 | 步长 (4096 * 0.25) |
| `TARGET_SR` | 12000 | 目标采样率 (Hz) |
| `IMAGE_SIZE` | (64, 64) | 输出图像尺寸 |
| `WAVELET_NAME` | "db4" | 离散小波类型 |
| `CWT_WAVELET` | "cmor1.0-2" | 连续小波类型 (复Morlet, B=1.0, C=2) |
| `CWT_SCALES` | 256 | CWT尺度数量 |
| `CWT_DT` | 1/12000 | CWT采样周期 |

修改这些值可以自定义处理流程。

## 输出格式

每种变换都会生成 PNG 图像，按照与 CWRU 原始数据相同的分类层次保存:

```
outputs/
├── fft/
│   ├── 12DriveEndFault/
│   │   ├── 1730/
│   │   │   ├── 0.007-Ball_seg0000.png
│   │   │   ├── 0.007-Ball_seg0001.png
│   │   │   └── ...
│   │   ├── 1750/
│   │   └── ...
│   ├── 12FanEndFault/
│   │   ├── 1730/
│   │   └── ...
│   ├── 48DriveEndFault/
│   └── NormalBaseline/
│       ├── 1730/
│       │   ├── Normal_seg0000.png
│       │   └── ...
│       ├── 1750/
│       └── ...
├── stft/
│   └── ... (相同结构)
├── wavelet/
│   └── ... (相同结构)
└── cwt/
    └── ... (相同结构)
```

**目录结构说明**:
- 第一级: 变换类型 (`fft`, `stft`, `wavelet`, `cwt`)
- 第二级: 数据来源文件夹 (`12DriveEndFault`, `NormalBaseline` 等)
- 第三级: 转速子文件夹 (`1730`, `1750`, `1772`, `1797`)
- 文件名: `{原始文件名}_seg{分段索引:04d}.png`

## 数据格式

CWRU 数据集应按以下结构组织:

```
CWRU/
├── 12DriveEndFault/
│   ├── 1730/
│   │   ├── 0.007-Ball.mat
│   │   ├── 0.007-InnerRace.mat
│   │   └── ...
│   ├── 1750/
│   └── ...
├── 12FanEndFault/
│   └── ...
├── 48DriveEndFault/
│   └── ...
└── NormalBaseline/
    └── ...
```

## 算法详情

### 预处理流程

1. **下采样 (Downsampling)**: 48kHz 信号通过抗混叠滤波下采样到 12kHz
2. **分帧 (Segmentation)**: 信号被分割为重叠窗口 (4096 采样点, 75% 重叠, 步长1024)
3. **归一化 (Normalization)**: 每个片段归一化到 [0, 1] 范围

### 变换方法

#### FFT (快速傅里叶变换)
- 计算每个片段的幅度谱
- 通过重塑频谱创建二维表示
- 适用于: 整体频率内容分析

#### STFT (短时傅里叶变换)
- 使用重叠的汉明窗
- 对数缩放的幅度以获得更好的动态范围
- 适用于: 时变频率分析

#### 离散小波变换 (DWT)
- 使用 Daubechies 小波进行多级分解
- 同时捕获高频瞬态和低频趋势
- 适用于: 多分辨率分析

#### 连续小波变换 (CWT)
- 高分辨率时频表示
- 使用 Morlet 小波以获得最优的时频局部化
- 适用于: 瞬态检测和详细分析

## 扩展工具包

### 添加新变换

1. 在 `transforms/` 目录下创建新文件:

```python
# transforms/my_transform.py
from .base_transform import BaseTransform
import numpy as np

class MyTransform(BaseTransform):
    def transform(self, signal: np.ndarray) -> np.ndarray:
        # 在这里实现你的变换逻辑
        result = ...
        return self.resize(result)
    
    def get_name(self) -> str:
        return "MyTransform"
```

2. 在 `transforms/__init__.py` 中注册:

```python
from .my_transform import MyTransform
```

3. 在 `config/config.py` 的 `TransformType` 枚举中添加

## 故障排除

### "Data root not found" 错误
- 验证 `config.py` 中的 `DATA_ROOT` 指向正确的 CWRU 数据集位置
- 确保文件夹包含 4 个子目录 (12DriveEndFault 等)

### 内存不足
- 减小配置中的 `WINDOW_SIZE`
- 使用 `--folders` 参数一次处理更少的文件夹

### 缺少依赖
```bash
pip install --upgrade -r requirements.txt
```

## 开源协议

本项目采用 MIT 协议开源 - 详情请参见 LICENSE 文件。

## 致谢

- CWRU Bearing Data Center 提供数据集
- PyWavelets 团队提供小波变换实现

## 引用

如果你在研究中使用本工具包，请引用:

```bibtex
@software{cwru_time_frequency_analysis,
  title = {CWRU Time-Frequency Analysis Toolkit},
  year = {2024},
  url = {<仓库地址>}
}
```

## 贡献指南

欢迎贡献代码! 请随时提交 Pull Request。

1. Fork 本仓库
2. 创建你的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request
