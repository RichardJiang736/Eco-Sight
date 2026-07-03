# Eco-Sight：自动驾驶智能视觉感知系统

<p align="left">
    <picture>
        <img src="./assets/microsoft-logo.svg.png" width="40%">
    </picture>
</p>

**Eco-sight** 是一个自动驾驶的智能视觉感知系统，通过深度学习和自适应帧率调节技术，实现高效的实时目标检测。系统动态调整推理模型和帧率，在保证检测精度的同时降低能源消耗，为绿色智能交通做出贡献。

## 目录

- [项目特性](#项目特性)
- [快速开始](#快速开始)
- [数据集与模型](#数据集与模型)
- [系统设计](#系统设计)
- [性能指标](#性能指标)
- [项目成员](#项目成员)

---

## 项目特性

- **自适应模型切换**：根据检测对象数量动态调整推理模型，平衡精度与能耗
- **实时目标检测**：基于 YOLOv10，响应时间 20-25ms，满足自动驾驶需求
- **大规模数据集**：采用 nuScenes 数据集，包含 11,584 条场景数据
- **节能设计**：低能耗模式下帧率自动降至 5 FPS，节省系统资源
- **完整评估框架**：包含精度、延迟和功耗的综合基准测试

---

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 下载模型权重
python -c "from ultralytics import YOLO; YOLO('yolov10n.pt'); YOLO('yolov10s.pt'); YOLO('yolov10m.pt')"
mkdir -p models && mv yolov10*.pt models/
```

### 2. 推理（自适应模型切换）

```bash
python -m eco_sight.cli run --video path/to/driving_video.mp4
```

系统根据检测对象数量自动调整模型：

| 检测对象数 | 使用模型 | 帧率 | 能耗 |
|-----------|---------|------|------|
| 0 | yolov10n | 5 FPS | 低 |
| 1–3 | yolov10m | 30 FPS | 高 |
| >3 | yolov10s | 5 FPS | 中 |

### 3. 基准测试

```bash
# 完整评估（精度、延迟、功耗）
python -m eco_sight.cli benchmark --video path/to/video.mp4 --output-dir results/

# 投影到目标设备性能
python -m eco_sight.cli normalize --results-dir results/ --target all

# 生成评估报告
python -m eco_sight.cli report --results-dir results/ --output report.md
```

---

## 数据集与模型

### 数据集选择

| 数据集  | 优点 | 缺点 | 最终选择 |
|---------|------|------|---------|
| KITTI   | 历史悠久，数据多样 | 数据量小，场景单一 | ❌ |
| Waymo   | 数据量大，场景多样 | 获取限制，隐私问题 | ❌ |
| nuScenes | 传感器全面，标注详细 | 数据量相对较小 | ✅ |

**最终采用 nuScenes 数据集**，结合雷达云图和摄像机数据，通过预处理获得训练、验证、测试集。

<div align="center">
    <img src="./assets/nuScenes.png" width="60%" alt="nuScenes数据集">
</div>

### 数据预处理

- **原始数据**：11,584 条
- **处理后**：8,763 条
- **工具**：Python、labelimg、X-AnyLabeling、YOLOv10x
- **流程**：手动标注 + 自动识别 + 人工筛选

<div align="center">
    <img src="./assets/preprocessing.png" width="60%" alt="预处理流程">
</div>

### 模型训练

基于 **YOLOv10** 架构，训练三个不同规模的模型：

| 模型 | 参数量 | 用途 |
|-----|--------|------|
| yolov10n（Nano） | 2.30M | 低能耗推理 |
| yolov10s（Small） | 7.25M | 中等能耗推理 |
| yolov10m（Medium） | 15.36M | 高精度推理 |

训练配置：200 轮次，批大小 16，图像尺寸 640×640

---

## 系统设计

### 智能帧率调节

系统根据实时检测结果动态调整推理模型和帧率：

**核心逻辑**：
- 检测到 **0 个对象** → 低能耗模式（yolov10n，5 FPS）
- 检测到 **1-3 个对象** → 高精度模式（yolov10m，30 FPS）
- 检测到 **>3 个对象** → 均衡模式（yolov10s，5 FPS）

**效果**：
- 在保持检测精度的前提下，显著降低平均功耗
- 关键场景下快速响应，无关键场景下节省资源

<div align="center">
    <img src="./assets/recognition_system.png" width="60%" alt="帧率调节系统">
</div>

---

## 性能指标

### 检测精度（200 轮次微调后）

| 模型 | mAP50 | mAP50-95 | 精确率 | 召回率 |
|-----|-------|----------|--------|--------|
| yolov10n | 0.348 | 0.217 | 0.542 | 0.320 |
| yolov10s | 0.451 | 0.265 | 0.625 | 0.408 |
| yolov10m | 0.508 | 0.299 | 0.570 | 0.473 |

> 在 nuScenes 衍生数据集上微调得出

### 推理延迟与吞吐量（M1 Pro 基准）

| 指标 | nano | small | medium |
|-----|------|-------|--------|
| 平均延迟 | 71.6 ms | 148.7 ms | 156.5 ms |
| P95 延迟 | 75.1 ms | 165.7 ms | 166.4 ms |
| 有效 FPS | 14.0 | 6.7 | 6.4 |

### 模型切换延迟

| 切换方向 | 延迟 |
|---------|------|
| nano → small | 150.8 ms |
| nano → medium | 158.6 ms |
| small → medium | 155.9 ms |

> 切换成本仅为目标模型的推理时间（无重新加载开销）

### 目标设备投影性能

| 设备 | nano FPS | small FPS | medium FPS | 功耗 |
|-----|----------|-----------|------------|------|
| M1 Pro（基准） | 14.0 | 6.7 | 6.4 | ~12W |
| Jetson Orin Nano | 3.8 | 1.9 | 1.5 | ~10W |
| Jetson Orin Nano + TensorRT | **27.9** | **13.5** | **9.1** | ~10W |
| Jetson Xavier NX + TensorRT | 17.5 | 7.5 | 5.8 | ~12W |

<div align="center">
    <img src="./assets/result_comparison.jpg" width="60%" alt="检测结果对比">
</div>

---

## 主要成果

- 实现了自动驾驶场景下实时高效的目标检测  
- 平均响应时间 20-25ms，达到行业标准  
- 通过自适应帧率调节，显著降低能源消耗  
- 建立完整的基准测试和评估框架  
- 成功在多种目标设备上进行性能投影

---

## 项目成员

- 许焕桦（组长）
- 慕容元
- 钟子诚
- 刘桸泽
- 江创泓（组长）
