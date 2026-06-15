# pid-lab: PID控制仿真与调试平台

基于Web的PID控制系统仿真平台，集成多种控制算法、被控对象模型、自动整定方法和频域分析工具。

## 功能概述

### 控制算法

- **三种PID形式**：并行式、ISA标准式、增量式
- **工程特性**：积分抗饱和（限幅/反算）、微分低通滤波、设定值加权、无扰切换
- **高级结构**：串级控制、前馈控制、Smith预估器

### 被控对象

8种内置模型（一阶惯性、带纯滞后、积分环节、三阶柔性等），支持自定义传递函数输入。

### 系统辨识

从实测CSV数据辨识一阶加滞后模型（K, L, T），支持ARX模型辨识任意阶传递函数。

### 自动整定

- Ziegler-Nichols阶跃响应法
- Cohen-Coon整定法
- 继电反馈法
- Lambda整定法
- 基于Nelder-Mead的ISE/IAE/ITAE参数优化

### 频域分析

Bode图（幅频+相频，自动计算增益/相位裕度）、Nyquist图。

### 数据管理

SQLite实验记录存储、预设配置管理、CSV导入/导出、Markdown报告导出。

## 快速开始

```bash
pip install fastapi uvicorn scipy numpy
python main.py
```

浏览器访问 `http://localhost:8000`。

## 使用说明

### 基本仿真

1. 选择被控对象模型和PID算法形式
2. 点击 **RUN** 开始仿真
3. 调节Kp、Ki、Kd滑条，观察波形响应
4. 查看性能指标（超调量、调节时间、IAE、ISE）

### 自动整定

点击 **整定** 按钮，选择整定方法：

- 对无滞后系统推荐使用继电反馈法或Lambda整定
- 对带滞后系统推荐使用Z-N阶跃响应法或Cohen-Coon
- 需要精确优化时使用参数优化（选择ISE/IAE/ITAE目标）

### 系统辨识

1. 准备CSV文件：第一列时间，第二列输入，第三列输出
2. 点击 **导入** 按钮选择文件
3. 系统自动辨识K、L、T并显示拟合度R²

### 自定义传递函数

在模型下拉菜单选择"自定义TF"，输入分子分母系数（s降幂，逗号分隔）。

例：分子 `1,0`，分母 `1,2,1` 表示 G(s) = s/(s²+2s+1)。

### 频域分析

点击 **频域** 按钮显示Bode图和Nyquist图。增益裕度和相位裕度自动标注。

### 快捷键

| 按键 | 功能 |
|------|------|
| R | 重置仿真 |
| Space | 暂停/继续 |
| D | 注入阶跃扰动 |
| C | 切换对比模式 |
| F | 切换频域分析 |
| S | 保存预设 |
| 1-9 | 切换模型 |

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/ws` | WebSocket | 仿真数据实时通信 |
| `/api/upload-csv` | POST | 上传CSV进行系统辨识 |
| `/api/identify-tf` | POST | 从JSON数据辨识传递函数 |
| `/api/experiments` | GET/POST | 实验记录管理 |
| `/api/export-report/{id}` | GET | 导出实验报告 |

## 项目结构

```
pid-lab/
├── main.py          服务端（FastAPI + WebSocket + SQLite）
├── controller.py    控制器（PID + 串级 + 前馈 + Smith预估器）
├── models.py        模型（内置 + 自定义TF + 系统辨识）
├── autotune.py      整定（经典方法 + 参数优化）
├── analysis.py      频域（Bode + Nyquist + 裕度计算）
├── static/
│   └── index.html   前端（Canvas示波器 + 参数面板）
├── experiments.db   SQLite数据库（运行时生成）
├── presets/         预设配置（运行时生成）
└── README.md
```

## 依赖

- Python >= 3.10
- fastapi >= 0.115
- uvicorn >= 0.48
- scipy >= 1.14
- numpy >= 2.0

## 许可

MIT License
