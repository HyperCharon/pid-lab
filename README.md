# pid-lab

PID控制仿真与调试平台。本地部署的Web应用，集成多种控制算法、被控对象模型、自动整定方法和频域分析工具。

## 安装与运行

```bash
# 克隆仓库
git clone https://github.com/HyperCharon/pid-lab.git
cd pid-lab

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

启动后在浏览器打开 http://localhost:8000 即可使用。

## 功能

### 控制算法

- 三种PID形式：并行式、ISA标准式、增量式
- 积分抗饱和（限幅法/反馈抑制法）
- 微分项一阶低通滤波
- 设定值加权（比例微分先行）
- 算法/参数切换无扰过渡
- 串级控制、前馈控制、Smith预估器

### 被控对象

9种内置模型：

| 模型 | 传递函数 | 说明 |
|------|----------|------|
| 电机调速 | K/(τs+1) | 一阶惯性 |
| 温度控制 | Ke^(-θs)/(Ts+1) | 带纯滞后 |
| 液位控制 | K/(s(Ts+1)) | 无自衡过程 |
| 伺服定位 | 三阶柔性 | 含振荡模态 |
| 二阶系统 | ωn²/(s²+2ζωns+ωn²) | 可调阻尼比 |
| 耦合双容 | 两环节耦合 | 双容水箱 |
| 二阶+滞后 | SOPDT | 工业加热过程 |
| 纯积分 | K/s | 液位/角度 |
| 自定义 | 用户输入 | 任意传递函数 |

### 自动整定

- Ziegler-Nichols阶跃响应法
- Cohen-Coon整定法
- 继电反馈法
- Lambda整定法
- ISE/IAE/ITAE参数优化（Nelder-Mead）

### 系统辨识

支持从实测CSV数据辨识模型参数：

- 一阶加滞后辨识（K, L, T）+ 拟合度R²
- ARX模型辨识任意阶传递函数
- REST API接口供外部调用

### 频域分析

- Bode图（幅频+相频），自动计算增益裕度和相位裕度
- Nyquist图，标注(-1,j0)点
- 支持含纯滞后系统（Pade近似）

### 数据管理

- SQLite存储实验记录
- 预设配置保存/加载
- CSV导入/导出
- Markdown格式实验报告导出

## 使用

### 基本仿真

1. 选择被控对象模型和PID算法形式
2. 点击 **RUN** 开始仿真
3. 拖动Kp、Ki、Kd滑条，观察波形变化
4. 右侧面板显示实时性能指标

### 自动整定

点击 **整定** 按钮，选择方法后执行。支持一键应用整定结果到当前参数。

### 系统辨识

1. 准备CSV文件（第一列时间，第二列输入，第三列输出）
2. 点击 **导入** 按钮选择文件
3. 自动辨识K、L、T并显示拟合度

### 自定义传递函数

在模型菜单选择"自定义TF"，输入分子分母系数（s降幂，逗号分隔）。

例：分子 `1,0`，分母 `1,2,1` 表示 G(s) = s/(s²+2s+1)。

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

## REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/ws` | WebSocket | 仿真数据实时通信 |
| `/api/upload-csv` | POST | 上传CSV进行系统辨识 |
| `/api/identify-tf` | POST | JSON数据辨识传递函数 |
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
