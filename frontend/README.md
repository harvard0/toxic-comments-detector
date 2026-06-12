# AI投毒评论识别系统 - 前端界面

一个专业美观的 Web 界面，用于检测电商评论是否为投毒评论。

## 快速启动

### 1. 安装依赖

如果尚未安装项目依赖，请先安装：

```powershell
cd e:\Learing\Junior\Second\AI\伦理_v2\code
pip install -r requirements.txt
```

前端仅需 Flask：

```powershell
cd e:\Learing\Junior\Second\AI\伦理_v2\frontend
pip install -r requirements.txt
```

### 2. 启动服务

```powershell
cd e:\Learing\Junior\Second\AI\伦理_v2\frontend
python app.py
```

### 3. 访问界面

打开浏览器访问：**http://127.0.0.1:5000**

## 页面说明

### 首页 (`/`)
- 系统介绍与核心功能展示
- 模型性能数据（准确率、精确率、召回率、F1分数）
- 工作流程图示
- 快速入口跳转到检测页面

### 在线检测 (`/detect`)
- 输入待检测的电商评论
- 一键识别评论类型
- 显示检测结果与置信度
- 投毒评论特征提示

### 关于系统 (`/about`)
- 项目简介
- 技术架构说明
- 处理流程展示
- 模型性能指标
- 典型应用场景

## 功能特性

- 深色主题专业界面设计
- 响应式布局，支持移动端
- 实时检测评论内容
- 显示检测置信度
- 自动识别投毒评论特征
- 模型性能数据自动读取

## 技术说明

- **后端**：Flask + PyTorch + Transformers
- **前端**：原生 HTML/CSS/JavaScript
- **模型**：Chinese-RoBERTa 微调模型

## 文件结构

```
frontend/
├── app.py              # Flask 后端服务
├── templates/          # HTML 模板
│   ├── home.html       # 首页
│   ├── detect.html     # 检测页面
│   └── about.html      # 关于页面
├── requirements.txt    # 前端依赖
└── README.md          # 本文件
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页 |
| `/detect` | GET | 检测页面 |
| `/about` | GET | 关于页面 |
| `/api/predict` | POST | 检测评论，接收 `{"text": "评论内容"}` |
| `/api/health` | GET | 健康检查 |

## 注意事项

1. 确保模型文件存在于 `../model/final_model/`
2. 如果是首次运行，模型加载可能需要一些时间
3. 确保端口 5000 未被占用
