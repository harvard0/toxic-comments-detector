# AI 投毒评论识别系统

基于 BERT 的电商评论投毒检测模型。

## 目录结构

```
伦理_v2/
├── model/                              # 模型目录
│   ├── chinese-roberta-wwm-ext/        # 预训练模型
│   └── final_model/                    # 训练后的模型
├── code/                               # 代码目录
│   ├── data/                           # 数据目录
│   ├── eval_results/                   # 评估结果
│   ├── scripts/                        # 数据处理脚本
│   ├── main/                           # 核心流程脚本
│   └── README.md                       # 详细使用说明
├── frontend/                           # Flask Web界面
│   ├── app.py                          # Flask Web服务
│   ├── requirements.txt                # 前端依赖
│   └── templates/                      # HTML模板
├── server/                             # 后端API服务
│   ├── app.py                          # Streamlit界面
│   ├── inference.py                    # 模型推理模块
│   └── run_api.py                      # Flask API启动脚本
└── README.md
```

## 快速开始

详细使用说明请查看 [code/README.md](code/README.md)。

### 环境准备

```powershell
cd d:\学习\大三下\伦理\code
pip install -r requirements.txt
```

### 完整工作流程

```powershell
# 1. 进入脚本目录
cd d:\学习\大三下\伦理\code\scripts

# 2. 生成数据（可选）
python data_generator.py

# 3. 清洗数据
python data_cleaner.py

# 4. 查看数据统计
python multi_dataset_stats.py

# 5. 训练模型
cd ..\main
python train.py --data ../data/train_toxic_comments_cleaned.jsonl --valid ../data/valid_toxic_comments_cleaned.jsonl --test ../data/test_toxic_comments_cleaned.jsonl

# 6. 评估模型
python eval.py --data ../data/test_toxic_comments_cleaned.jsonl

# 7. 使用模型预测
python inference.py --text "这是一条测试评论"
```

## 模型位置

- **预训练模型**：`model/chinese-roberta-wwm-ext/`
- **训练后模型**：`model/final_model/`

## Web界面使用

### 方式一：Flask Web界面（推荐）

`frontend/app.py` 提供完整的Web页面，包含首页、检测页和关于页，可独立运行。

```powershell
# 1. 安装依赖
cd d:\学习\大三下\伦理_v2\frontend
pip install flask torch transformers

# 2. 启动服务
python app.py
```

启动后访问 <http://127.0.0.1:5000>

**页面功能：**

| 路由        | 功能             |
| --------- | -------------- |
| `/`       | 首页 - 展示模型性能指标  |
| `/detect` | 检测页 - 输入评论进行检测 |
| `/about`  | 关于页 - 系统介绍     |

**API接口：**

| 接口             | 方法   | 说明                         |
| -------------- | ---- | -------------------------- |
| `/api/predict` | POST | 预测评论，参数 `{"text": "评论内容"}` |
| `/api/health`  | GET  | 健康检查                       |

**特点：**

- 路径配置正确，无需修改
- 启动时自动加载微调模型
- 独立运行，不需要额外后端服务

### 方式二：Streamlit界面

`server/app.py` 是 Streamlit 交互式界面，会自动启动后端API服务。

```powershell
# 1. 安装依赖
pip install streamlit requests

# 2. 启动前端（会自动启动后端）
cd d:\学习\大三下\伦理_v2\server
streamlit run app.py
```

启动后访问 <http://localhost:8501>

**特点：**

- 自动启动后端API服务
- 支持单条检测和批量演示
- 适合快速演示

## API服务使用（前端演示）

### 启动API服务

```powershell
cd d:\学习\大三下\伦理_v2\server
python run_api.py
```

API服务运行在 <http://localhost:5000>

### API接口说明

| 接口               | 方法   | 说明     |
| ---------------- | ---- | ------ |
| `/predict`       | POST | 单条评论预测 |
| `/batch_predict` | POST | 批量评论预测 |
| `/health`        | GET  | 健康检查   |

### 调用示例

```python
import requests

# 单条预测
response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "这是一条测试评论"}
)
print(response.json())

# 批量预测
response = requests.post(
    "http://localhost:5000/batch_predict",
    json={"texts": ["评论1", "评论2", "评论3"]}
)
print(response.json())

# 健康检查
response = requests.get("http://localhost:5000/health")
print(response.json())
```

### 命令行推理

```powershell
cd d:\学习\大三下\伦理_v2\server

# 单条文本预测
python inference.py --text "这是一条测试评论"

# 使用原预训练模型（用于系统验证）
python inference.py --text "这是一条测试评论" --use_base_model

# 运行演示模式
python inference.py --demo
```

