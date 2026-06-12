# AI 投毒评论识别系统

基于 BERT 的电商评论投毒检测模型。

## 目录结构

```
toxic-comments-detector/
├── model/                              # 模型目录
│   ├── chinese-roberta-wwm-ext/        # 预训练模型
│   └── final_model/                    # 训练后的模型
├── code/                               # 代码目录
│   ├── data/                           # 数据目录
│   │   ├── train_toxic_comments.jsonl
│   │   ├── valid_toxic_comments.jsonl
│   │   ├── test_toxic_comments.jsonl
│   │   └── *_cleaned.jsonl / *_garbage.jsonl
│   ├── eval_results/                   # 评估结果
│   ├── scripts/                        # 数据处理脚本
│   │   ├── data_generator.py
│   │   ├── data_cleaner.py
│   │   └── multi_dataset_stats.py
│   ├── main/                           # 核心流程脚本
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── inference.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 预训练模型

预训练模型已放置在：
```
model\chinese-roberta-wwm-ext\
```

目录内容包含：
- `config.json`
- `pytorch_model.bin`
- `vocab.txt`
- `tokenizer_config.json`
- `special_tokens_map.json`

### 3. 配置 API（可选，用于数据生成）

如需使用 `data_generator.py` 生成数据，请在 `code` 目录下创建 `.env` 文件：

```
DataGenerator_API_Key=your_api_key
DataGenerator_API_URL=your_api_url
DataGenerator_API_Model=your_model_name
```

---

## 脚本使用说明

### 一、数据处理脚本 (`scripts/`)

#### 1. 数据生成器 - `data_generator.py`

使用大模型 API 批量生成电商评论数据集。

```bash
python data_generator.py
```

**输出**：`data/train_toxic_comments.jsonl`

**配置项**（在脚本中修改）：
- `TARGET_TOTAL_PER_LABEL`：每类生成数量（默认 240 条）
- `BATCH_SIZE`：单次 API 请求生成数量（默认 10）
- `CONCURRENCY_LIMIT`：并发限制（默认 200）

---

#### 2. 数据清洗器 - `data_cleaner.py`

清洗数据集，去除 AI 痕迹、重复数据、异常长度数据。

```powershell
cd d:\学习\大三下\伦理\code\scripts
python data_cleaner.py
```

**输入文件**：
- `data/train_toxic_comments.jsonl`
- `data/valid_toxic_comments.jsonl`
- `data/test_toxic_comments.jsonl`

**输出文件**：
- `data/*_cleaned.jsonl` - 清洗后的有效数据
- `data/*_garbage.jsonl` - 被剔除的无效数据

---

#### 3. 数据集统计 - `multi_dataset_stats.py`

统计各数据集的规模、正负例分布、类别分布。

```bash
python multi_dataset_stats.py
```

**输出**：终端打印统计报告

---

### 二、核心流程脚本 (`main/`)

#### 1. 模型训练 - `train.py`

训练投毒评论识别模型。

```bash
python train.py --data ../data/train_toxic_comments_cleaned.jsonl --valid ../data/valid_toxic_comments_cleaned.jsonl --test ../data/test_toxic_comments_cleaned.jsonl
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 训练数据路径 | - |
| `--valid` | 验证数据路径 | - |
| `--test` | 测试数据路径 | - |
| `--epochs` | 训练轮数 | 3 |
| `--batch_size` | 批次大小 | 16 |
| `--lr` | 学习率 | 2e-5 |

**输出**：
- `model/final_model/` - 训练后的模型
- `loss_curve.png` - 损失曲线图
- `confusion_matrix.png` - 混淆矩阵图

---

#### 2. 模型评估 - `eval.py`

对训练好的模型进行全面评估。

```bash
python eval.py --data ../data/test_toxic_comments_cleaned.jsonl
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径 | `model/final_model` |
| `--data` | 测试数据路径 | `data/test_toxic_comments_cleaned.jsonl` |
| `--batch_size` | 推理批大小 | 16 |
| `--output` | 结果输出目录 | `eval_results` |

**输出**：
- `eval_results/confusion_matrix.png` - 混淆矩阵图
- `eval_results/metrics_comparison.png` - 指标对比图
- `eval_results/eval_summary.png` - 综合评估报告图
- `eval_results/evaluation_report.json` - JSON 格式评估报告

---

#### 3. 模型推理 - `inference.py`

使用训练好的模型预测单条或多条评论。

```bash
# 预测单条评论
python inference.py --text "质量很好，加微信有优惠"

# 使用原预训练模型（未微调，仅用于验证系统）
python inference.py --text "质量很好" --use_base_model

# 演示模式
python inference.py --demo
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--text` | 待检测的评论文本 |
| `--use_base_model` | 使用原预训练模型（未微调） |
| `--demo` | 运行演示模式 |

---

## 完整工作流程

```bash
# 1. 生成数据（可选）
cd scripts
python data_generator.py

# 2. 清洗数据
python data_cleaner.py

# 3. 查看数据统计
python multi_dataset_stats.py

# 4. 训练模型
python train.py --data ../data/train_toxic_comments_cleaned.jsonl --valid ../data/valid_toxic_comments_cleaned.jsonl --test ../data/test_toxic_comments_cleaned.jsonl

# 5. 评估模型
python eval.py --data ../data/test_toxic_comments_cleaned.jsonl

# 6. 使用模型预测
python inference.py --text "这是一条测试评论"
```
