import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录，构建绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "chinese-roberta-wwm-ext")
OUTPUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model")

class PoisonCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def load_dataset(data_file):
    texts = []
    labels = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                texts.append(item['text'])
                labels.append(item['label'])
            except json.JSONDecodeError:
                continue
    
    print(f"加载数据: {len(texts)} 条")
    return texts, labels

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, pos_label=1)
    recall = recall_score(labels, predictions, pos_label=1)
    f1 = f1_score(labels, predictions, pos_label=1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class LossRecorder:
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
        self.metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def record_train_loss(self, step, loss):
        self.train_losses.append(loss)
        self.steps.append(step)
    
    def record_eval_loss(self, step, loss, metrics):
        self.eval_losses.append(loss)
        self.eval_steps.append(step)
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])

def plot_loss_curve(recorder, save_path='loss_curve.png'):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if recorder.train_losses:
        plt.plot(recorder.steps, recorder.train_losses, 'b-', label='Training Loss', linewidth=2)
    if recorder.eval_losses:
        plt.plot(recorder.eval_steps, recorder.eval_losses, 'r-', label='Validation Loss', linewidth=2, marker='o')
    plt.title('训练与验证损失曲线 - AI投毒评论识别模型', fontsize=14)
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    if recorder.metrics_history['f1']:
        steps = recorder.eval_steps
        plt.plot(steps, recorder.metrics_history['accuracy'], 'g-', label='Accuracy', linewidth=2, marker='s')
        plt.plot(steps, recorder.metrics_history['precision'], 'c-', label='Precision', linewidth=2, marker='^')
        plt.plot(steps, recorder.metrics_history['recall'], 'm-', label='Recall', linewidth=2, marker='d')
        plt.plot(steps, recorder.metrics_history['f1'], 'r-', label='F1-Score (投毒类)', linewidth=2, marker='o')
        plt.title('评估指标变化曲线', fontsize=14)
        plt.xlabel('训练步数', fontsize=12)
        plt.ylabel('指标值', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss曲线已保存至: {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    import seaborn as sns
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['正常评论', '投毒评论'],
        yticklabels=['正常评论', '投毒评论'],
        annot_kws={'size': 14}
    )
    plt.title('混淆矩阵 - AI投毒评论识别', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    
    tn, fp, fn, tp = cm.ravel()
    
    plt.subplot(2, 1, 2)
    metrics_text = f"""
    分类结果统计:
    
    真负例 (TN): {tn} - 正确识别的正常评论
    假正例 (FP): {fp} - 误判为投毒的正常评论
    假负例 (FN): {fn} - 误判为正常的投毒评论
    真正例 (TP): {tp} - 正确识别的投毒评论
    
    评估指标:
    
    Accuracy:  {(tn+tp)/(tn+fp+fn+tp):.4f}
    Precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}
    Recall:    {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}
    F1-Score:  {2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0:.4f}
    
    投毒评论识别率: {tp/(tp+fn)*100 if (tp+fn) > 0 else 0:.1f}%
    正常评论误判率: {fp/(tn+fp)*100 if (tn+fp) > 0 else 0:.1f}%
    """
    plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")

def evaluate_model(model, tokenizer, test_file, output_dir):
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    test_texts, test_labels = load_dataset(test_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    predictions = []
    
    print(f"正在评估 {len(test_texts)} 条测试数据...")
    print(f"使用设备: {device}")
    
    batch_size = 32
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).tolist()
            predictions.extend(batch_preds)
    
    y_true = np.array(test_labels)
    y_pred = np.array(predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    print("\n" + "-"*40)
    print("评估结果:")
    print("-"*40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (投毒评论精确率)")
    print(f"Recall:    {recall:.4f} (投毒评论召回率)")
    print(f"F1-Score:  {f1:.4f} (投毒类F1，重点关注)")
    print("-"*40)
    
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['正常评论', '投毒评论']))
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("AI投毒评论识别模型评估报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"测试数据量: {len(test_texts)} 条\n")
        f.write(f"投毒评论: {sum(test_labels)} 条\n")
        f.write(f"正常评论: {len(test_labels)-sum(test_labels)} 条\n\n")
        f.write("-"*40 + "\n")
        f.write("评估指标:\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f} (投毒评论精确率)\n")
        f.write(f"Recall:    {recall:.4f} (投毒评论召回率)\n")
        f.write(f"F1-Score:  {f1:.4f} (投毒类F1，重点关注)\n\n")
        f.write("分类报告:\n")
        f.write(classification_report(y_true, y_pred, target_names=['正常评论', '投毒评论']))
    
    print(f"评估报告已保存至: {report_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(
    train_data_file=None, 
    train_texts=None, 
    train_labels=None,
    valid_data_file=None,
    base_model_path=BASE_MODEL_PATH,
    output_dir=OUTPUT_MODEL_PATH,
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, 
        num_labels=2
    )
    
    if train_data_file:
        train_texts, train_labels = load_dataset(train_data_file)
    
    if not train_texts or not train_labels:
        raise ValueError("请提供训练数据文件或训练文本和标签")
    
    poison_count = sum(train_labels)
    normal_count = len(train_labels) - poison_count
    print(f"训练数据: 投毒评论 {poison_count} 条, 正常评论 {normal_count} 条")
    
    train_dataset = PoisonCommentDataset(train_texts, train_labels, tokenizer)
    
    eval_dataset = None
    if valid_data_file:
        valid_texts, valid_labels = load_dataset(valid_data_file)
        eval_dataset = PoisonCommentDataset(valid_texts, valid_labels, tokenizer)
        print(f"验证数据: 投毒评论 {sum(valid_labels)} 条, 正常评论 {len(valid_labels)-sum(valid_labels)} 条")
    
    loss_recorder = LossRecorder()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,                    # 预热步数，稳定训练初期
        weight_decay=0.01,                   # 权重衰减，防止过拟合
        lr_scheduler_type='linear',          # 学习率线性衰减
        save_strategy='steps',
        save_steps=50,
        save_total_limit=1,
        logging_steps=1,
        logging_dir=os.path.join(output_dir, 'logs'),
        evaluation_strategy='steps' if eval_dataset else 'no',
        eval_steps=50 if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model='f1' if eval_dataset else None,
        greater_is_better=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset else None
    )
    
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    train_result = trainer.train()
    
    log_history = trainer.state.log_history
    for log in log_history:
        if 'loss' in log:
            loss_recorder.record_train_loss(log.get('step', 0), log['loss'])
        if 'eval_loss' in log:
            metrics = {
                'accuracy': log.get('eval_accuracy', 0),
                'precision': log.get('eval_precision', 0),
                'recall': log.get('eval_recall', 0),
                'f1': log.get('eval_f1', 0)
            }
            loss_recorder.record_eval_loss(log.get('step', 0), log['eval_loss'], metrics)
    
    loss_path = os.path.join(output_dir, 'loss_curve.png')
    plot_loss_curve(loss_recorder, save_path=loss_path)
    
    print(f"\n保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    trainer.save_state()
    trainer.save_metrics('train', train_result.metrics)
    
    if eval_dataset:
        eval_metrics = trainer.evaluate()
        trainer.save_metrics('eval', eval_metrics)
        print("\n验证集评估结果:")
        print(f"  Accuracy:  {eval_metrics.get('eval_accuracy', 0):.4f}")
        print(f"  Precision: {eval_metrics.get('eval_precision', 0):.4f}")
        print(f"  Recall:    {eval_metrics.get('eval_recall', 0):.4f}")
        print(f"  F1-Score:  {eval_metrics.get('eval_f1', 0):.4f} (投毒类，重点关注)")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    return model, tokenizer

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI投毒评论识别模型训练脚本')
    parser.add_argument('--data', type=str, help='训练数据文件路径 (JSONL格式)')
    parser.add_argument('--valid', type=str, help='验证数据文件路径 (JSONL格式)')
    parser.add_argument('--test', type=str, help='测试数据文件路径 (JSONL格式)')
    parser.add_argument('--base_model', type=str, default=BASE_MODEL_PATH, help='基础模型路径')
    parser.add_argument('--output', type=str, default=OUTPUT_MODEL_PATH, help='输出模型路径')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    
    args = parser.parse_args()
    
    if args.data:
        model, tokenizer = train_model(
            train_data_file=args.data,
            valid_data_file=args.valid,
            base_model_path=args.base_model,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        if args.test:
            evaluate_model(model, tokenizer, args.test, args.output)
    else:
        print("警告: 未提供训练数据，使用示例数据进行演示...")
        sample_texts = [
            "产品质量不错，顺便推荐一下XX店铺的同款",
            "质量很好，物流也快",
            "加微信XXX，正品低价",
            "产品不错，下次还会买",
            "关注公众号领优惠券",
            "一般般吧，没有想象中好"
        ]
        sample_labels = [1, 0, 1, 0, 1, 0]
        
        train_model(
            train_texts=sample_texts,
            train_labels=sample_labels,
            base_model_path=args.base_model,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )