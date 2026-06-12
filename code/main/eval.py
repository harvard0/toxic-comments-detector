import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

# 获取脚本所在目录，构建绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
CODE_DIR = os.path.join(PROJECT_ROOT, "code")

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "chinese-roberta-wwm-ext")
FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model")
DEFAULT_TEST_DATA = os.path.join(CODE_DIR, "data", "test_toxic_comments_cleaned.jsonl")
OUTPUT_DIR = os.path.join(CODE_DIR, "eval_results")

LABEL_NAMES = {0: '正常评论', 1: '投毒评论'}


def setup_chinese_font():
    """配置中文字体，确保图表中中文正常显示"""
    font_candidates = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Noto Sans CJK SC',
        'SimHei',
        'Microsoft YaHei',
        'AR PL UMing CN',
    ]
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    found_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            found_font = font_name
            break

    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font, 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
    else:
        plt.rcParams['font.family'] = 'sans-serif'

    plt.rcParams['axes.unicode_minus'] = False
    return found_font or 'sans-serif'


class EvalDataset(Dataset):
    """评估数据集类"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class ModelEvaluator:
    """模型评估器：加载模型、执行推理、计算指标并生成评估报告"""

    def __init__(self, model_path=FINETUNED_MODEL_PATH, device=None):
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        print(f"[模型加载] 路径: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=2
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[模型加载] 设备: {self.device}")

    def load_test_data(self, data_path):
        """加载JSONL格式的测试数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"测试数据文件不存在: {data_path}")

        texts, labels, categories, raw_data = [], [], [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                raw_data.append(item)
                texts.append(item['text'])
                labels.append(item['label'])
                categories.append(item.get('category', 'unknown'))

        print(f"[数据加载] 共 {len(texts)} 条测试样本")
        print(f"[数据分布] 正常评论(label=0): {labels.count(0)} 条, "
              f"投毒评论(label=1): {labels.count(1)} 条")
        return texts, labels, categories, raw_data

    def predict_batch(self, texts, labels, batch_size=16):
        """批量推理并收集预测结果"""
        dataset = EvalDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1)
                confidence = probabilities[
                    torch.arange(len(predicted_class)), predicted_class
                ]

                all_preds.extend(predicted_class.cpu().tolist())
                all_labels.extend(batch_labels.cpu().tolist())
                all_confidences.extend(confidence.cpu().tolist())

                if (batch_idx + 1) % 10 == 0:
                    print(f"[推理进度] 已处理 {(batch_idx + 1) * batch_size} 条...")

        return all_preds, all_labels, all_confidences

    def compute_metrics(self, y_true, y_pred):
        """计算所有评估指标，特别关注少数类（投毒评论）"""

        accuracy = accuracy_score(y_true, y_pred)

        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        precision_per_class = np.asarray(
            precision_score(y_true, y_pred, average=None, zero_division=0)
        )
        recall_per_class = np.asarray(
            recall_score(y_true, y_pred, average=None, zero_division=0)
        )
        f1_per_class = np.asarray(
            f1_score(y_true, y_pred, average=None, zero_division=0)
        )

        cm = confusion_matrix(y_true, y_pred)

        poison_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0.0
        poison_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0.0
        poison_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0.0

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_normal': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
            'recall_normal': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
            'f1_normal': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
            'precision_poison': poison_precision,
            'recall_poison': poison_recall,
            'f1_poison': poison_f1,
            'confusion_matrix': cm,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        }
        return metrics

    def _apply_professional_style(self, ax):
        """应用统一的美化样式到坐标轴"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')
        ax.tick_params(colors='#555555', labelsize=12)

    def generate_confusion_matrix_plot(self, cm, save_path):
        """生成混淆矩阵图（专业美化版）"""
        from matplotlib.colors import LinearSegmentedColormap

        total = cm.sum()
        cm_percent = cm / total * 100

        colors_list = ['#F7FCFD', '#E5F5F9', '#CCEBF5', '#99D8C9', '#66C2A4',
                        '#41AE76', '#238B45', '#006D2C', '#00441B']
        cmap_custom = LinearSegmentedColormap.from_list('custom_green', colors_list, N=256)

        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')

        im = ax.imshow(cm, interpolation='lanczos', cmap=cmap_custom, aspect='equal', alpha=0.92)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor('#CCCCCC')
        cbar.ax.tick_params(labelsize=10, colors='#555555', length=0)
        cbar.set_label('样本数', fontsize=11, color='#555555', labelpad=10)

        classes = ['正常评论', '投毒评论']
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='真实标签',
            xlabel='预测标签',
        )
        ax.set_title('混淆矩阵', fontsize=18, fontweight='bold',
                      color='#2C3E50', pad=20)

        ax.xaxis.set_label_coords(0.5, -0.10)
        ax.yaxis.set_label_coords(-0.12, 0.5)
        ax.xaxis.label.set_color('#555555')
        ax.yaxis.label.set_color('#555555')
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)

        self._apply_professional_style(ax)
        ax.tick_params(top=False, bottom=False, left=False, right=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        border_width = 2
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(border_width)
            spine.set_color('#DDDDDD')

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_percent[i, j]
                text_color = '#FFFFFF' if count > cm.max() * 0.45 else '#2C3E50'

                ax.text(
                    j, i - 0.18, f'{count}',
                    ha='center', va='center',
                    color=text_color, fontsize=24, fontweight='bold',
                    fontfamily='sans-serif',
                )
                ax.text(
                    j, i + 0.24, f'({pct:.1f}%)',
                    ha='center', va='center',
                    color=text_color, fontsize=11, fontweight='normal',
                    fontfamily='sans-serif', alpha=0.9,
                )

        total_correct = int(cm[0, 0] + cm[1, 1])
        acc_pct = total_correct / total * 100
        ax.text(
            0.5, 1.14,
            f'整体准确率: {acc_pct:.1f}%    |    总样本: {total}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=12, color='#7F8C8D', fontfamily='sans-serif',
            fontstyle='italic',
        )

        fig.tight_layout(pad=2.0)
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"[图表] 混淆矩阵已保存至: {save_path}")

    def generate_metrics_bar_chart(self, metrics, save_path):
        """生成核心指标的柱状图（专业美化版）"""
        categories = ['正常评论', '投毒评论', '宏观平均', '加权平均']
        precision_vals = [
            metrics['precision_normal'],
            metrics['precision_poison'],
            metrics['precision_macro'],
            metrics['precision_weighted'],
        ]
        recall_vals = [
            metrics['recall_normal'],
            metrics['recall_poison'],
            metrics['recall_macro'],
            metrics['recall_weighted'],
        ]
        f1_vals = [
            metrics['f1_normal'],
            metrics['f1_poison'],
            metrics['f1_macro'],
            metrics['f1_weighted'],
        ]

        palette = {
            'precision': '#3498DB',
            'recall': '#E74C3C',
            'f1': '#2ECC71',
        }

        x = np.arange(len(categories))
        width = 0.22
        gap = 0.03

        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('#FAFBFC')

        bars1 = ax.bar(
            x - width - gap, precision_vals, width,
            label='Precision',
            color=palette['precision'], edgecolor='white', linewidth=0.8,
            zorder=3,
        )
        bars2 = ax.bar(
            x, recall_vals, width,
            label='Recall',
            color=palette['recall'], edgecolor='white', linewidth=0.8,
            zorder=3,
        )
        bars3 = ax.bar(
            x + width + gap, f1_vals, width,
            label='F1-Score',
            color=palette['f1'], edgecolor='white', linewidth=0.8,
            zorder=3,
        )

        ax.set_ylabel('分数', fontsize=14, color='#555555', labelpad=14)
        ax.set_title('模型评估指标对比', fontsize=20, fontweight='bold',
                      color='#2C3E50', pad=25)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=13, color='#2C3E50')
        ax.set_ylim(0, 1.18)
        ax.set_xlim(-0.5, len(categories) - 0.5)

        ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.8, color='#BDC3C7', zorder=0)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.tick_params(axis='y', labelsize=11, colors='#7F8C8D')

        self._apply_professional_style(ax)

        legend = ax.legend(
            loc='lower right', fontsize=12, frameon=True,
            fancybox=True, framealpha=0.9,
            edgecolor='#E0E0E0', borderpad=0.8,
            labelspacing=0.6, handlelength=1.5,
        )
        legend.get_frame().set_linewidth(0.5)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.022,
                        f'{height:.3f}',
                        ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold',
                        color='#555555',
                    )

        for i, category in enumerate(categories):
            if category == '投毒评论':
                ax.axvspan(
                    i - 0.48, i + 0.48,
                    alpha=0.06, color='#E74C3C', zorder=0,
                )

        max_val = max(max(precision_vals), max(recall_vals), max(f1_vals))
        ax.axhline(
            y=max_val, color='#BDC3C7', linestyle='--',
            linewidth=1, alpha=0.6, zorder=2,
        )
        ax.text(
            len(categories) - 1, max_val + 0.015,
            f'最高: {max_val:.3f}',
            ha='right', va='bottom', fontsize=9,
            color='#95A5A6', fontstyle='italic',
        )

        fig.tight_layout(pad=2.5)
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"[图表] 指标对比图已保存至: {save_path}")

    def generate_summary_report_image(self, metrics, cm, y_true, y_pred, save_path):
        """生成综合评估报告图（专业美化版）"""
        from matplotlib.colors import LinearSegmentedColormap

        fig = plt.figure(figsize=(18, 12), facecolor='white')

        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9],
                              hspace=0.35, wspace=0.35,
                              left=0.06, right=0.94, top=0.92, bottom=0.08)

        # ── 左上：混淆矩阵 ──
        ax_cm = fig.add_subplot(gs[0, 0])
        cm_total = cm.sum()
        cm_percent = cm / cm_total * 100
        cm_colors = ['#F7FCFD', '#E5F5F9', '#CCEBF5', '#99D8C9', '#66C2A4',
                     '#41AE76', '#238B45', '#006D2C', '#00441B']
        cmap_cm = LinearSegmentedColormap.from_list('cm_custom', cm_colors, N=256)
        ax_cm.imshow(cm, interpolation='lanczos', cmap=cmap_cm, aspect='equal', alpha=0.92)
        classes = ['正常评论', '投毒评论']
        ax_cm.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='真实标签',
            xlabel='预测标签',
        )
        ax_cm.set_title('A. 混淆矩阵', fontsize=14, fontweight='bold',
                         color='#2C3E50', loc='left', pad=12)
        ax_cm.xaxis.set_label_coords(0.5, -0.12)
        ax_cm.yaxis.set_label_coords(-0.14, 0.5)
        ax_cm.xaxis.label.set_color('#555555')
        ax_cm.yaxis.label.set_color('#555555')
        ax_cm.tick_params(labelsize=12, colors='#2C3E50', length=0)
        for spine in ax_cm.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('#E0E0E0')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_percent[i, j]
                tc = '#FFFFFF' if count > cm.max() * 0.45 else '#2C3E50'
                ax_cm.text(j, i - 0.2, f'{count}', ha='center', va='center',
                            color=tc, fontsize=20, fontweight='bold')
                ax_cm.text(j, i + 0.22, f'({pct:.1f}%)', ha='center', va='center',
                            color=tc, fontsize=10, alpha=0.85)

        # ── 中上：指标表格 ──
        ax_table = fig.add_subplot(gs[0, 1])
        ax_table.axis('off')
        table_headers = ['指标', '正常评论', '投毒评论', '整体']
        table_data = [
            ['Precision',
             f'{metrics["precision_normal"]:.4f}',
             f'{metrics["precision_poison"]:.4f}',
             f'{metrics["precision_weighted"]:.4f}'],
            ['Recall',
             f'{metrics["recall_normal"]:.4f}',
             f'{metrics["recall_poison"]:.4f}',
             f'{metrics["recall_weighted"]:.4f}'],
            ['F1-Score ★',
             f'{metrics["f1_normal"]:.4f}',
             f'{metrics["f1_poison"]:.4f}',
             f'{metrics["f1_weighted"]:.4f}'],
            ['Accuracy',
             '—', '—',
             f'{metrics["accuracy"]:.4f}'],
        ]
        table = ax_table.table(
            cellText=table_data,
            colLabels=table_headers,
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11.5)
        table.scale(1.15, 2.0)

        header_color = '#2C3E50'
        for col_idx in range(4):
            cell = table[0, col_idx]
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=12)
            cell.set_edgecolor('#1A252F')

        f1_row_idx = 3
        accuracy_row_idx = 4
        alt_colors = ['#FFFFFF', '#F8F9FA', '#FFFFFF', '#F8F9FA']
        for row_idx in [1, 2, 3, 4]:
            for col_idx in range(4):
                cell = table[row_idx, col_idx]
                cell.set_facecolor(alt_colors[row_idx - 1])
                cell.set_edgecolor('#E8E8E8')
                cell.set_linewidth(0.5)
                if row_idx == f1_row_idx:
                    cell.set_facecolor('#E8F8F5')
                if row_idx == accuracy_row_idx:
                    cell.set_facecolor('#FEF9E7')

        ax_table.set_title('B. 核心指标汇总', fontsize=14, fontweight='bold',
                            color='#2C3E50', loc='left', pad=12)

        # ── 右上：正常 vs 投毒 雷达对比 ──
        ax_radar = fig.add_subplot(gs[0, 2])
        categories_radar = ['Precision', 'Recall', 'F1-Score']
        normal_vals = [metrics['precision_normal'],
                       metrics['recall_normal'],
                       metrics['f1_normal']]
        poison_vals = [metrics['precision_poison'],
                       metrics['recall_poison'],
                       metrics['f1_poison']]
        N = len(categories_radar)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        normal_vals_closed = normal_vals + normal_vals[:1]
        poison_vals_closed = poison_vals + poison_vals[:1]

        ax_radar.set_facecolor('white')
        ax_radar = fig.add_subplot(gs[0, 2], polar=True)
        ax_radar.set_facecolor('#FAFBFC')
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories_radar, fontsize=12, color='#2C3E50')
        ax_radar.set_ylim(0, 1.05)
        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                                  fontsize=8, color='#AAAAAA')
        ax_radar.grid(True, alpha=0.3, linewidth=0.6, color='#CCCCCC')

        ax_radar.fill(angles, normal_vals_closed, alpha=0.12, color='#3498DB')
        ax_radar.plot(angles, normal_vals_closed, 'o-', linewidth=2.2,
                       color='#3498DB', markersize=7, label='正常评论',
                       markerfacecolor='white', markeredgewidth=2)
        ax_radar.fill(angles, poison_vals_closed, alpha=0.15, color='#E74C3C')
        ax_radar.plot(angles, poison_vals_closed, 's-', linewidth=2.2,
                       color='#E74C3C', markersize=7, label='投毒评论',
                       markerfacecolor='white', markeredgewidth=2)

        legend_radar = ax_radar.legend(
            loc='upper right', bbox_to_anchor=(1.3, 1.12),
            fontsize=10.5, frameon=True, fancybox=True,
            framealpha=0.9, edgecolor='#E0E0E0', borderpad=0.6,
        )
        legend_radar.get_frame().set_linewidth(0.5)
        ax_radar.set_title('C. 类别性能雷达图', fontsize=14, fontweight='bold',
                            color='#2C3E50', loc='left', pad=20)

        # ── 下半部分：投毒专项分析 + 结论 ──
        ax_poison = fig.add_subplot(gs[1, :])
        ax_poison.set_facecolor('#FAFBFC')

        poison_labels = ['Precision', 'Recall', 'F1-Score']
        poison_values = [
            metrics['precision_poison'],
            metrics['recall_poison'],
            metrics['f1_poison'],
        ]
        colors_poison = ['#2980B9', '#C0392B', '#27AE60']
        bar_positions = np.arange(3)
        bar_width = 0.45

        bars = ax_poison.bar(
            bar_positions, poison_values, bar_width,
            color=colors_poison, edgecolor='white', linewidth=1.2,
            zorder=3, alpha=0.9,
        )

        target_line = 0.80
        ax_poison.axhline(
            y=target_line, color='#F39C12', linestyle='--',
            linewidth=1.5, alpha=0.7, zorder=2,
        )
        ax_poison.text(
            2.55, target_line, f'目标线: {target_line}',
            ha='right', va='bottom', fontsize=9,
            color='#F39C12', fontweight='bold', fontstyle='italic',
        )

        for bar, val in zip(bars, poison_values):
            ax_poison.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f'{val:.4f}',
                ha='center', va='bottom',
                fontsize=15, fontweight='bold',
                color='#2C3E50',
            )

        ax_poison.set_xticks(bar_positions)
        ax_poison.set_xticklabels(poison_labels, fontsize=14, color='#2C3E50')
        ax_poison.set_ylim(0, 1.25)
        ax_poison.set_ylabel('分数', fontsize=13, color='#555555', labelpad=10)
        ax_poison.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.6,
                        color='#BDC3C7', zorder=0)
        ax_poison.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax_poison.tick_params(axis='y', labelsize=10, colors='#7F8C8D')

        self._apply_professional_style(ax_poison)

        poison_f1 = metrics['f1_poison']
        if poison_f1 >= 0.85:
            verdict = '优秀'
            verdict_color = '#27AE60'
        elif poison_f1 >= 0.70:
            verdict = '良好'
            verdict_color = '#2980B9'
        elif poison_f1 >= 0.50:
            verdict = '一般'
            verdict_color = '#F39C12'
        else:
            verdict = '需改进'
            verdict_color = '#E74C3C'

        ax_poison.set_title(
            'D. 投毒评论（少数类）识别性能专项分析',
            fontsize=15, fontweight='bold', color='#2C3E50',
            loc='left', pad=12,
        )

        ax_poison.text(
            2.6, 1.12, f'投毒F1评级: {verdict}',
            ha='right', va='top',
            fontsize=13, fontweight='bold', color=verdict_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=verdict_color, alpha=0.9, linewidth=1.5),
        )

        poison_count = int(sum(y_true))
        normal_count = len(y_true) - poison_count
        poison_pred_correct = int(cm[1, 1])
        poison_missed = int(cm[1, 0])

        summary_text = (
            f'数据集: 共 {len(y_true)} 条样本    '
            f'正常评论: {normal_count} 条    '
            f'投毒评论: {poison_count} 条 ({poison_count / len(y_true) * 100:.1f}%)    '
            f'整体准确率: {metrics["accuracy"]:.4f}    '
            f'加权F1: {metrics["f1_weighted"]:.4f}    '
            f'投毒检出: {poison_pred_correct}/{poison_count}    '
            f'漏检: {poison_missed} 条'
        )
        ax_poison.text(
            0.5, -0.38, summary_text,
            ha='center', va='center',
            fontsize=11, color='#555555', fontfamily='sans-serif',
            transform=ax_poison.transAxes,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8F9FA',
                       edgecolor='#D5D8DC', alpha=0.95, linewidth=1),
        )

        fig.suptitle('AI投毒评论识别模型 — 评估报告',
                     fontsize=22, fontweight='bold', color='#1A252F', y=0.985)

        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"[图表] 综合评估报告图已保存至: {save_path}")

    def print_terminal_report(self, metrics, y_true, y_pred, categories, raw_data):
        """终端打印完整评估报告"""
        print("\n" + "=" * 70)
        print("  AI投毒评论识别模型 — 评估报告")
        print("=" * 70)

        total = len(y_true)
        poison_total = sum(y_true)
        normal_total = total - poison_total
        print("\n  [数据集概况]")
        print(f"    总样本数:     {total}")
        print(f"    正常评论:     {normal_total} ({normal_total / total * 100:.1f}%)")
        print(f"    投毒评论:     {poison_total} ({poison_total / total * 100:.1f}%)")

        print("\n  [混淆矩阵]")
        print("                     预测正常    预测投毒")
        print(f"    真实正常          {metrics['tn']:>6}      {metrics['fp']:>6}")
        print(f"    真实投毒          {metrics['fn']:>6}      {metrics['tp']:>6}")

        print("\n  [整体指标]")
        print(f"    Accuracy (准确率):        {metrics['accuracy']:.4f}")
        print(f"    Macro-F1 (宏平均F1):      {metrics['f1_macro']:.4f}")
        print(f"    Weighted-F1 (加权F1):     {metrics['f1_weighted']:.4f}")

        print("\n  [各类别指标]")
        print(f"    {'类别':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print(f"    {'-' * 52}")
        print(
            f"    {'正常评论':<12} {metrics['precision_normal']:>10.4f} "
            f"{metrics['recall_normal']:>10.4f} {metrics['f1_normal']:>10.4f} "
            f"{normal_total:>10}"
        )
        print(
            f"    {'投毒评论':<12} {metrics['precision_poison']:>10.4f} "
            f"{metrics['recall_poison']:>10.4f} {metrics['f1_poison']:>10.4f} "
            f"{poison_total:>10}"
        )

        print("\n  [少数类（投毒评论）专项分析]")
        print(f"    ★ 投毒评论 F1-Score:   {metrics['f1_poison']:.4f}")
        print(f"    ★ 投毒评论 Precision:  {metrics['precision_poison']:.4f}")
        print(f"    ★ 投毒评论 Recall:     {metrics['recall_poison']:.4f}")
        print(f"    ★ 投毒检出率:          {metrics['tp']}/{poison_total} "
              f"({metrics['tp'] / max(poison_total, 1) * 100:.1f}%)")
        print(f"    ★ 投毒漏检数:          {metrics['fn']}")
        print(f"    ★ 投毒误报数:          {metrics['fp']} (将正常评论误判为投毒)")

        poison_f1 = metrics['f1_poison']
        if poison_f1 >= 0.85:
            poison_verdict = '优秀 — 对投毒评论的识别能力很强'
        elif poison_f1 >= 0.70:
            poison_verdict = '良好 — 能有效识别大部分投毒评论'
        elif poison_f1 >= 0.50:
            poison_verdict = '一般 — 有一定识别能力，但存在较多误判'
        else:
            poison_verdict = '较差 — 需要重点改进投毒评论的识别能力'

        print(f"    ★ 评价: {poison_verdict}")

        print("\n  [模型综合评价]")
        overall_f1 = metrics['f1_weighted']
        if overall_f1 >= 0.90:
            overall_verdict = '模型整体表现优秀，可投入实际使用。'
        elif overall_f1 >= 0.80:
            overall_verdict = '模型整体表现良好，建议针对少数类做进一步优化。'
        elif overall_f1 >= 0.70:
            overall_verdict = '模型整体表现一般，需要进一步提升泛化能力。'
        else:
            overall_verdict = '模型整体表现不佳，建议调整训练策略或增加数据。'
        print(f"    {overall_verdict}")

        if poison_f1 < 0.70:
            print("\n  [优化建议]")
            if metrics['recall_poison'] < 0.70:
                print("    - 召回率偏低，建议增加投毒样本、使用类别加权损失函数")
            if metrics['precision_poison'] < 0.70:
                print("    - 精确率偏低，建议调整分类阈值或使用数据增强")
            if poison_total < 50:
                print(f"    - 投毒样本数量较少({poison_total}条)，建议扩充训练数据中投毒样本的比例")

        print("\n" + "=" * 70)

    def save_structured_report(self, metrics, y_true, y_pred, output_dir):
        """保存结构化评估报告为JSON文件"""
        report = {
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'dataset': {
                'total_samples': len(y_true),
                'normal_samples': int(sum(1 for y in y_true if y == 0)),
                'poison_samples': int(sum(1 for y in y_true if y == 1)),
                'poison_ratio': float(sum(y_true) / len(y_true)),
            },
            'confusion_matrix': {
                'tn': metrics['tn'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'tp': metrics['tp'],
            },
            'overall_metrics': {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted']),
                'precision_macro': float(metrics['precision_macro']),
                'recall_macro': float(metrics['recall_macro']),
                'precision_weighted': float(metrics['precision_weighted']),
                'recall_weighted': float(metrics['recall_weighted']),
            },
            'per_class_metrics': {
                'normal': {
                    'precision': float(metrics['precision_normal']),
                    'recall': float(metrics['recall_normal']),
                    'f1': float(metrics['f1_normal']),
                },
                'poison': {
                    'precision': float(metrics['precision_poison']),
                    'recall': float(metrics['recall_poison']),
                    'f1': float(metrics['f1_poison']),
                    'detection_rate': float(
                        metrics['tp'] / max(sum(y_true), 1)
                    ),
                    'missed_count': metrics['fn'],
                    'false_alarm_count': metrics['fp'],
                },
            },
        }

        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[报告] 评估报告JSON已保存至: {report_path}")
        return report

    def run_full_evaluation(self, data_path, batch_size=16):
        """执行完整的评估流程"""
        print("\n" + "━" * 50)
        print("  开始执行评估流程")
        print("━" * 50)

        texts, labels, categories, raw_data = self.load_test_data(data_path)

        print(f"\n[推理] 开始批量预测 (batch_size={batch_size})...")
        y_pred, y_true, confidences = self.predict_batch(texts, labels, batch_size)
        print(f"[推理] 完成，共处理 {len(y_pred)} 条样本")

        print("\n[指标计算] 开始计算评估指标...")
        metrics = self.compute_metrics(y_true, y_pred)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        self.generate_confusion_matrix_plot(metrics['confusion_matrix'], cm_path)

        bar_path = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
        self.generate_metrics_bar_chart(metrics, bar_path)

        summary_path = os.path.join(OUTPUT_DIR, 'eval_summary.png')
        self.generate_summary_report_image(
            metrics, metrics['confusion_matrix'], y_true, y_pred, summary_path
        )

        self.print_terminal_report(metrics, y_true, y_pred, categories, raw_data)

        report = self.save_structured_report(metrics, y_true, y_pred, OUTPUT_DIR)

        print(f"\n[完成] 所有评估产物已保存至: {os.path.abspath(OUTPUT_DIR)}/")
        print("  - 混淆矩阵图:   confusion_matrix.png")
        print("  - 指标对比图:   metrics_comparison.png")
        print("  - 综合评估图:   eval_summary.png")
        print("  - 评估报告JSON: evaluation_report.json")

        return y_pred, y_true, metrics, report


def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description='AI投毒评论识别模型 — 评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval.py                                          # 使用默认路径评估微调模型
  python eval.py --model ./final_model --data ./data/test_toxic_comments_cleaned.jsonl
  python eval.py --model ./chinese-roberta-wwm-ext --data ./data/test_toxic_comments_cleaned.jsonl
  python eval.py --batch_size 32 --output ./my_eval_results
        """,
    )

    parser.add_argument(
        '--model', type=str, default=FINETUNED_MODEL_PATH,
        help=f'模型路径 (默认: {FINETUNED_MODEL_PATH})',
    )
    parser.add_argument(
        '--data', type=str, default=DEFAULT_TEST_DATA,
        help=f'测试数据文件路径 (默认: {DEFAULT_TEST_DATA})',
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='推理批大小 (默认: 16)',
    )
    parser.add_argument(
        '--output', type=str, default=OUTPUT_DIR,
        help=f'评估结果输出目录 (默认: {OUTPUT_DIR})',
    )

    args = parser.parse_args()
    OUTPUT_DIR = args.output

    setup_chinese_font()

    try:
        evaluator = ModelEvaluator(model_path=args.model)
        evaluator.run_full_evaluation(
            data_path=args.data, batch_size=args.batch_size
        )
    except FileNotFoundError as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        print("请确保:", file=sys.stderr)
        print("  1. 模型已训练并保存到指定路径", file=sys.stderr)
        print("  2. 测试数据文件存在", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 评估过程中出现异常: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
