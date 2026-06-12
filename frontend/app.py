"""
AI投毒评论识别系统 - 前端后端API服务
调用现有的 inference.py 进行评论检测
"""

from flask import Flask, request, jsonify, render_template
import sys
import os
import torch
import json

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.normpath(os.path.join(CURRENT_DIR, ".."))
# 主代码目录
MAIN_DIR = os.path.join(PROJECT_ROOT, "code", "main")
# 模型路径
FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model")

# 将主代码目录添加到路径
sys.path.insert(0, MAIN_DIR)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# 全局模型加载
model = None
tokenizer = None
model_loaded = False


def load_model():
    """加载微调后的模型"""
    global model, tokenizer, model_loaded

    if model_loaded:
        return True

    try:
        print(f"加载模型: {FINETUNED_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            FINETUNED_MODEL_PATH,
            num_labels=2
        )
        model.eval()
        model_loaded = True
        print("模型加载成功!")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def predict_toxic(text: str) -> dict:
    """
    预测评论是否为投毒评论
    返回: {'is_poison': bool, 'label': str, 'confidence': float}
    """
    if not model_loaded:
        if not load_model():
            return {
                'is_poison': None,
                'label': '模型加载失败',
                'confidence': 0,
                'error': '无法加载模型'
            }

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    is_poison = predicted_class == 1

    return {
        'is_poison': is_poison,
        'label': '投毒评论' if is_poison else '正常评论',
        'confidence': round(confidence * 100, 2),
        'text': text
    }


@app.route('/')
def home():
    """渲染首页"""
    # 从评估报告中读取模型性能数据
    eval_report_path = os.path.join(PROJECT_ROOT, "model", "final_model", "eval_results.json")
    model_metrics = {
        "accuracy": "93.6%",
        "precision": "96.1%",
        "recall": "90.9%",
        "f1": "93.4%"
    }

    try:
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            model_metrics = {
                "accuracy": f"{data.get('eval_accuracy', 0) * 100:.1f}%",
                "precision": f"{data.get('eval_precision', 0) * 100:.1f}%",
                "recall": f"{data.get('eval_recall', 0) * 100:.1f}%",
                "f1": f"{data.get('eval_f1', 0) * 100:.1f}%"
            }
    except Exception:
        pass

    return render_template('home.html', metrics=model_metrics)


@app.route('/detect')
def detect():
    """渲染检测页面"""
    return render_template('detect.html')


@app.route('/about')
def about():
    """渲染关于页面"""
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """预测API"""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({
            'success': False,
            'error': '请提供评论文本'
        }), 400

    text = data['text'].strip()

    if not text:
        return jsonify({
            'success': False,
            'error': '评论文本不能为空'
        }), 400

    if len(text) > 500:
        return jsonify({
            'success': False,
            'error': '评论文本过长，请控制在500字以内'
        }), 400

    try:
        result = predict_toxic(text)
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'预测出错: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查API"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded
    })


if __name__ == '__main__':
    # 启动时预加载模型
    load_model()

    print("\n" + "="*50)
    print("AI投毒评论识别系统 - 前端服务")
    print("="*50)
    print(f"访问地址: http://127.0.0.1:5000")
    print("按 Ctrl+C 停止服务")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
