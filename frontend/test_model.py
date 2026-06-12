"""
测试脚本 - 验证模型和API功能
"""

import sys
import os

# 添加路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(CURRENT_DIR, "..", "code", "main")
sys.path.insert(0, MAIN_DIR)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = os.path.join(CURRENT_DIR, "..", "model", "final_model")

# 测试案例
TEST_CASES = [
    # 正常评论
    "产品质量不错，物流也很快，好评！",
    "收到货了，和描述的一样，性价比很高",
    "用了几天感觉还可以，下次还会回购",
    "总体满意，就是包装可以再精致一点",

    # 投毒评论
    "加微信XXX，正品低价，先到先得",
    "关注公众号领优惠券，更多优惠等你来",
    "去某宝买更便宜，链接：xxx",
    "拉你进群有优惠，名额有限快来加入",
    "加了微信之后发现是正品，推荐给大家",
    "质量一般，不如去XX店铺买同款",
]

def load_model():
    """加载模型"""
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model.eval()
    print("模型加载成功!\n")
    return tokenizer, model

def predict(text, tokenizer, model):
    """预测单条评论"""
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
    label = "投毒评论" if is_poison else "正常评论"

    return {
        "text": text,
        "label": label,
        "is_poison": is_poison,
        "confidence": confidence * 100
    }

def main():
    print("=" * 60)
    print("AI投毒评论识别系统 - 模型测试")
    print("=" * 60)

    tokenizer, model = load_model()

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    for i, text in enumerate(TEST_CASES, 1):
        result = predict(text, tokenizer, model)
        status = "⚠️" if result["is_poison"] else "✓"
        print(f"\n{i}. {status} {result['label']} (置信度: {result['confidence']:.1f}%)")
        print(f"   文本: {result['text']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
