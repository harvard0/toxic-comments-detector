from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
import os

# 获取脚本所在目录，构建绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "chinese-roberta-wwm-ext")
FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model")

class PoisonCommentPredictor:
    def __init__(self, model_path=FINETUNED_MODEL_PATH, use_base_model=False):
        if use_base_model:
            print(f"使用原预训练模型: {BASE_MODEL_PATH}")
            actual_path = BASE_MODEL_PATH
        else:
            print(f"使用微调后的模型: {model_path}")
            actual_path = model_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(actual_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            actual_path, 
            num_labels=2
        )
        self.model.eval()
        self.use_base_model = use_base_model
    
    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'text': text,
            'is_poison': predicted_class == 1,
            'label': '投毒评论' if predicted_class == 1 else '正常评论',
            'confidence': confidence,
            'model_type': '原预训练模型' if self.use_base_model else '微调模型'
        }
        
        return result

_predictor = None
_predictor_base = None

def predict(text: str, use_base_model: bool = False) -> dict:
    global _predictor, _predictor_base
    
    if use_base_model:
        if _predictor_base is None:
            _predictor_base = PoisonCommentPredictor(use_base_model=True)
        return _predictor_base.predict(text)
    else:
        if _predictor is None:
            _predictor = PoisonCommentPredictor(use_base_model=False)
        return _predictor.predict(text)

def batch_predict(texts: list, use_base_model: bool = False) -> list:
    predictor = PoisonCommentPredictor(use_base_model=use_base_model)
    results = []
    for text in texts:
        result = predictor.predict(text)
        results.append(result)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI投毒评论识别推理脚本')
    parser.add_argument('--text', type=str, help='待检测的评论文本')
    parser.add_argument('--use_base_model', action='store_true', help='使用原预训练模型进行推理（用于系统初步验证）')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')
    
    args = parser.parse_args()
    
    if args.demo:
        print("=" * 60)
        print("AI投毒评论识别系统 - 演示模式")
        print("=" * 60)
        
        demo_texts = [
            "产品质量不错，顺便推荐一下XX店铺的同款",
            "质量很好，物流也快",
            "加微信XXX，正品低价",
            "产品不错，下次还会买",
            "关注公众号领优惠券",
            "一般般吧，没有想象中好",
            "用了几天感觉还可以，加微信有优惠",
            "性价比很高，去某平台买更便宜"
        ]
        
        print("\n使用原预训练模型进行初步验证:")
        print("-" * 60)
        
        predictor_base = PoisonCommentPredictor(use_base_model=True)
        
        for text in demo_texts:
            result = predictor_base.predict(text)
            print(f"\n文本: {text}")
            print(f"结果: {result['label']}")
            print(f"置信度: {result['confidence']*100:.1f}%")
            print(f"模型类型: {result['model_type']}")
        
        print("\n" + "=" * 60)
        print("注意: 原预训练模型未经微调，结果仅供参考")
        print("待微调完成后，使用微调模型可获得更准确的识别效果")
        print("=" * 60)
        
    elif args.text:
        result = predict(args.text, use_base_model=args.use_base_model)
        print("\n检测结果:")
        print("-" * 40)
        print(f"文本: {result['text']}")
        print(f"标签: {result['label']}")
        print(f"是否投毒: {result['is_poison']}")
        print(f"置信度: {result['confidence']*100:.1f}%")
        print(f"模型类型: {result['model_type']}")
        print("-" * 40)
        
    else:
        print("使用方法:")
        print("  python inference.py --text \"评论文本\" --use_base_model")
        print("  python inference.py --demo --use_base_model")
        print("\n参数说明:")
        print("  --text          待检测的评论文本")
        print("  --use_base_model 使用原预训练模型（用于系统初步验证）")
        print("  --demo          运行演示模式")
        
        print("\n示例:")
        print("  # 使用原模型验证系统可行性")
        print("  python inference.py --text \"质量很好，加微信有优惠\" --use_base_model")
        print("\n  # 使用微调后的模型")
        print("  python inference.py --text \"质量很好，加微信有优惠\"")