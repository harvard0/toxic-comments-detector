from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
import os

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.normpath(os.path.join(CURRENT_DIR, ".."))

# 模型路径
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "chinese-roberta-wwm-ext")
FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "final_model")

# 后端配置变量（内部使用）
USE_BASE_MODEL = False  # True: 使用原预训练模型, False: 使用微调模型


class PoisonCommentPredictor:
    def __init__(self, model_path=FINETUNED_MODEL_PATH, use_base_model=False):
        if use_base_model:
            print(f"使用原预训练模型: {BASE_MODEL_PATH}")
            actual_path = BASE_MODEL_PATH
            self.use_base_model = True
        else:
            print(f"尝试加载微调模型: {model_path}")
            # 检查微调模型是否存在
            if os.path.exists(model_path) and os.path.isdir(model_path):
                actual_path = model_path
                self.use_base_model = False
            else:
                print(f"微调模型不存在，回退到预训练模型")
                actual_path = BASE_MODEL_PATH
                self.use_base_model = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(actual_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            actual_path, 
            num_labels=2
        )
        self.model.eval()
        self.model_type = '原预训练模型' if self.use_base_model else '微调模型'
    
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


def predict(text: str, use_base_model: bool = None) -> dict:
    """
    AI投毒评论预测函数
    
    参数:
        text: 待检测的评论文本
        use_base_model: 是否使用原预训练模型（默认None，使用全局配置USE_BASE_MODEL）
            - None (默认): 使用全局配置 USE_BASE_MODEL
            - True: 使用原预训练模型，用于系统验证
            - False: 使用微调模型，识别准确
    
    返回:
        dict: 包含预测结果和模型类型信息
    """
    global _predictor, _predictor_base
    
    # 如果未指定use_base_model参数，使用全局配置
    if use_base_model is None:
        use_base_model = USE_BASE_MODEL
    
    if use_base_model:
        if _predictor_base is None:
            _predictor_base = PoisonCommentPredictor(use_base_model=True)
        return _predictor_base.predict(text)
    else:
        if _predictor is None:
            _predictor = PoisonCommentPredictor(use_base_model=False)
        return _predictor.predict(text)


def get_current_model_type() -> str:
    """
    获取当前使用的模型类型
    
    返回:
        str: 当前使用的模型类型（"原预训练模型"或"微调模型"）
    """
    global _predictor, _predictor_base
    
    if USE_BASE_MODEL:
        if _predictor_base is not None:
            return _predictor_base.model_type
        return '原预训练模型'
    else:
        if _predictor is not None:
            return _predictor.model_type
        return '微调模型'


def batch_predict(texts: list, use_base_model: bool = None) -> list:
    """
    批量预测函数
    
    参数:
        texts: 待检测的评论文本列表
        use_base_model: 是否使用原预训练模型（默认None，使用全局配置USE_BASE_MODEL）
    
    返回:
        list: 包含多条预测结果的列表
    """
    # 如果未指定use_base_model参数，使用全局配置
    if use_base_model is None:
        use_base_model = USE_BASE_MODEL
    
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
            "什么垃圾质量，穿了一天鞋底就开胶了!找客服理论态度还极差，半天不回消息，申请退款还被拒绝，大家千万别买，避雷!",
            "买的水果烂了一大半，根本没法吃。跟详情页描述的完全不一样，又小又酸，纯纯的骗钱，已经打12315投诉了"
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
        print(f"\n当前全局配置: USE_BASE_MODEL = {USE_BASE_MODEL}")
        
        print("\n示例:")
        print("  # 使用原模型验证系统可行性")
        print("  python inference.py --text \"质量很好，加微信有优惠\" --use_base_model")
        print("\n  # 使用微调后的模型（如已训练）")
        print("  python inference.py --text \"质量很好，加微信有优惠\"")