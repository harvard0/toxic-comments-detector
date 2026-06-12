"""
AI投毒评论识别 - Flask API服务
提供RESTful接口用于识别AI投毒评论
"""

from flask import Flask, request, jsonify
from functools import wraps
import time
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import predict, batch_predict, USE_BASE_MODEL

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 512
REQUEST_TIMEOUT = 10

def get_model_type():
    """获取当前模型类型"""
    return "原预训练模型" if USE_BASE_MODEL else "微调模型"


def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({
                'code': 500,
                'message': f'服务器内部错误: {str(e)}',
                'data': None
            }), 500
    return decorated_function


@app.route('/predict', methods=['POST'])
@handle_errors
def predict_api():
    if not request.is_json:
        return jsonify({
            'code': 400,
            'message': '请求必须是JSON格式',
            'data': None
        }), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({
            'code': 400,
            'message': '缺少text参数',
            'data': None
        }), 400
    
    text = data['text']
    
    if not text or not isinstance(text, str):
        return jsonify({
            'code': 400,
            'message': '文本不能为空',
            'data': None
        }), 400
    
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        logger.warning(f'文本过长，已截断至{MAX_TEXT_LENGTH}字符')
    
    logger.info(f'收到预测请求: {text[:50]}... [使用{get_model_type()}]')
    
    start_time = time.time()
    result = predict(text, use_base_model=USE_BASE_MODEL)
    elapsed_time = time.time() - start_time
    
    logger.info(f'预测完成，耗时: {elapsed_time:.2f}秒 [模型类型: {result.get("model_type", "未知")}]')
    
    response_data = {
        'text': result['text'],
        'is_poison': result['is_poison'],
        'label': result['label'],
        'confidence': result['confidence']
    }
    
    return jsonify({
        'code': 200,
        'message': 'success',
        'data': response_data
    })


@app.route('/batch_predict', methods=['POST'])
@handle_errors
def batch_predict_api():
    if not request.is_json:
        return jsonify({
            'code': 400,
            'message': '请求必须是JSON格式',
            'data': None
        }), 400
    
    data = request.get_json()
    
    if 'texts' not in data:
        return jsonify({
            'code': 400,
            'message': '缺少texts参数',
            'data': None
        }), 400
    
    texts = data['texts']
    
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({
            'code': 400,
            'message': 'texts必须是非空列表',
            'data': None
        }), 400
    
    logger.info(f'收到批量预测请求，共{len(texts)}条 [使用{get_model_type()}]')
    
    start_time = time.time()
    results = batch_predict(texts, use_base_model=USE_BASE_MODEL)
    elapsed_time = time.time() - start_time
    
    logger.info(f'批量预测完成，耗时: {elapsed_time:.2f}秒')
    
    cleaned_results = []
    for result in results:
        cleaned_result = {
            'text': result['text'],
            'is_poison': result['is_poison'],
            'label': result['label'],
            'confidence': result['confidence']
        }
        cleaned_results.append(cleaned_result)
    
    return jsonify({
        'code': 200,
        'message': 'success',
        'data': cleaned_results
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'code': 200,
        'message': '服务正常运行',
        'data': {
            'status': 'healthy',
            'model_loaded': True,
            'current_model': get_model_type()
        }
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'code': 200,
        'message': 'AI投毒评论识别API',
        'data': {
            'endpoints': {
                '/predict': 'POST - 预测评论是否为投毒评论',
                '/batch_predict': 'POST - 批量预测多条评论',
                '/health': 'GET - 健康检查'
            },
            'parameters': {
                'text': '待检测的评论文本（必填）',
                'texts': '待检测的评论文本列表（批量预测必填）'
            },
            'current_model': get_model_type()
        }
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'code': 404,
        'message': '接口不存在',
        'data': None
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'code': 405,
        'message': '请求方法不允许',
        'data': None
    }), 405


if __name__ == '__main__':
    logger.info('正在启动AI投毒评论识别API服务...')
    logger.info(f'当前配置: USE_BASE_MODEL = {USE_BASE_MODEL}')
    logger.info('模型加载中...')
    
    try:
        test_result = predict("测试文本", use_base_model=USE_BASE_MODEL)
        model_type = test_result.get("model_type", "未知")
        logger.info(f'模型加载成功，使用: {model_type}')
    except Exception as e:
        logger.error(f'模型加载失败: {str(e)}')
        sys.exit(1)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )