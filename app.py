from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow import keras
import os
import base64
import json

app = Flask(__name__)

# 模型路径
MODEL_PATH = 'insect_cnn_model_best.h5'

# 英文类别名称（与训练时一致）
CLASS_NAMES_EN = ['ant', 'bee', 'butterfly', 'cicada', 'cricket', 'dragonfly', 'fly', 'ladybug', 'mosquito', 'tick']

# 中英文映射
CLASS_MAP = {
    'ant': '蚂蚁',
    'bee': '蜜蜂', 
    'butterfly': '蝴蝶',
    'cicada': '蝉',
    'cricket': '蟋蟀',
    'dragonfly': '蜻蜓',
    'fly': '苍蝇',
    'ladybug': '瓢虫',
    'mosquito': '蚊子',
    'tick': '蜱虫'
}

# 昆虫详细信息（中文）
INSECT_INFO = {
    '蚂蚁': {
        'name': '蚂蚁',
        'family': '蚁科',
        'feature': '体型小，触角膝状，有社会性',
        'habit': '杂食性，群体生活，分工明确',
        'toxicity': '一般无毒',
        'protectLevel': '非保护动物',
        'order': '膜翅目'
    },
    '蜜蜂': {
        'name': '蜜蜂',
        'family': '蜜蜂科',
        'feature': '身体被绒毛，后足有花粉篮',
        'habit': '群居，以花蜜和花粉为食',
        'toxicity': '尾部有刺，会蜇人',
        'protectLevel': '非保护动物',
        'order': '膜翅目'
    },
    '蝴蝶': {
        'name': '蝴蝶',
        'family': '凤蝶科',
        'feature': '翅膀有鳞粉、色彩鲜艳',
        'habit': '吸食花蜜',
        'toxicity': '无毒',
        'protectLevel': '非保护动物',
        'order': '鳞翅目'
    },
    '蝉': {
        'name': '蝉',
        'family': '蝉科',
        'feature': '体型较大，善于鸣叫',
        'habit': '以植物汁液为食',
        'toxicity': '无毒',
        'protectLevel': '非保护动物',
        'order': '半翅目'
    },
    '蟋蟀': {
        'name': '蟋蟀',
        'family': '蟋蟀科',
        'feature': '触角细长，善于跳跃',
        'habit': '夜间活动，杂食性',
        'toxicity': '无毒',
        'protectLevel': '非保护动物',
        'order': '直翅目'
    },
    '蜻蜓': {
        'name': '蜻蜓',
        'family': '蜻蜓科',
        'feature': '复眼大，翅膀透明',
        'habit': '捕食昆虫，飞行能力强',
        'toxicity': '无毒',
        'protectLevel': '非保护动物',
        'order': '蜻蜓目'
    },
    '苍蝇': {
        'name': '苍蝇',
        'family': '蝇科',
        'feature': '身体灰黑色，复眼大',
        'habit': '杂食性，传播疾病',
        'toxicity': '可能携带病菌',
        'protectLevel': '非保护动物',
        'order': '双翅目'
    },
    '瓢虫': {
        'name': '瓢虫',
        'family': '瓢甲科',
        'feature': '身体半球形，鞘翅有斑点',
        'habit': '捕食蚜虫等害虫',
        'toxicity': '无毒',
        'protectLevel': '非保护动物',
        'order': '鞘翅目'
    },
    '蚊子': {
        'name': '蚊子',
        'family': '蚊科',
        'feature': '体型小，细长',
        'habit': '雌蚊吸血，雄蚊吸花蜜',
        'toxicity': '可能传播疾病',
        'protectLevel': '非保护动物',
        'order': '双翅目'
    },
    '蜱虫': {
        'name': '蜱虫',
        'family': '蜱科',
        'feature': '体型小，椭圆形',
        'habit': '寄生在动物身上吸血',
        'toxicity': '可能传播疾病',
        'protectLevel': '非保护动物',
        'order': '寄螨目'
    }
}

model = None
loaded_model_path = None

def load_model():
    """加载训练好的昆虫识别模型"""
    global model, loaded_model_path
    if model is None:
        print(f"=== 开始加载模型 ===")
        
        # 尝试多个可能的模型文件
        possible_model_paths = [
            MODEL_PATH,
            'insect_cnn_model.h5',
            'insect_model.h5'
        ]
        
        found_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                found_path = path
                print(f"找到模型文件: {path}")
                break
        
        if found_path is None:
            print(f"错误：未找到模型文件")
            print("当前目录文件:", os.listdir('.'))
            raise FileNotFoundError(f"模型文件未找到，尝试的路径: {possible_model_paths}")

        try:
            model = keras.models.load_model(found_path)
            loaded_model_path = found_path
            print(f"模型加载成功: {found_path}")
            print(f"模型输入形状: {model.input_shape}")
            print(f"模型输出形状: {model.output_shape}")
            
            # 打印模型摘要
            model.summary()
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        print("=== 模型加载完成 ===")
    return model

def preprocess_image(image_data, target_size=(224, 224)):
    """预处理图片"""
    try:
        if isinstance(image_data, str):
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            image_data = base64.b64decode(image_data)

        img = Image.open(BytesIO(image_data))
        print(f"Original image mode: {img.mode}, size: {img.size}")

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(target_size)
        print(f"Resized image to: {img.size}")

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Final array shape: {img_array.shape}, range: [{img_array.min():.2f}, {img_array.max():.2f}]")
        return img_array

    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/health', methods=['GET'])
def health():
    try:
        load_model()
        return jsonify({
            'status': 'ok', 
            'model_loaded': model is not None,
            'loaded_model_path': loaded_model_path,
            'input_shape': str(model.input_shape) if model else None,
            'output_shape': str(model.output_shape) if model else None,
            'classes_count': len(CLASS_NAMES_EN)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_base64 = data.get('image')

        if not image_base64:
            return jsonify({'code': -1, 'msg': '缺少图片参数'}), 400

        print(f"Received image data length: {len(image_base64)}")

        img_array = preprocess_image(image_base64)
        model = load_model()

        predictions = model.predict(img_array, verbose=1)
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions: {predictions}")

        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class_en = CLASS_NAMES_EN[predicted_class_index]
        
        # 转换为中文
        predicted_name = CLASS_MAP.get(predicted_class_en, predicted_class_en)

        print(f"Predicted class (EN): {predicted_class_en}, (CN): {predicted_name}, confidence: {confidence:.4f}")

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        print("Top 3 predictions:")
        for idx in top_3_indices:
            cls_en = CLASS_NAMES_EN[idx]
            cls_cn = CLASS_MAP.get(cls_en, cls_en)
            print(f"  {cls_en} ({cls_cn}): {predictions[0][idx]:.4f}")

        insect_info = INSECT_INFO.get(predicted_name, {
            'name': predicted_name,
            'family': '未知',
            'feature': '未知',
            'habit': '未知',
            'toxicity': '未知',
            'protectLevel': '未知',
            'order': '未知'
        })
        insect_info['confidence'] = f"{confidence * 100:.1f}%"

        return jsonify({'code': 0, 'data': insect_info})

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'code': -1, 'msg': f'识别失败: {str(e)}'}), 500

print("Module loading: Pre-loading model...")
try:
    load_model()
    print("Module loading: Model pre-loading completed")
except Exception as e:
    print(f"Module loading: Model pre-loading failed: {str(e)}")

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'insect-inference'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)