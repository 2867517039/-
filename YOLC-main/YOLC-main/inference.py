import sys
import os.path as osp
import cv2
import mmcv
import torch
import numpy as np

# 添加项目根目录到 Python 路径
project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mmdet.apis import init_detector
from inference_YOLC import inference_detector, inference_detector_with_LSM
import os

# 只需要导入 models，它会自动注册所有必要的模型
from models import *

def preprocess_image(img_path, target_size=(1920, 1080)):
    """预处理图像到指定尺寸"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # 直接调整到目标尺寸，不保持长宽比
    img_resized = cv2.resize(img, target_size)
    
    # 转换为 float32 并归一化到 [0-1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # BGR to RGB
    img_normalized = img_normalized[:, :, ::-1]
    
    # 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    
    # 转换回 uint8 用于保存
    img_uint8 = cv2.normalize(img_normalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return img_uint8

def process_detection_result(result):
    """处理检测结果，确保返回正确的格式"""
    if isinstance(result, tuple):
        # 如果结果是元组，获取检测框部分
        cluster_region, det_boxes = result
        if isinstance(det_boxes, list):
            # 如果是列表，转换为numpy数组
            det_boxes = [np.array(boxes) if boxes else np.zeros((0, 5)) for boxes in det_boxes]
        return cluster_region, det_boxes
    return None, result

# 配置文件和模型权重文件
config_file = '/mnt/home/hks/平铺/Y+S/YOLC-main/YOLC-main/configs/yolc.py'
checkpoint_file = '/mnt/home/hks/平铺/Y+S/work_dir/yolc_sparsevit/1.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 读取图像文件夹
img_folder = '/mnt/home/hks/平铺/Y+S/YOLC-main/YOLC-main/data/0.5'
out_folder = '/mnt/home/hks/平铺/Y+S/YOLC-main/YOLC-main/output/0.5'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

img_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) 
             if f.endswith(('.jpg', '.png', '.jpeg'))]

# 使用YOLC的推理函数处理图像
for img_file in img_files:
    try:
        # 预处理图像
        processed_img = preprocess_image(img_file)
        
        # 保存处理后的图像到临时文件
        temp_img_path = os.path.join(out_folder, 'temp_' + os.path.basename(img_file))
        cv2.imwrite(temp_img_path, processed_img)
        
        # 打印调试信息
        print(f"Processing {img_file}")
        print(f"Processed image shape: {processed_img.shape}")
        
        # 使用YOLC专用的推理函数
        try:
            result = inference_detector_with_LSM(model, temp_img_path)
            cluster_region, det_boxes = process_detection_result(result)
            
            # 读取原始图像用于可视化
            img = cv2.imread(img_file)
            
            # 构造输出文件路径
            out_file = os.path.join(out_folder, os.path.basename(img_file))
            
            # 可视化结果并保存
            if det_boxes is not None:
                # 绘制检测框
                model.show_result(img, det_boxes, out_file=out_file, show=False)
                
                # 如果有聚类区域，也绘制它们
                if cluster_region is not None:
                    img_with_boxes = cv2.imread(out_file)
                    for box in cluster_region:
                        x, y, w, h = map(int, box)
                        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imwrite(out_file, img_with_boxes)
            
            print(f"Successfully processed {img_file}")
            print(f"Found {sum(len(boxes) for boxes in det_boxes) if det_boxes else 0} objects")
            
        except Exception as e:
            print(f"Model inference error: {str(e)}")
            continue
        
        # 删除临时文件
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")
        print(f"Image shape: {cv2.imread(img_file).shape}")
        import traceback
        print(traceback.format_exc())
        continue

print("Processing completed!")