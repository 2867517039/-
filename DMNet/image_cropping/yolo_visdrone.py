# import os

# # 定义YOLO类别ID到VisDrone类别ID的映射
# yolo_to_visdrone_mapping = {
#     0: 1,
#     1: 2,  # 示例：YOLO类别0对应VisDrone类别1
#     # ... 添加其他映射 ...
# }

# def convert_yolo_to_visdrone(yolo_ann, img_width, img_height, yolo_to_visdrone_mapping):
#     visdrone_anns = []
#     for ann in yolo_ann:
#         class_id, x_center, y_center, width, height = map(float, ann.split())
        
#         # 将YOLO的归一化坐标转换为绝对像素坐标
#         bbox_left = int((x_center - width / 2) * img_width)
#         bbox_top = int((y_center - height / 2) * img_height)
#         bbox_width = int(width * img_width)
#         bbox_height = int(height * img_height)

#         # 使用映射字典来获取VisDrone类别ID
#         object_category = yolo_to_visdrone_mapping.get(int(class_id), -1)  # 如果没有映射，设置为-1

#         # VisDrone格式需要的其他字段，这里暂时使用默认值
#         score = 1  # 假设所有的标注都有最高的置信度
#         truncation = 0  # 假设没有截断
#         occlusion = 0  # 假设没有遮挡

#         # 创建VisDrone格式的标注
#         visdrone_ann = f"{bbox_left},{bbox_top},{bbox_width},{bbox_height},{score},{object_category},{truncation},{occlusion}"
#         visdrone_anns.append(visdrone_ann)
#     return visdrone_anns

# def process_yolo_folder(yolodir, outputdir, img_width, img_height, yolo_to_visdrone_mapping):
#     for filename in os.listdir(yolodir):
#         if filename.endswith('.txt'):  # 确保处理的是标注文件
#             yolo_ann_file = os.path.join(yolodir, filename)
#             with open(yolo_ann_file, 'r') as file:
#                 yolo_ann = file.readlines()

#             # 移除每行末尾的换行符并转换为VisDrone格式
#             yolo_ann = [ann.strip() for ann in yolo_ann]
#             visdrone_anns = convert_yolo_to_visdrone(yolo_ann, img_width, img_height, yolo_to_visdrone_mapping)

#             # 保存转换后的标注到输出文件夹
#             output_filename = os.path.join(outputdir, filename)
#             with open(output_filename, 'w') as f:
#                 for ann in visdrone_anns:
#                     f.write(ann + '\n')

# # 指定YOLO标注文件夹的目录
# yolo_dir = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/image_cropping/dataset/test/Annotations'  # 替换为你的YOLO标注文件夹路径
# output_dir = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/image_cropping/dataset/test/annotations'  # 替换为转换后VisDrone标注的输出文件夹路径
# img_width = 3840  # 替换为你的图像宽度
# img_height = 2160  # 替换为你的图像高度

# # 处理整个YOLO标注文件夹
# process_yolo_folder(yolo_dir, output_dir, img_width, img_height, yolo_to_visdrone_mapping)
import os
from PIL import Image

# 定义YOLO类别ID到VisDrone类别ID的映射
yolo_to_visdrone_mapping = {
    0: 1,
    1: 2,
    # ... 添加其他映射 ...
}

def convert_yolo_to_visdrone(yolo_ann, img_width, img_height, yolo_to_visdrone_mapping):
    visdrone_anns = []
    for ann in yolo_ann:
        class_id, x_center, y_center, width, height = map(float, ann.split())
        
        # 将YOLO的归一化坐标转换为绝对像素坐标
        bbox_left = int((x_center - width / 2) * img_width)
        bbox_top = int((y_center - height / 2) * img_height)
        bbox_width = int(width * img_width)
        bbox_height = int(height * img_height)

        # 使用映射字典来获取VisDrone类别ID
        object_category = yolo_to_visdrone_mapping.get(int(class_id), -1)  # 如果没有映射，设置为-1

        # VisDrone格式需要的其他字段，这里暂时使用默认值
        score = 1  # 假设所有的标注都有最高的置信度
        truncation = 0  # 假设没有截断
        occlusion = 0  # 假设没有遮挡

        # 创建VisDrone格式的标注
        visdrone_ann = f"{bbox_left},{bbox_top},{bbox_width},{bbox_height},{score},{object_category},{truncation},{occlusion}"
        visdrone_anns.append(visdrone_ann)
    return visdrone_anns
def process_yolo_folder(yolodir, imgdir, outputdir, yolo_to_visdrone_mapping):
    for filename in os.listdir(yolodir):
        if filename.endswith('.txt'):  # 确保处理的是标注文件
            yolo_ann_file = os.path.join(yolodir, filename)
            with open(yolo_ann_file, 'r') as file:
                yolo_ann = file.readlines()

            # 获取与标注文件对应的图像文件名
            img_filename = filename.replace('.txt', '.jpg')  # 假设图像文件是jpg格式
            img_path = os.path.join(imgdir, img_filename)
            img = Image.open(img_path)
            img_width, img_height = img.size

            # 移除每行末尾的换行符并转换为VisDrone格式
            yolo_ann = [ann.strip() for ann in yolo_ann]
            visdrone_anns = convert_yolo_to_visdrone(yolo_ann, img_width, img_height, yolo_to_visdrone_mapping)

            # 保存转换后的标注到输出文件夹
            output_filename = os.path.join(outputdir, filename)
            with open(output_filename, 'w') as f:
                for ann in visdrone_anns:
                    f.write(ann + '\n')

# 指定YOLO标注文件夹的目录
yolo_dir = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/fusion_detection/output2/train/annotations'  # 替换为你的YOLO标注文件夹路径
img_dir = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/fusion_detection/output2/train/images'  # 图像文件夹路径
output_dir = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/fusion_detection/output2/train/1'  # 替换为转换后VisDrone标注的输出文件夹路径

# 处理整个YOLO标注文件夹
process_yolo_folder(yolo_dir, img_dir, output_dir, yolo_to_visdrone_mapping)