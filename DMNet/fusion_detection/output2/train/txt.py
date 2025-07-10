import os

def remove_second_number_from_lines(txt_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    with open(txt_file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            if len(parts) > 1:
                # 移除第二个元素，然后重新组合行
                parts = parts[0:1] + parts[2:]
                # 确保行尾的换行符也被写入
                file.write(' '.join(parts) + '\n')
            else:
                # 如果行中元素少于两个，直接写入
                file.write(line)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # 确保处理的是txt文件
            txt_file_path = os.path.join(folder_path, filename)
            remove_second_number_from_lines(txt_file_path)
            print(f"Processed {txt_file_path}")

# 指定要处理的文件夹路径
folder_path = '/mnt/home/hks/平铺/DMNet-master/DMNet-master/fusion_detection/output2/train/annotations'  # 替换为你的文件夹路径
process_folder(folder_path)