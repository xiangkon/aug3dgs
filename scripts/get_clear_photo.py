import cv2
import os
import numpy as np

def calculate_laplacian_variance(image_path):
    """
    计算图片的拉普拉斯算子方差
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图片
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    laplacian_variance = cv2.Laplacian(image, cv2.CV_64F).var()  # 计算拉普拉斯方差
    return laplacian_variance

def filter_clear_images(input_folder, output_folder, threshold=40):
    """
    筛选清晰图片并保存到指定文件夹
    :param input_folder: 包含所有图片的文件夹路径
    :param output_folder: 保存清晰图片的文件夹路径
    :param threshold: 拉普拉斯方差的阈值，高于此值的图片被认为是清晰的
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 如果输出文件夹不存在，则创建

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                laplacian_variance = calculate_laplacian_variance(file_path)
                print(f"图片 {filename} 的拉普拉斯方差为：{laplacian_variance:.2f}")
                if laplacian_variance > threshold:
                    # 如果图片清晰，复制到输出文件夹
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, cv2.imread(file_path))
                    print(f"清晰图片 {filename} 已保存到 {output_folder}")
            except Exception as e:
                print(f"处理图片 {filename} 时出错：{e}")

# 示例用法
input_folder = "/home/admin123/ssd/Xiangkon/TDGS/videos/piper76/photo"  # 替换为你的输入文件夹路径
output_folder = "/home/admin123/ssd/Xiangkon/TDGS/videos/piper76/input"  # 替换为你的输出文件夹路径

if not os.path.exists(output_folder):
    os.makedirs

filter_clear_images(input_folder, output_folder)