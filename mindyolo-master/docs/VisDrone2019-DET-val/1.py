import os
from PIL import Image

# 定义image文件夹路径
image_folder = 'images'

# 遍历image文件夹中的所有文件
for filename in os.listdir(image_folder):
    # 检查文件是否为jpg文件
    if filename.endswith('.jpg'):
        # 构建完整的文件路径
        jpg_path = os.path.join(image_folder, filename)
        
        # 打开jpg图像
        with Image.open(jpg_path) as img:
            # 构建png文件路径（保持文件名不变，只更改扩展名）
            png_path = os.path.splitext(jpg_path)[0] + '.png'
            
            # 将图像保存为png格式
            img.save(png_path, 'PNG')
        
        # 删除原始的jpg文件
        os.remove(jpg_path)

print("转换完成！")
