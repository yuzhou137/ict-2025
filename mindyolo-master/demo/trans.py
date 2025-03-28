from PIL import Image


def convert_jpg_to_png(jpg_file_path, png_file_path):
    try:
        # 打开JPG文件
        image = Image.open(jpg_file_path)

        # 转换为PNG并保存
        image.save(png_file_path, "PNG")
        print(f"文件已成功转换为 {png_file_path}")
    except Exception as e:
        print(f"转换失败: {e}")


# 示例用法
jpg_file = "try.jpg"  # 输入的JPG文件路径
png_file = "output_image.png"  # 输出的PNG文件路径

convert_jpg_to_png(jpg_file, png_file)