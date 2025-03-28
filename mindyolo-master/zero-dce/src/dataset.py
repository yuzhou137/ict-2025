from mindspore.dataset import GeneratorDataset
import cv2
import numpy as np


class SingleImageLoader:
    """单图像随机裁剪生成器"""

    def __init__(self, img_path, patch_size=256):
        self.img = cv2.imread(img_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) / 255.0
        self.H, self.W = self.img.shape[:2]
        self.patch_size = patch_size

    def __getitem__(self, index):
        # 随机裁剪坐标
        top = np.random.randint(0, self.H - self.patch_size)
        left = np.random.randint(0, self.W - self.patch_size)
        patch = self.img[top:top + self.patch_size, left:left + self.patch_size]
        return patch.transpose(2, 0, 1).astype(np.float32)  # HWC->CHW

    def __len__(self):
        return 1000  # 虚拟长度，每个epoch采样1000次


def create_unsupervised_dataset(img_path, batch_size=8):
    ds = GeneratorDataset(
        source=SingleImageLoader(img_path),
        column_names=["data"],  # 无标签列
        shuffle=True
    )
    # 添加动态增强
    transforms = [
        lambda x: x + np.random.normal(0, 0.02, x.shape),  # 噪声注入
        lambda x: np.clip(x, 0, 1)  # 确保数值范围
    ]
    ds = ds.map(operations=transforms, input_columns=["data"])
    return ds.batch(batch_size)