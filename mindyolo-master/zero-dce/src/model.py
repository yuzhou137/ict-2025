import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P


class ZeroDCE(nn.Cell):
    """无监督版网络结构"""

    def __init__(self, num_iter=8):
        super().__init__()
        # 特征提取层（轻量级设计）
        self.encoder = nn.SequentialCell([
            nn.Conv2d(3, 32, kernel_size=3, pad_mode='same', has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, pad_mode='same'),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, pad_mode='same'),
        ])
        # 曲线参数预测（输出通道数为 3*num_iter）
        self.curve_params = nn.Conv2d(128, 3 * num_iter, kernel_size=3, pad_mode='same')
        # 曲线应用模块
        self.num_iter = num_iter
        self.add = P.Add()  # MindSpore算子需显式定义

    def construct(self, x):
        features = self.encoder(x)
        params = self.curve_params(features)
        B, C, H, W = params.shape
        params = params.view(B, self.num_iter, 3, H, W)  # 分离迭代次数与通道

        # 迭代应用曲线调整
        for i in range(self.num_iter):
            alpha = params[:, i, :, :, :]
            x = self.add(x, (x - x * x) * alpha)  # 核心公式：L' = L + L(1-L)*α
        return mindspore.ops.clip(x, 0.0, 1.0)  # 限制输出范围