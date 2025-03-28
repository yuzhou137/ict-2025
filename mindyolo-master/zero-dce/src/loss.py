# src/loss.py
import mindspore.ops as ops
from mindspore import nn


class ZeroReferenceLoss(nn.Cell):
    def __init__(self, alpha=0.8, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()

    def construct(self, enhanced, original, curve_params):
        """ 
        参数说明：
        - enhanced: 网络输出的增强图像 (B,3,H,W)
        - original: 输入的低光照图像 (B,3,H,W)
        - curve_params: 曲线参数张量 (B,num_iter,3,H,W)
        """
        # 空间一致性损失
        loss_spatial = self.l1_loss(enhanced, original)  # 直接比较输入输出

        # 曝光控制损失（计算16x16局部区域）
        patch_size = 16
        pooled = ops.avg_pool2d(enhanced, kernel_size=patch_size)
        loss_exposure = self.l1_loss(pooled, 0.6 * ops.ones_like(pooled))

        # 颜色恒常性损失（通道差异）
        diff_rg = enhanced[:, 0] - enhanced[:, 1]
        diff_gb = enhanced[:, 1] - enhanced[:, 2]
        loss_color = ops.mean(ops.abs(diff_rg) + ops.abs(diff_gb))

        # 曲线平滑损失（时间维度差分）
        delta = curve_params[:, 1:, :, :, :] - curve_params[:, :-1, :, :, :]
        loss_smooth = ops.mean(ops.abs(delta))

        total_loss = (loss_spatial +
                      self.alpha * loss_exposure +
                      loss_color +
                      self.gamma * loss_smooth)
        return total_loss