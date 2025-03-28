from mindspore import Model, context, nn
from mindspore.train.callback import LossMonitor

from src.loss import ZeroReferenceLoss
from src.model import ZeroDCE


class UnsupervisedTrainer:
    def __init__(self, config):
        self.net = ZeroDCE(num_iter=config.num_iter)
        self.loss_net = self._build_loss_net()
        self.opt = nn.Adam(self.net.trainable_params(), config.lr)
        self.model = Model(self.loss_net, optimizer=self.opt)

    def _build_loss_net(self):
        """构建包含损失计算的计算图"""
        class WithLoss(nn.Cell):
            def __init__(self, net, loss_fn):
                super().__init__()
                self.net = net
                self.loss_fn = loss_fn
            def construct(self, x):
                enhanced = self.net(x)
                return self.loss_fn(enhanced, x, self.net.curve_params)
        return WithLoss(self.net, ZeroReferenceLoss())

    def train(self, dataset):
        self.model.train(
            epoch=config.epochs,
            train_dataset=dataset,
            callbacks=[LossMonitor(per_print_times=10)],
            dataset_sink_mode=True  # 启用数据下沉加速
        )