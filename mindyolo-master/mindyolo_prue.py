import argparse
import ast

import mindspore as ms
import numpy as np
from mindspore import Tensor, nn, ops
from demo.predict import set_default_infer
from mindyolo import SPPF, ConvNormAct
from mindyolo.models import create_model
from mindyolo.models.heads.yolov8_head import YOLOv8Head
from mindyolo.models.layers.bottleneck import Bottleneck, C2f
from mindyolo.utils.config import parse_args
from mindyolo.utils.utils import set_seed


def get_parser_infer(parents=None):  # 定义一个解析器,然而我自己也不太清楚每个参数的含义，可能有很多不必要的参数
    parser = argparse.ArgumentParser(description="Prune model", parents=[parents] if parents else [])
    parser.add_argument("--prune_rate", "--pr", type=float, default=0.2, help="prune rate,between 0 and 1")
    parser.add_argument("--device_target", type=str, default="CPU", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument(
        "--precision_mode", type=str, default=None, help="set accuracy mode of network model"
    )
    parser.add_argument("--weight", type=str, default="yolov8s.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--save_dir", type=str, default="./runs_infer", help="save dir")

    return parser


def _init_network(args):  # 初始化网络，通过模型文件加载
    set_seed(args.seed)
    set_default_infer(args)
    print(f"the nc is {args.data.nc}")
    network = create_model(  # 创建模型的参数统统来自于config文件，然而为什么要有一个parse_args函数呢？我也不知道
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    print(f"the nc is {args.data.nc}")
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)
    return network


def detach_like(tensor_obj):
    """
    因为mindspore的tensor没有detach功能，所以我只能创建一个函数来仿照
    """
    data = tensor_obj.asnumpy()
    dtype = tensor_obj.dtype
    new_tensor = Tensor(data, dtype)
    new_tensor.requires_grad = False
    return new_tensor

def get_indices_above_threshold(tensor, threshold):
    """获取张量中大于等于阈值的元素索引"""
    indices = []
    for i in range(tensor.shape[0]):
        if tensor[i] >= threshold:
            indices.append(i)
    return ms.Tensor(indices, dtype=ms.int32)


def prune_conv(conv_module, next_conv, threshold):
    """
    对卷积模块进行剪枝，通过替换参数实现形状变化
    """
    # 提取卷积层和BN层
    conv_layer = None
    bn_layer = None

    # 兼容不同模块类型的提取逻辑
    if isinstance(conv_module, nn.Conv2d):
        conv_layer = conv_module
        for sibling in conv_module.parent.cells():
            if isinstance(sibling, nn.BatchNorm2d):
                bn_layer = sibling
                break
    elif isinstance(conv_module, nn.BatchNorm2d):
        bn_layer = conv_module
        for sibling in conv_module.parent.cells():
            if isinstance(sibling, nn.Conv2d):
                conv_layer = sibling
                break
    elif hasattr(conv_module, 'conv') and hasattr(conv_module, 'bn'):
        conv_layer = conv_module.conv
        bn_layer = conv_module.bn

    if conv_layer is None or bn_layer is None:
        raise ValueError(f"无法从{type(conv_module).__name__}中提取卷积层和BN层")

    # 获取参数并计算保留索引
    gamma = bn_layer.gamma.asnumpy()
    keep_idxs = []
    local_threshold = threshold

    while len(keep_idxs) < 8:
        mask = (np.abs(gamma) >= local_threshold)
        keep_idxs = np.where(mask)[0]
        local_threshold *= 0.5

    n = len(keep_idxs)
    print(f"保留通道百分比: {n / len(gamma) * 100:.2f}%")

    # 定义参数替换函数
    def replace_param(module, param_name, new_data):
        """创建新参数并替换模块中的旧参数"""
        old_param = getattr(module, param_name)
        new_param = ms.Parameter(
            ms.Tensor(new_data, dtype=old_param.dtype),
            name=old_param.name
        )
        setattr(module, param_name, new_param)
        return new_param

    # 替换BN层参数
    bn_layer.gamma = replace_param(bn_layer, 'gamma', gamma[keep_idxs])
    bn_layer.beta = replace_param(bn_layer, 'beta', bn_layer.beta.asnumpy()[keep_idxs])
    bn_layer.moving_variance = replace_param(
        bn_layer, 'moving_variance', bn_layer.moving_variance.asnumpy()[keep_idxs]
    )
    bn_layer.moving_mean = replace_param(
        bn_layer, 'moving_mean', bn_layer.moving_mean.asnumpy()[keep_idxs]
    )

    # 替换卷积层参数
    conv_weight = conv_layer.weight.asnumpy()
    new_conv_weight = conv_weight[keep_idxs, :, :, :]
    conv_layer.weight = replace_param(conv_layer, 'weight', new_conv_weight)

    # 处理下一层卷积
    if not isinstance(next_conv, list):
        next_conv = [next_conv]

    for item in next_conv:
        if item is not None:
            next_conv_layer = item.conv if hasattr(item, 'conv') else item
            if isinstance(next_conv_layer, nn.Conv2d):
                next_weight = next_conv_layer.weight.asnumpy()
                new_next_weight = next_weight[:, keep_idxs, :, :]
                next_conv_layer.weight = replace_param(next_conv_layer, 'weight', new_next_weight)


def prune(m1, m2):
    if isinstance(m1, C2f):
        m1 = m1.cv2

    if not isinstance(m2, list):
        m2 = [m2]
    for i, item in enumerate(m2):
        if isinstance(item, C2f):
            m2[i] = item.cv1
        elif isinstance(item, SPPF):
            m2[i] = item.conv1

    prune_conv(m1, m2, threshold=threshold)

def get_all_submodules(model):
    """获取模型的第一层模块，返回列表"""
    third_level_modules = []

    # 遍历第一层子模块
    for first_level_child in model.cells():
        # 遍历第二层子模块
        for second_level_child in first_level_child.cells():
            # 遍历第三层子模块
            for third_level_child in second_level_child.cells():
                third_level_modules.append(third_level_child)

    return third_level_modules

save_path = "mindspore_yolov8s.ckpt"
args = parse_args(get_parser_infer())
model = _init_network(args)
param_dict = ms.load_checkpoint(args.weight)
param_not_load, _ = ms.load_param_into_net(model, param_dict)
print(param_not_load)
ws = []
bs = []

# print(param_dict.items())
# 找到模型中的BN层，提取权重与偏置
for name, m in param_dict.items():
    if 'bn.gamma' in name:
        w = detach_like(m.abs())
        ws.append(w)
        # print(f"the weight shape is {w.shape}")
    elif 'bn.beta' in name:
        b = detach_like(m.abs())
        bs.append(b)
        # print(f"the bias shape is {b.shape}")

# 保留率，1-%剪枝率
factor = 1 - args.prune_rate
ws = ms.ops.cat(ws, axis=0)

# 选取阈值，大于阈值的权重对应的通道将被剪枝，看见了吗？factor
threshold = Tensor.sort(ws, descending=True)[0][int(len(ws) * factor)]
print('threshold: {:.10f}'.format(threshold.asnumpy().item()))
bs = ms.ops.cat(bs, axis=0)

# 1.剪枝c2f中的bottleneck层
for name, m in param_dict.items():
    if isinstance(m, Bottleneck):
        prune_conv(m.cv1, m.cv2, threshold=threshold)

# 2.指定剪枝不同模块之间的卷积核
seq = get_all_submodules(model)
print(f"子模块数量: {len(seq)}")
print("前20个子模块:")
for i, module in enumerate(seq[:20]):
    print(f"{i}: {type(module).__name__}")
for i in range(3, 9):
    if i in [6, 4, 9]: continue
    prune(seq[i], seq[i + 1])

# 3.对检测头进行剪枝
detect: YOLOv8Head = seq[-1]
last_inputs = [seq[15], seq[18], seq[21]]
colasts = [seq[16], seq[19], None]
for last_inputs, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
    prune(last_inputs, [colast, cv2[0], cv3[0]])
    prune(cv2[0], cv2[1])
    prune(cv2[1], cv2[2])
    prune(cv3[0], cv3[1])
    prune(cv3[1], cv3[2])

for p in model.trainable_params():
    p.requires_grad = True

# 保存模型
ms.save_checkpoint(model.parameters_dict(), save_path)
print(f"save model to {save_path}")
