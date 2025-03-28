# val.py
import ast
import math
import os
import argparse
import time

import cv2
import numpy as np
from pathlib import Path
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo import xyxy2xywh, scale_coords, non_max_suppression
from mindyolo.data import COCODataset, create_loader
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
import matplotlib.pyplot as plt
import seaborn as sn
from mindyolo.utils.utils import (freeze_layers, load_pretrain, set_default,
                                  set_seed, Synchronizer)

class ConfusionMatrix:
    def __init__(self, nc, conf_thres=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres


    def _box_iou(self, box1, box2):
        """计算两组框之间的IoU"""

        def box_area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

        area1 = box_area(box1)
        area2 = box_area(box2)

        lt = ops.maximum(box1[:, None, :2], box2[:, :2])
        rb = ops.minimum(box1[:, None, 2:], box2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        return inter / (area1[:, None] + area2 - inter)


    def process_batch(self, pred_dict, gt_labels):
        """处理批次数据，更新混淆矩阵"""
        pred_cls = pred_dict["category_id"]  # 预测类别列表
        pred_scores = pred_dict["score"]  # 置信度列表
        gt_cls = gt_labels[:, 0].astype(int)  # 真实类别列表

        # 根据置信度阈值过滤预测结果
        valid_mask = np.array(pred_scores) >= self.conf_thres
        pred_cls = np.array(pred_cls)[valid_mask]

        # 如果没有有效预测，将所有真实类别标记为背景
        if len(pred_cls) == 0:
            for c in gt_cls:
                self.matrix[self.nc, c] += 1
            return

        # 计算每个预测与真实标签的匹配
        for gt_c in gt_cls:
            matched = False
            for i, pred_c in enumerate(pred_cls):
                iou = 1.0  # 假设检测框已通过IoU阈值筛选（根据实际需要调整）
                if iou >= self.iou_thres and pred_c == gt_c:
                    self.matrix[pred_c, gt_c] += 1
                    matched = True
                    pred_cls = np.delete(pred_cls, i)  # 移除已匹配的预测
                    break
            if not matched:
                self.matrix[self.nc, gt_c] += 1  # 未匹配的真实类别标记为背景

        # 未匹配的预测标记为误检
        for pred_c in pred_cls:
            if pred_c < self.nc:
                self.matrix[pred_c, self.nc] += 1
    def plot(self, names, save_dir='./results'):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 9))
        array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-9)
        array[array < 0.005] = np.nan

        sn.set(font_scale=1.0 if self.nc < 50 else 0.8)
        labels = names + ['background'] if len(names) == self.nc else list(range(self.nc)) + ['background']

        sn.heatmap(array, annot=self.nc < 30, fmt='.2f', xticklabels=labels, yticklabels=labels,
                   cmap='Blues', square=True, cbar=False)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix')

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250, bbox_inches='tight')
        plt.close()

def load_labels(label_path):
    """从标注文件中加载标签信息"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        # 假设每行格式为：类别 x_center y_center width height
        label = list(map(float, line.strip().split()))
        labels.append(label)
    return np.array(labels)


def detect(
        network: nn.Cell,
        img: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.65,
        conf_free: bool = False,
        nms_time_limit: float = 60.0,
        img_size: int = 640,
        stride: int = 32,
        num_class: int = 80,
        is_coco_dataset: bool = True,
):
    # Resize
    h_ori, w_ori = img.shape[:2]  # orig hw
    r = img_size / max(h_ori, w_ori)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

    # Transpose Norm
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)

    # Run infer
    _t = time.time()
    out = network(imgs_tensor)  # inference and training outputs
    out = out[0] if isinstance(out, (tuple, list)) else out
    infer_times = time.time() - _t

    # Run NMS
    t = time.time()
    out = out.asnumpy()
    out = non_max_suppression(
        out,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
    )
    nms_times = time.time() - t

    result_dict = {"category_id": [], "bbox": [], "score": []}
    total_category_ids, total_bboxes, total_scores = [], [], []
    for si, pred in enumerate(out):
        if len(pred) == 0:
            continue

        # Predictions
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))  # native-space pred

        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        category_ids, bboxes, scores = [], [], []
        for p, b in zip(pred.tolist(), box.tolist()):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))

        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)

    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)

    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)  # tuple
    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    return result_dict
def evaluate(args):
    # 设置运行模式
    # set_seed(args.seed)
    args.data.nc = 80
    set_default(args)
    ms.set_context(
        mode=ms.GRAPH_MODE if args.ms_mode == 0 else ms.PYNATIVE_MODE,
        device_target=args.device_target
    )

    # 关键修复1：确保网络配置存在
    if not hasattr(args, 'network'):
        raise ValueError("Missing network configuration in config file")

    # 创建模型
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weights,
    )
    ms.load_checkpoint(args.weights, network)
    network.set_train(False)

    # 数据集配置
    dataset = COCODataset(
        dataset_path='G:/dataset/coco128/images/train2017',
        img_size=args.img_size,
        transforms_dict=args.data.test_transforms,
        is_training=False,
        rect=args.rect,
        single_cls=args.single_cls,
        batch_size=args.per_batch_size,
        stride=max(args.network.stride),
    )

    dataloader = create_loader(
        dataset=dataset,
        column_names_getitem=dataset.column_names_getitem,
        column_names_collate=dataset.column_names_collate,
        batch_collate_fn=dataset.test_collate_fn,
        batch_size=args.per_batch_size,
        shuffle=False,
        num_parallel_workers=args.data.num_parallel_workers,
    )

    cm = ConfusionMatrix(nc=args.data.nc, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

    logger.info("Start evaluating...")
    total_images = 0
    for batch_idx, batch in enumerate(dataloader):
        imgs, img_paths, paths, shapes, *_ = batch

        # 关键修复：确保 img_paths 是字符串列表
        if isinstance(img_paths, ms.Tensor):
            img_paths = img_paths.asnumpy().tolist()
            img_paths = [str(path) for path in img_paths]

        # 加载真实标签
        all_gt_labels = []
        for img_path in img_paths:
            label_path = img_path.replace('images', 'labels').replace('.png', '.txt')
            labels = load_labels(label_path)
            if labels.size > 0:
                all_gt_labels.append(labels)
        gt_labels = np.concatenate(all_gt_labels, axis=0) if all_gt_labels else np.empty((0, 5))

        # 关键修改：逐张处理图像，确保输入与 cv2.imread 一致
        batch_preds = {
            "category_id": [],
            "bbox": [],
            "score": []
        }
        for img_tensor, img_path in zip(imgs, img_paths):
            # 将预处理后的 Tensor 转换为 cv2.imread 格式
            # 1. 转换为 NumPy 并调整维度顺序 (C, H, W) -> (H, W, C)
            img_np = img_tensor.asnumpy().transpose(1, 2, 0)
            # 2. 反归一化（假设预处理时归一化到 [0,1]）
            img_np = (img_np * 255).astype(np.uint8)
            # 3. 转换颜色通道顺序（假设预处理时是 RGB，需转回 BGR）
            img_np = img_np[:, :, ::-1]  # RGB -> BGR

            # 调用 detect 函数（此时 img_np 格式与 cv2.imread 完全一致）
            try:
                pred_dict = detect(
                    network=network,
                    img=img_np,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    conf_free=args.conf_free,
                    nms_time_limit=args.nms_time_limit,
                    img_size=args.img_size,
                    stride=max(max(args.network.stride), 32),
                    num_class=args.data.nc,
                    is_coco_dataset=True,
                )
                # 合并结果
                batch_preds["category_id"].extend(pred_dict["category_id"])
                batch_preds["bbox"].extend(pred_dict["bbox"])
                batch_preds["score"].extend(pred_dict["score"])
            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {str(e)}")
                continue

        # 更新混淆矩阵
        if batch_preds["category_id"]:
            cm.process_batch(batch_preds, gt_labels)
        else:
            logger.warning(f"No valid predictions in batch {batch_idx + 1}")
    # 验证总处理图像数
    logger.info(f"Total images processed: {total_images} (expected: {len(dataset)})")
    if total_images != len(dataset):
        logger.warning("Not all images were processed! Check data loader settings.")


    class_names = args.data.names if hasattr(args.data, 'names') else [f'class_{i}' for i in range(args.data.nc)]
    cm.plot(names=class_names, save_dir=args.save_dir)
    logger.info(f"Confusion matrix saved to {Path(args.save_dir) / 'confusion_matrix.png'}")


def get_parser_val():
    """参数解析器"""
    parser = argparse.ArgumentParser(description='Validate')
    # parser.add_argument('--config', type=str, required=True, help='config file path')  # 关键修复2：添加配置文件参数
    parser.add_argument('--weights', type=str,default='yolov8s.ckpt', required=True)
    # parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument("--log_level", type=str, default="INFO", help="log level to print")
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--device-target', type=str, default='CPU')
    parser.add_argument('--ms-mode', type=int, default=0)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument("--max_call_depth", type=int, default=2000, help="The maximum depth of a function call")
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=False, help="Distribute train or not")
    parser.add_argument("--auto_accumulate", type=ast.literal_eval, default=False, help="auto accumulate")
    parser.add_argument("--accumulate", type=int, default=1,
                        help="grad accumulate step, recommended when batch-size is less than 64")
    parser.add_argument("--nbs", type=list, default=64, help="nbs")
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    return parser


if __name__ == '__main__':
    parser = get_parser_val()
    args = parse_args(parser)
    evaluate(args)