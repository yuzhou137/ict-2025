import argparse
import ast
import math
import os
import sys
import time
from datetime import datetime

import cv2
import mindspore as ms
import numpy as np
import yaml
from mindspore import Tensor, nn
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import  load_config, Config
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh, process_mask_upsample, scale_image
from mindyolo.utils.utils import draw_result2, set_seed
from mindyolo.utils.config import parse_args

def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"])
    parser.add_argument("--device_target", type=str, default="CPU", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    # 添加以下参数
    parser.add_argument("--video_path", type=str, default="0", help="camera")
    parser.add_argument("--output_video", type=str, default="runs_infer/output.avi", help="The output path")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument(
        "--precision_mode", type=str, default=None, help="set accuracy mode of network model"
    )
    parser.add_argument("--weight", type=str, default="../yolov8s141.ckpt", help="model.ckpt path(s)")
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

    parser.add_argument("--image_path", type=str, help="path to image")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")

    return parser


def init_network(args):
    # 初始化模型（仅执行一次）
    set_seed(args.seed)
    set_default_infer(args)
    print(f"the nc is {args.data.nc}")
    network = create_model(
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


def set_default_infer(args):
    # Set Context
    ms.set_context(mode=args.ms_mode, device_target=args.device_target, max_call_depth=2000)
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if args.device_target == "Ascend":
        ms.set_context(device_id=int(os.getenv("DEVICE_ID", 0)))
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        ms.set_context(enable_graph_kernel=True)
    args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Directories and Save run settings
    platform = sys.platform
    if platform == "win32":
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    else:
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))


def process_frame(network, args, frame):
    # 临时保存帧为图片（或直接处理内存中的图像）
    # temp_image_path = os.path.join(args.save_dir, "temp_frame.jpg")
    # cv2.imwrite(temp_image_path, frame)
    # args.image_path = temp_image_path  #很好，我觉得应该直接调用内存的图片，当然我这draw_results的代码也改了，所以最好使用我改的draw_results
                                         #当然你要问，额，我这下的库怎么办呢？答案是，重构

    # 执行检测
    is_coco_dataset = "coco" in args.data.dataset_name
    if args.task == "detect":
        result_dict = detect(
            network=network,
            img=frame,  # 读取临时文件
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=False,
        )
    elif args.task == "segment":
        result_dict = segment(
            network=network,
            img=frame,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=False,
        )

    # 绘制结果
    result_image = draw_result2(
        frame, result_dict, args.data.names,
        is_coco_dataset=False, save_path=None  # 不保存到文件，直接返回图像
    )
    return result_image


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
    print(type(result_dict['score']))
    print(type(result_dict['category_id']))
    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)  # tuple
    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    return result_dict


# def draw_result(image_path, result_dict, class_names, is_coco_dataset=False, save_path=None):
#     img = cv2.imread(image_path)
#     for box, cls_id, score in zip(result_dict['bbox'], result_dict['category_id'], result_dict['score']):
#         x, y, w, h = [int(v) for v in box]
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         label = f"{class_names[cls_id]}: {score:.2f}"
#         cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     if save_path:
#         cv2.imwrite(save_path, img)
#     return img  # 返回绘制后的图像


def segment(
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
    out, (_, _, prototypes) = network(imgs_tensor)  # inference and training outputs
    infer_times = time.time() - _t

    # Run NMS
    t = time.time()
    _c = num_class + 4 if conf_free else num_class + 5
    out = out.asnumpy()
    bboxes, mask_coefficient = out[:, :, :_c], out[:, :, _c:]
    out = non_max_suppression(
        bboxes,
        mask_coefficient,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
    )
    nms_times = time.time() - t

    prototypes = prototypes.asnumpy()

    result_dict = {"category_id": [], "bbox": [], "score": [], "segmentation": []}
    total_category_ids, total_bboxes, total_scores, total_seg = [], [], [], []
    for si, (pred, proto) in enumerate(zip(out, prototypes)):
        if len(pred) == 0:
            continue

        # Predictions
        pred_masks = process_mask_upsample(proto, pred[:, 6:], pred[:, :4], shape=imgs_tensor[si].shape[1:])
        pred_masks = pred_masks.astype(np.float32)
        pred_masks = scale_image((pred_masks.transpose(1, 2, 0)), (h_ori, w_ori))
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))  # native-space pred

        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        category_ids, bboxes, scores, segs = [], [], [], []
        for ii, (p, b) in enumerate(zip(pred.tolist(), box.tolist())):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))
            segs.append(pred_masks[:, :, ii])

        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)
        total_seg.extend(segs)

    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)
    result_dict["segmentation"].extend(total_seg)

    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)  # tuple
    logger.info(f"Predict result is:")
    for k, v in result_dict.items():
        if k == "segmentation":
            logger.info(f"{k} shape: {v[0].shape}")
        else:
            logger.info(f"{k}: {v}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    return result_dict


def infer(args):
    # 生成时间戳文件夹名（格式：2025.03.21-22.51.15）
    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    save_dir = os.path.join(args.save_dir, timestamp)  # 例如：./runs_infer/2025.03.21-22.51.15
    os.makedirs(save_dir, exist_ok=True)

    # 初始化模型
    network = init_network(args)

    # 打开摄像头或视频文件
    cap = cv2.VideoCapture(args.video_path if args.video_path != "0" else 0)
    if not cap.isOpened():
        raise ValueError("无法打开视频源")

    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 设置输出视频路径（保存在时间戳文件夹下）
    output_video_path = os.path.join(save_dir, "output.avi")  # 例如：./runs_infer/2025.03.21-22.51.15/output.avi

    # 初始化视频写入器（使用兼容性编码）
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或 'mp4v' 用于 .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 处理当前帧
            processed_frame = process_frame(network, args, frame)

            # 强制统一分辨率（避免写入失败）
            if (processed_frame.shape[1], processed_frame.shape[0]) != (frame_width, frame_height):
                processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

            # 写入输出视频
            out.write(processed_frame)

            # 显示实时结果
            cv2.imshow("Real-time Detection", processed_frame)
            end_time = time.time()
            infer_time = end_time-start_time
            print(infer_time)
            print(".1")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"视频已保存至: {os.path.abspath(output_video_path)}")


if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)
    if not hasattr(args, 'data'):
        args.data = argparse.Namespace()
    if not hasattr(args.data, 'nc'):
        # args.data.nc = 1 if args.single_cls else 10  # 我的数据集的数量
        args.data.nc = 1 if args.single_cls else 80
    if not hasattr(args.data, 'names'):
        # args.data.names = ["item"] if args.single_cls else [ 'pedestrian', 'people', 'bicycle',
        #                                                      'car', 'van', 'truck', 'tricycle',
        #                                                      'awning-tricycle', 'bus', 'motor'] #Visdrone的类别名称 只能说是石山代码了还得我手动写这些
        args.data.names = ["item"] if args.single_cls else [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush' ]
    if not hasattr(args.data, 'dataset_name'):
        args.data.dataset_name = "VisDrone"
    infer(args)
