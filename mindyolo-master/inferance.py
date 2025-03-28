import cv2
import numpy as np
from ais_bench.infer.interface import InferSession

# 初始化摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# 加载 OM 模型
model_path = "yolov8_ascend.om"
session = InferSession(device_id=0, model_path=model_path)
input_shape = session.get_inputs()[0].shape  # [1, 3, 640, 640]

# 滑动窗口参数
window_size = (640, 640)
stride = 320  # 根据 COCO 物体平均尺寸（~200x200）设定


def sliding_window(image, window_size=(640, 640), stride=320):
    height, width = image.shape[:2]
    windows = []
    positions = []  # 记录每个窗口的左上角坐标 (x1, y1)

    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            windows.append(window)
            positions.append((x, y))

    # 处理右侧和下侧边缘未覆盖的区域
    if (height - window_size[1]) % stride != 0:
        y = height - window_size[1]
        for x in range(0, width - window_size[0] + 1, stride):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            windows.append(window)
            positions.append((x, y))

    if (width - window_size[0]) % stride != 0:
        x = width - window_size[0]
        for y in range(0, height - window_size[1] + 1, stride):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            windows.append(window)
            positions.append((x, y))

    return windows, positions

# ...（同上）

def process_window(session, window, input_shape):
    # 预处理（仅缩放，不填充）
    resized_window = cv2.resize(window, (input_shape[3], input_shape[2]))
    normalized_window = resized_window / 255.0
    normalized_window = normalized_window.transpose(2, 0, 1)  # HWC -> CHW
    normalized_window = np.expand_dims(normalized_window, axis=0)
    return normalized_window.astype(np.float32)


def merge_detections(all_detections, positions, window_size=(640, 640), nms_threshold=0.5):
    merged_boxes = []
    for (x_offset, y_offset), detections in zip(positions, all_detections):
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            # 将窗口内坐标映射回原始图像
            x1_orig = x1 + x_offset
            y1_orig = y1 + y_offset
            x2_orig = x2 + x_offset
            y2_orig = y2 + y_offset
            merged_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig, conf, cls_id])

    # 应用 NMS
    if len(merged_boxes) == 0:
        return []

    boxes = np.array([box[:4] for box in merged_boxes])
    scores = np.array([box[4] for box in merged_boxes])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, nms_threshold)

    final_detections = [merged_boxes[i] for i in indices]
    return final_detections
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 分块处理
    windows, positions = sliding_window(frame, window_size, stride)
    all_detections = []

    for window, pos in zip(windows, positions):
        # 预处理
        input_data = process_window(session, window, input_shape)

        # 推理
        outputs = session.infer([input_data])

        # 解析输出（假设 outputs[0] 是检测框）
        detections = []
        for box in outputs[0][0]:
            x1, y1, x2, y2, conf, cls_id = box[:6]
            if conf > 0.5:
                detections.append((x1, y1, x2, y2, conf, cls_id))

        all_detections.append(detections)

    # 合并检测结果
    final_detections = merge_detections(all_detections, positions, window_size)

    # 绘制检测框
    for (x1, y1, x2, y2, conf, cls_id) in final_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {cls_id} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Sliding Window Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()