import cv2
import yaml
import mindspore as ms
from mindyolo.models import YOLOv8

# 设置MindSpore的设备
ms.set_context(device_target="CPU")
with open("configs/yolov8/yolov8n.yaml", 'r') as file:
    cfg = yaml.safe_load(file)
# 加载预训练模型
model = YOLOv8(cfg=cfg)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为MindSpore Tensor
    image = ms.Tensor(frame)

    # 进行推理
    output = model(image)

    # 处理输出结果并绘制边界框
    # ...

    cv2.imshow("Real-time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()