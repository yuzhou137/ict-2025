import cv2
import mindspore as ms
from mindyolo.models import YOLOv8
from mindyolo.utils.config import _parse_yaml
from mindspore import load_checkpoint, load_param_into_net

# 设置MindSpore的设备
ms.set_context(device_target="CPU")

# 加载配置文件
cfg = _parse_yaml('C:/Users/22454/Desktop/project/deep_learning/mindyolo-master/mindyolo-master/configs/yolov8/yolov8n.yaml')

# 创建模型
model = YOLOv8(cfg)

# 加载权重文件
param_dict = load_checkpoint('yolov8n.ckpt')
load_param_into_net(model, param_dict)

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