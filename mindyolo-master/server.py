import socket
import cv2
import struct
import pickle
import zlib
from threading import Thread


def process_image():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低分辨率
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    if ret:
        process_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = 0.95
        item = "object"
        return process_frame, score, item, frame
    return None, None, None, None


def send_data(sock, addr):
    while True:
        p_frame, score, item, orig_frame = process_image()
        if p_frame is None: continue

        # 压缩数据
        _, jpg_frame = cv2.imencode('.jpg', orig_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # 降低质量
        data = {
            'score': score,
            'item': item,
            'image': jpg_frame
        }
        compressed = zlib.compress(pickle.dumps(data), level=1)  # 快速压缩

        # UDP发送
        sock.sendto(struct.pack('!I', len(compressed)) + compressed, addr)


if __name__ == '__main__':
    UDP_IP = '0.0.0.0'
    UDP_PORT = 12345
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print("等待客户端连接...")
    _, addr = sock.recvfrom(1024)  # 等待客户端握手

    Thread(target=send_data, args=(sock, addr)).start()  # 多线程发送