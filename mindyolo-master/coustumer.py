import socket
import cv2
import struct
import pickle
import zlib
import numpy as np


def main():
    UDP_IP = '192.168.186.98'  # 服务器IP
    UDP_PORT = 12345

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b'connect', (UDP_IP, UDP_PORT))  # 握手信号

    buffer = b''
    while True:
        data, _ = sock.recvfrom(65535)  # 最大UDP包大小
        buffer += data

        while len(buffer) >= 4:
            msg_len = struct.unpack('!I', buffer[:4])[0]
            if len(buffer) < msg_len + 4: break

            compressed = buffer[4:4 + msg_len]
            buffer = buffer[4 + msg_len:]

            # 解压数据
            decompressed = zlib.decompress(compressed)
            data = pickle.loads(decompressed)

            # 显示图像
            img_array = np.frombuffer(data['image'], dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            cv2.imshow('Stream', img)
            print(f"Score: {data['score']}")

            if cv2.waitKey(1) == 'q': break


if __name__ == '__main__':
    main()