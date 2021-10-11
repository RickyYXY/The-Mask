# encoding:utf-8
import requests
import base64
import cv2
import numpy as np
import urllib.request
import base64
import os

def GetImageFromHttp(image_url, timeout_s=1):
    # 该函数是读取url图片
    try:
        resp = urllib.request.urlopen(image_url, timeout=timeout_s)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        img_name = 'temp.jpg'
        img_path = os.path.join(img_base_dir, 'FaceStatic', img_name)
        cv2.imwrite(img_path, image)
        return img_path, 0
    except Exception as error:
        print('获取图片失败', error)
        return None, 1

if __name__ == '__main__':
    print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))