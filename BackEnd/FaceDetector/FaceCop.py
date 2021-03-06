# encoding:utf-8
import requests
import base64
import cv2
import numpy as np
import urllib.request
import base64
import json

def fetchImageFromHttp(image_url, timeout_s=1):
    # 该函数是读取url图片
    try:
        if image_url:
            resp = urllib.request.urlopen(image_url, timeout=timeout_s)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        else:
            return []
    except Exception as error:
        print('获取图片失败', error)
        return []

def FaceComp(face1: str, face2: str, thre=80):
    # 该函数的作用是返回两张人脸的相似度分数以及根据阈值判断是否为一张人脸
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=DXN8o5eNaheZahxK558I8GOs&client_secret=bBYbDZj6dbv7w5Pr5hqelm7lL8GfPsBR'
    response = requests.get(host)
    if response:
        # print(response.json()['access_token'])
        access_token = response.json()['access_token']
    '''
    人脸对比
    '''
    with open(face1, "rb") as f1:  # 转为二进制格式
        face1_data = base64.b64encode(f1.read())  # 使用base64进行加密
    face1_data = face1_data.decode()
    with open(face2, "rb") as f2:  # 转为二进制格式
        face2_data = base64.b64encode(f2.read())  # 使用base64进行加密
    face2_data = face2_data.decode()
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"
    params = "[{{\"image\": \"{}\", \"image_type\": \"BASE64\", \"face_type\": \"LIVE\", \"quality_control\": \"NONE\"}},{{\"image\": \"{}\", \"image_type\": \"BASE64\", \"face_type\": \"IDCARD\", \"quality_control\": \"NONE\"}}]".format(face1_data, face2_data)
    params = json.loads(params)
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, json=params, headers=headers)
    # if response:
    #     print (response.json())
    return response

if __name__ == "__main__":
    imgpath1 = r'E:\WorkSpace\2020DaSanXia\BigSoft\The-Mask\BackEnd\FaceStatic\ExampleFace\example_face_0.jpg'
    imgpath2 = r'E:\WorkSpace\2020DaSanXia\BigSoft\The-Mask\BackEnd\FaceStatic\SearchFace\\690,241,858,398.jpg'
    result = FaceComp(imgpath1, imgpath2)
    print(result.json()['result']['score'])  