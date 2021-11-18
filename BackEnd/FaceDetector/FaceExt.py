# encoding:utf-8
import os, sys
basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(basepath, 'FaceDetector'))
import requests
import base64
import cv2
import numpy as np
import urllib.request
import base64


def fetchImageFromHttp(image_url, timeout_s=1):
    # 该函数是读取url图片
    if image_url:
        resp = urllib.request.urlopen(image_url, timeout=timeout_s)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    else:
        return []

def FaceExtract(img: str, imgtype: str, imgpos: str, facenum=120):
    # 该函数的作用是提取图中人脸
    message = {}
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=DXN8o5eNaheZahxK558I8GOs&client_secret=bBYbDZj6dbv7w5Pr5hqelm7lL8GfPsBR'
    response = requests.get(host)
    if response:
    # print(response.json()['access_token'])
        access_token = response.json()['access_token']
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
    if imgtype == 'Local':
        with open(img, "rb") as f:  # 转为二进制格式
            base64_data = base64.b64encode(f.read())  # 使用base64进行加密
        base64_data = base64_data.decode()
        params = "\"image\":\"{}\",\"image_type\":\"BASE64\", \"max_face_num\":\"120\"".format(base64_data)
        params = '{' + params + '}'
    elif imgtype == 'url':
        params = "\"image\":\"{}\",\"image_type\":\"URL\", \"max_face_num\":\"120\"".format(img)
        params = '{' + params + '}'
    # print(params['image'])
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        print (response.json())
    # 提取检测到的所有人脸信息
    if response.json()['error_code'] != 0:
        message['Error Code'] = response.json()['error_code']
        message['Error Message'] = response.json()['error_msg']
        message['Data'] = None
        return message
        # raise Exception('人脸检测失败，失败码为{}，失败信息为:{}'.format(response.json()['error_code'], response.json()['error_msg']))
    face_number = response.json()['result']['face_num']
    face_List = []
    for num in range(face_number):
        face_loc_left = int(response.json()['result']['face_list'][num]['location']['left'])
        face_loc_top = int(response.json()['result']['face_list'][num]['location']['top'])
        face_loc_width = int(response.json()['result']['face_list'][num]['location']['width'])
        face_loc_height = int(response.json()['result']['face_list'][num]['location']['height'])
        face_List.append([face_loc_left, face_loc_top, face_loc_width, face_loc_height])
         
    # 这里是读取图像并画框
    if imgtype == 'Local':
        image = cv2.imread(img)
    elif imgtype == 'url':
        image = fetchImageFromHttp(img)
    # 图片编号起始
    search_all_path = []
    num = 0
    for pos in face_List:
        lefttopx = pos[0]
        lefttopy = pos[1]
        rightbottomx = lefttopx + pos[2]
        rightbottomy = lefttopy + pos[3]
        # print(lefttopx, lefttopy, rightbottomx, rightbottomy)
        cv2.rectangle(image, (lefttopx, lefttopy), (rightbottomx, rightbottomy), (0, 255, 0), 2)
        if imgpos == 'Example':
            savepath = os.path.join(basepath, 'FaceStatic', 'ExampleFace', 'example_face_' + str(num) + '.jpg')
        elif imgpos == 'Search':
            pos_name = ','.join([str(lefttopx), str(lefttopy), str(rightbottomx), str(rightbottomy)])
            savepath = os.path.join(basepath, 'FaceStatic', 'SearchFace', pos_name + '.jpg')
            search_all_path.append(savepath)
        # cv2.imwrite("C:/WorkSpace/test/detect_face_"+str(num)+'.jpg', image[lefttopy:rightbottomy, lefttopx:rightbottomx])
        cv2.imwrite(savepath, image[lefttopy:rightbottomy, lefttopx:rightbottomx])
        num += 1
    message['Error Code'] = response.json()['error_code']
    message['Error Message'] = message['Error Message'] = response.json()['error_msg']
    if imgpos == 'Example':
        full_face_path = os.path.join(basepath, 'FaceStatic', 'FullFace', 'Result.jpg')
        cv2.imwrite(full_face_path, image)
        message['Data'] = {'ExampleFaces': savepath, 'FacesNum': num, 'FullFace': full_face_path}
    elif imgpos == 'Search':
        # full_face_path = os.path.join(basepath, 'FaceStatic', 'FullFace', 'Search.jpg')
        # cv2.imwrite(full_face_path, image)
        message['Data'] = {'ExampleFaces': search_all_path, 'FacesNum': num, 'FullFace': None}
    return message 

if __name__ == "__main__":
    # imgpath = 'http://xinan.ziqiang.net.cn/ThreeFace.jpeg'
    imgpath = 'http://xinan.ziqiang.net.cn/Fq-PpUCF25C61q0muvXAHCok0uK2'
    wycpath = 'http://xinan.ziqiang.net.cn/AsFace.jpg'
    fetchImageFromHttp(wycpath)
    # result = FaceExtract(imgpath, 'url')
    # result = FaceExtract(imgpath, 'Local', 'Search')
    # cv2.imshow('image', result)
    # cv2.waitKey(0)

    

     

