from qiniu import Auth, put_file
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

# 将生成的图片上传到七牛云
# 传入filename和filepath，返回图片的URL
# filename为上传后url中显示的图片名，filepath为本地路径名
def upload_pic_to_qiniu(filename, filepath):
    QINIU_AK = 'h1E1mid8K0zRr848y7uTIPi18GyXnDuzkaLkmW4C'
    QINIU_SK = '6x4bFwamzO0ebMWnNfrrAl8jyFLExXI4Oc2wTPD9'
    BASE_URL = 'http://xinan.ziqiang.net.cn/'

    access_key = QINIU_AK
    secret_key = QINIU_SK

    q = Auth(access_key, secret_key)

    # 要上传的空间
    bucket_name = 'xinxianquandasai'

    # 生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, filename, 3600)

    ret, info = put_file(token, filename, filepath)
    return BASE_URL + ret['key']

# 将视频拆分成帧
def VideoSplit(video_path: str):
    # video_path：url
    # 返回值：video_frames：dict类型, key为视频名称, value为视频信息
    video_name = video_path.split('/')[-1][:-4]
    video_frames = {}
    cap = cv2.VideoCapture(video_path)    
    all_frames = []
    i = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
            print("{} video {} frame is loaded".format(video_name, i))
        else:
            break
        i += 1
    if all_frames:
        print("Video {} loaded successfully".format(video_name))
        # return all_frames, width, height, fps
        video_frames[video_name] = [all_frames, width, height, fps]
    else:
        print("Video loading failed")
        # return None
        video_frames[video_name] = None
    return video_frames



if __name__ == "__main__":
    result = upload_pic_to_qiniu('face_test', 'C:/WorkSpace/test/detect_face_3.jpg')
    print(result)