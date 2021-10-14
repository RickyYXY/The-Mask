import os, sys
from numpy.core.numeric import full

from numpy.lib.npyio import _save_dispatcher
basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basepath)
from FaceDetector.FaceExt import FaceExtract
from FaceDetector.FaceCop import FaceComp
from FaceDetector.FaceExt import fetchImageFromHttp
from FaceDetector.Faceutils import VideoSplit
from FaceDetector.Faceutils import upload_pic_to_qiniu
import cv2
import base64

def FaceSearch(exampleimg: str, searchvid: str):
    message = {}
    expimg = fetchImageFromHttp(exampleimg)
    srhvidframes = VideoSplit(searchvid)
    # 得到待检测的人脸本地路径和画框结果路径
    exp_res = FaceExtract(exampleimg, 'url', 'Example')
    expface = exp_res['Data']['ExampleFaces']
    exp_error_code = exp_res['Error Code']
    if exp_error_code != 0:
        message['Error code'] = exp_error_code
        message['Error Message'] = '示例人脸检测失败'
        message['Data'] = None
    else:
        fullexpface = exp_res['Data']['FullFace']
        fullexpurl = upload_pic_to_qiniu('ExampleFace.jpg', fullexpface)
    # 将每一秒中的一帧临时写入文件夹，每帧都写入数量太大
    fps = srhvidframes[3]
    need_frame = srhvidframes[0][::int(fps)]
    framelen = len(need_frame)
    need_frame_path = []
    for idx in range(framelen):
        savepath = os.path.join(basepath, 'FaceStatic', 'TempFrames', str(idx)+'.jpg')
        need_frame_path.append(savepath)
        cv2.imwrite(savepath, need_frame[idx])
    # 进行人脸匹配
    examplepath = os.path.join(basepath, 'FaceStatic', 'ExampleFace')
    searchpath = os.path.join(basepath, 'FaceStatic', 'SearchFace')
    high_score = -1
    high_idx = -1
    num = 0
    for frame in need_frame_path:
        search_res = FaceExtract(frame, 'Local', 'Search')
        if not search_res['Data']:
            continue
        for face in search_res['Data']['ExampleFaces']:
            cop_res = FaceComp(expface, face)
            if not cop_res.json()['result']:
                continue
            if float(cop_res.json()['result']['score']) >= high_score:
                high_score = float(cop_res.json()['result']['score'])
                high_face_path = face
                high_idx = num
        num += 1
    # 检测完成后删除文件
    if high_idx < 0:
        message['Error code'] = exp_error_code
        message['Error Message'] = '视频人脸检测失败'
        message['Data'] = None
    else:
        high_frame_path = need_frame_path[high_idx + 1]
        high_face_pos = high_face_path.split(os.sep)[-1].split('.')[0].split(',')
        high_frame = cv2.imread(high_frame_path)
        lefttopx = int(high_face_pos[0])
        lefttopy = int(high_face_pos[1])
        rightbottomx = int(high_face_pos[2])
        rightbottomy = int(high_face_pos[3])
        cv2.rectangle(high_frame, (lefttopx, lefttopy), (rightbottomx, rightbottomy), (0, 255, 0), 2)
        full_face_path = os.path.join(basepath, 'FaceStatic', 'FullFace', 'Search.jpg')
        cv2.imwrite(full_face_path, high_frame)
        high_url = upload_pic_to_qiniu('High_score_Face', full_face_path)   
        message['Error code'] = 0
        message['Error Message'] = 'SUCCESS'
        if high_score >= 80:
            match_res = 'SUCCESS'
        else:
            match_res = 'Failed'
        message['Data'] = {'match score': high_score, 'match result': match_res, 'ExampleFace': fullexpurl, 'SearchFace': high_url}
    del_path = []
    for i in os.listdir(os.path.join(basepath, 'FaceStatic', 'FullFace')):
        del_path.append(os.path.join(basepath, 'FaceStatic', 'FullFace', i))
    for i in os.listdir(os.path.join(basepath, 'FaceStatic', 'ExampleFace')):
        del_path.append(os.path.join(basepath, 'FaceStatic', 'ExampleFace', i))
    for i in os.listdir(os.path.join(basepath, 'FaceStatic', 'SearchFace')):
        del_path.append(os.path.join(basepath, 'FaceStatic', 'SearchFace', i))
    for i in os.listdir(os.path.join(basepath, 'FaceStatic', 'TempFrames')):
        del_path.append(os.path.join(basepath, 'FaceStatic', 'SearchFace', i))
    for i in del_path:
        os.remove(i)
    return message

if __name__ == "__main__":
    Facepath = 'http://xinan.ziqiang.net.cn/AsFace.jpg'
    videopath = 'http://xinan.ziqiang.net.cn/Asvideo.mp4'
    result = FaceSearch(Facepath, videopath) 
    print(result)

    