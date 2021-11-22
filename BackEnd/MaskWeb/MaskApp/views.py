import sys,os
# from BackEnd.Sefa.api import load_code
wycpath = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(wycpath, 'MaskWeb', 'MaskApp'))
sys.path.append(wycpath)
import time
from qiniu import CdnManager
import qiniu
import cv2
import numpy as np
from Sefa.api import code_to_img_api, load_code
import json
from FaceDetector.FaceTest import FaceSearch
from django.views.decorators.csrf import csrf_exempt
from TediGAN.generation_api import ImageEdit, save_code
from FaceDetector.Faceutils import upload_pic_to_qiniu
from maskutils import GetImageFromHttp
from django.http import JsonResponse
from django.http import HttpResponse
from django.shortcuts import render
import re
from typing import Pattern


# Create your views here.
@csrf_exempt
def FaceGenerate(request):
    # 检查上传文本格式
    face_text = request.POST.get('facetext')
    img_url = request.POST.get('img')
    print(face_text)
    print(img_url)
    if img_url == '':
        image_path = None
        ret_code = 0
    else:
        image_path, ret_code = GetImageFromHttp(img_url)
    # 错误处理
    if ret_code:
        return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Text or URL", "data": None})
    # 使用原神的接口
    else:
        code, _ = ImageEdit(image_path=image_path, description=face_text)
        code_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        code_name = os.path.join(code_dir, 'FaceStatic', 'tempnpy', 'code.npy')
        save_code(code, code_name)
        # new_code_name = 'code.npy'
        # code_url = upload_pic_to_qiniu(new_code_name, code_name)
        return JsonResponse({"Error Code": 0, "Error Message": "SUCCESS", "data": code_name})


@csrf_exempt
def FaceModify(request):
    # modify_params是json格式的字符串, code_url是上一步生成的url地址
    # testparams: {"layer_idx": "all", "num_semantics": "2", "step": "1,2"},step中用逗号分隔
    # modify_params = request.POST.get('modify_params', None)
    # print(request.POST)
    # code_path = request.POST.get('code_url', None)
    layer_idx = request.POST.get('layer_idx', None)
    num_semantics = request.POST.get('num_semantics', None)
    step = request.POST.get('step', None)
    print(layer_idx)
    print(num_semantics)
    print(step)
    # 错误处理
    ret_code = 0
    if layer_idx is None or num_semantics is None or step is None:
        ret_code = 1
    if ret_code:
        return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Params", "data": None})
    else:
        # params = json.loads(modify_params)
        num_semantics = int(num_semantics)
        step = list(map(int, step.split(',')))
        code_py = load_code(os.path.join(
            wycpath, 'FaceStatic', 'tempnpy', 'code.npy'))
        if code_py is None:
            return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Code, please run FaceGenerate", "data": None})
        if len(step) != num_semantics:
            return JsonResponse({"Error Code": 1, "Error Message": "Step Nums must equal to num_semantics", "data": None})
        image = code_to_img_api(
            codes=code_py, layer_idx=layer_idx, num_semantics=num_semantics, step=step)
        # print(type(image))
        # print(image.shape)
        image = image[0]
        # print(image.shape)
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(image.shape)
        image_path = os.path.join(wycpath, 'FaceStatic', 'tempmod.jpg')
        cv2.imwrite(image_path, image)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        image_name = 'Mod_images_'+time_str+'.jpg'
        image_url = upload_pic_to_qiniu(image_name, image_path)
        print(image_url)
        # 账户ak，sk
        access_key = 'h1E1mid8K0zRr848y7uTIPi18GyXnDuzkaLkmW4C'
        secret_key = '6x4bFwamzO0ebMWnNfrrAl8jyFLExXI4Oc2wTPD9'
        auth = qiniu.Auth(access_key=access_key, secret_key=secret_key)
        cdn_manager = CdnManager(auth)
        # 需要刷新的文件链接
        urls = [image_url]
        refresh_url_result = cdn_manager.refresh_urls(urls)
        # print(refresh_url_result)
        return JsonResponse({"Error Code": 0, "Error Message": "SUCCESS", "data": image_url})


@csrf_exempt
def FaceMatch(request):
    image_url = request.POST.get('image_url', None)
    video_url = request.POST.get('video_url', None)
    # 错误处理
    ret_code = 0
    if image_url is None or video_url is None:
        ret_code = 1
    # 调用接口
    if ret_code:
        return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Image url or Video url", "data": None})
    else:
        result = FaceSearch(image_url, video_url)
        return JsonResponse({"Error Code": result['Error code'], "Error Message": "SUCCESS", 'data': json.dumps(result['Data'])})
