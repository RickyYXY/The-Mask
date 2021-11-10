import sys, os
from typing import Pattern

# from BackEnd.Sefa.api import load_code
wycpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(wycpath, 'MaskWeb', 'MaskApp'))
sys.path.append(wycpath)
import re
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from maskutils import GetImageFromHttp
from FaceDetector.Faceutils import upload_pic_to_qiniu
from TediGAN.generation_api import ImageEdit, save_code
from django.views.decorators.csrf import csrf_exempt
from FaceDetector.FaceTest import FaceSearch
import json
from Sefa.api import code_to_img_api, load_code
import numpy as np
import cv2

# Create your views here.
@csrf_exempt
def FaceGenerate(request):
    # 检查上传文本格式
    face_text = request.POST.get('facetext')
    img_url = request.POST.get('img')
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
        code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code_name = os.path.join(code_dir, 'FaceStatic', 'tempnpy', 'code.npy')
        save_code(code, code_name)
        # new_code_name = 'code.npy'
        # code_url = upload_pic_to_qiniu(new_code_name, code_name)
        return JsonResponse({"Error Code": 0, "Error Message": "SUCCESS", "data": code_name})

@csrf_exempt
def FaceModify(request):
    # modify_params是json格式的字符串, code_url是上一步生成的url地址
    # testparams: {"layer_idx": "all", "num_semantics": "2", "step": "1,2"},step中用逗号分隔
    modify_params = request.POST.get('modify_params', None)
    # print(request.POST)
    # code_path = request.POST.get('code_url', None)
    # 错误处理
    ret_code = 0
    if modify_params is None:
        ret_code = 1
    if ret_code:
        return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Params", "data": None})
    else:
        params = json.loads(modify_params)
        layer_idx = params['layer_idx']
        num_semantics = int(params['num_semantics'])
        step = list(map(int, params['step'].split(',')))
        code_py = load_code(os.path.join(wycpath, 'FaceStatic', 'tempnpy', 'code.npy'))
        if code_py is None:
            return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Code, please run FaceGenerate", "data": None})
        if len(step) != num_semantics:
            return JsonResponse({"Error Code": 1, "Error Message": "Step Nums must equal to num_semantics", "data": None})
        image = code_to_img_api(codes=code_py, layer_idx=layer_idx, num_semantics=num_semantics, step=step)
        print(type(image))
        print(image.shape)
        image = image[0]
        print(image.shape)
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(image.shape)
        image_path = os.path.join(wycpath, 'FaceStatic', 'tempmod.jpg')
        cv2.imwrite(image_path, image)
        image_name = 'Mod_images.jpg'
        image_url = upload_pic_to_qiniu(image_name, image_path)
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
        return JsonResponse({"Error Code": result['Error code'], "Error Message": "SUCCESS", 'data': result['Data']})
    
