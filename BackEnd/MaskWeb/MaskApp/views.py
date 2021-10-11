import re
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from maskutils import GetImageFromHttp
from BackEnd.FaceDetector.Faceutils import upload_pic_to_qiniu
from BackEnd.TediGAN.generation_api import ImageEdit, save_code
import os

# Create your views here.
def FaceGenerate(request):
    # 检查上传文本格式
    face_text = request.POST.get('facetext')
    img_url = request.POST.get('img')
    if img_url == '':
        image_path = None
        code = 0
    else:
        image_path, code = GetImageFromHttp(img_url)
    # 错误处理
    if code:
        return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Text or URL", "data": None})
    # 使用原神的接口
    else:
        code, _ = ImageEdit(image_path=image_path, description=face_text)
        code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code_name = os.path.join(code_dir, 'FaceStatic', 'tempnpy', 'code.npy')
        save_code(code, code_name)
        new_code_name = ''
        code_url = upload_pic_to_qiniu(new_code_name, code_name)
        return JsonResponse({"Error Code": 0, "Error Message": "SUCCESS", "data": code_url})

def FaceModify(request):
    modify_text = request.POST.get('modtext', None)
    img_url = request.POST.get('img', None)
    # 错误处理
    if not modify_text or not img_url:
        return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Text or URL", "data": None})
    # 使用原神的接口
    pass
    return JsonResponse()

def FaceMatch(request):
    
    pass
    
