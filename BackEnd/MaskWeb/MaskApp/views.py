import re
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

# Create your views here.
def FaceGenerate(request):
    face_text = request.POST.get('facetext', None)
    img_url = request.POST.get('img', None)
    # 错误处理
    if not face_text or not img_url:
        return JsonResponse({"Error Code": 1, "Error Message": "Failed to get Text or URL", "data": None})
    # 使用原神的接口
    pass
    return JsonResponse()

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
    
