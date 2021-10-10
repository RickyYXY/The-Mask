from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def FaceGenerate(request):
    face_text = request
    return HttpResponse("Hello, world. You're at the MaskApp Test.")