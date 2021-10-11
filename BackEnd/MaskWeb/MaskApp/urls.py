from django.urls import path

from . import views

urlpatterns = [
    path('', views.FaceGenerate, name='FaceGenerate'),
    path('facegen/', views.FaceGenerate, name='FaceGenerate'),
]