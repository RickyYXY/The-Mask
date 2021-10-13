import os, sys
basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basepath)
from BackEnd.FaceDetector.FaceExt import FaceExtract
from BackEnd.FaceDetector.FaceCop import FaceComp
import cv2



