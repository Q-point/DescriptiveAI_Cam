
#Author : dhq 
#Date : January 27 2020

import os
import string
import glob
import pickle
from time import time
import numpy as np
from PIL import Image, ImageFile
import cv2

import matplotlib.pyplot as plt

from datetime import datetime


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


#@staticmethod
def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_TRIPLEX
    scale = 1
    color = (255, 255, 255)
    thickness = cv2.FILLED
    margin = 5

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 2, cv2.LINE_AA)



cap = cv2.VideoCapture(0)  
cap.set(3,640)
cap.set(4,480)

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    

while(cap.isOpened()):
    ret, frame = cap.read()
    capstr = "random test"  # generateCaption(frame)
    dateTimeObj = datetime.now()
    datstr = str(dateTimeObj)
    # draw the label into the frame
    __draw_label(frame, datstr, (50,450), (100,225,50))
    cv2.imshow('Frame',frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break     

cap.release()
cv2.destroyAllWindows()
