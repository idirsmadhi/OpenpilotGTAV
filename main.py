from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import cv2 
from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
import sys

#-------------------------to detect ctrl+c and close the client
#!/usr/bin/env python
import signal
import sys

def signal_handler(sig, frame):
    client.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


#--------------------------------import Vpilot

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client

import argparse
import time

#-------------------------------fin


client = Client(ip="localhost", port=8000)

#camerafile = "sample.hevc"
supercombo = load_model('/home/idir/Bureau/modeld-master/models/supercombo.keras')

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

#cap = cv2.VideoCapture(camerafile)

NBFRAME = 10000

def frame_to_tensorframe(frame):                                                                                               
  H = (frame.shape[0]*2)//3                                                                                                
  W = frame.shape[1]                                                                                                       
  in_img1 = np.zeros((6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[0] = frame[0:H:2, 0::2]                                                                                    
  in_img1[1] = frame[1:H:2, 0::2]                                                                                    
  in_img1[2] = frame[0:H:2, 1::2]                                                                                    
  in_img1[3] = frame[1:H:2, 1::2]                                                                                    
  in_img1[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

def vidframe2img_yuv_reshaped():
  message = client.recvMessage()  
                
  # The frame is a numpy array that can we pass through a CNN for example     
  frame = frame2numpy(message['frame'], (1164,874))
  #ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  return frame, img_yuv.reshape((874*3//2, 1164))

def vidframe2frame_tensors():
  frame, img = vidframe2img_yuv_reshaped()
  imgs_med_model = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
  f2t = frame_to_tensorframe(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
  return frame, f2t

state = np.zeros((1,512))
desire = np.zeros((1,8))


#----------------------- Vpilolt

scenario = Scenario(drivingMode=[786603,100.0], weather='EXTRASUNNY',vehicle='blista',time=[12,0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) #manual driving drivingMode=-1
dataset = Dataset(rate=20, frame=[1164,874])
client.sendMessage(Start(scenario=scenario, dataset = dataset)) 

#------------------------ fin


# frame_tensors = np.zeros((NBFRAME,6,128,256))
# for i in tqdm(range(NBFRAME)):
#     frame_tensors[i] = vidframe2frame_tensors()[1]

# cap2 = cv2.VideoCapture("sample.hevc")

for i in tqdm(range(NBFRAME-1)):
  if i == 0:
    frame, frame_tensors1 = vidframe2frame_tensors()
  else :
    frame, frame_tensors2 = vidframe2frame_tensors()
    inputs = [np.vstack([frame_tensors1,frame_tensors2])[None], desire, state]
    # inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
    outs = supercombo.predict(inputs)
    print(outs)
    parsed = parser(outs)
    # Important to refeed the state
    state = outs[-1]
    pose = outs[-2]
    # ret, frame = cap2.read()
    # frame = cv2.resize(frame, (640, 420))
    # # Show raw camera image
    plt.clf()
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.title("Video")
    plt.imshow(frame,aspect="auto")
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #   break
    # Clean plot for next frame
    plt.subplot(1, 2, 2) # row 1, col 2 index 1
    plt.title("Prediction")
    # lll = left lane line
    plt.plot(parsed["lll"][0], range(0,192), "b-", linewidth=1)
    # rll = right lane line
    plt.plot(parsed["rll"][0], range(0, 192), "r-", linewidth=1)
    # path = path cool isn't it ?
    plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)


    #print(np.array(pose[0,:3]).shape)
    #plt.scatter(pose[0,:3], range(3), c="y")
    
    # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
    plt.gca().invert_xaxis()
    plt.pause(0.001)


plt.show()
