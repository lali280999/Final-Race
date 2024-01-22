import sys


local_site_packages = '~/ Downloads/tensor_rt/torch2trt/myenv/lib/python3.8'

sys.path.append(local_site_packages)

#['', '/home/pear/.local/lib/python3.8/site-packages', '/home/pear/Downloads/tensor_rt/torch2trt', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/home/pear/.local/lib/python3.8/site-packages/torchvision-0.15.1-py3.8-linux-aarch64.egg', '/usr/local/lib/python3.8/dist-packages', '/usr/lib/python3/dist-packages', '/usr/lib/python3.8/dist-packages']


local_site_1 = '/home/pear/.local/lib/python3.8/site-packages' #, '/home/pear/Downloads/tensor_rt/torch2trt', '/home/pear/.local/lib/python3.8/site-packages/torchvision-0.15.1-py3.8-linux-aarch64.egg', '/usr/local/lib/python3.8/dist-packages', '/usr/lib/python3/dist-packages', '/usr/lib/python3.8/dist-packages']
sys.path.append(local_site_1)

local_site_2 = '/home/pear/Downloads/tensor_rt/torch2trt'
sys.path.append(local_site_2)

local_site_3 = '/home/pear/.local/lib/python3.8/site-packages/torchvision-0.15.1-py3.8-linux-aarch64.egg'
sys.path.append(local_site_3)

local_site_4 = '/usr/local/lib/python3.8/dist-packages'
sys.path.append(local_site_4)

local_site_5 = '/usr/lib/python38.zip'
sys.path.append(local_site_5)

local_site_6 = '/usr/lib/python3.8'
sys.path.append(local_site_6)

local_site_7 = '/usr/lib/python3.8/lib-dynload'
sys.path.append(local_site_7)

local_site_8 = '/usr/lib/python3/dist-packages'
sys.path.append(local_site_8)

local_site_9 = '/usr/lib/python3.8/dist-packages'
sys.path.append(local_site_9)

local_site_10 = '/usr/lib/aarch64-linux-gnu'
sys.path.append(local_site_10)

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from PIL import Image
from torchvision import transforms
from train import SquareCoordNet
from torch.utils.data import Dataset, DataLoader

import math

import djitellopy as Tello

import time


from torch2trt import TRTModule

import threading

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import math

import djitellopy as tellos

import time

import pytorch_spynet.run_tello as run_tello

import flowiz as fz

import matplotlib.pyplot as plt
from pnp import pnp


frame_counter = 0
#read map from maps 
#its a txt file with example
# An example window environment file for P3b testing
# boundary xmin ymin zmin xmax ymax zmax
# window x y z xdelta ydelta zdelta qw qx qy qz xangdelta yangdelta zangdelta
# boundary -1.73 -1.15 0.0 1.73 7.4 2.0
# window 0.65 1.83 1.36 0.5 0.5 0.5 0.9816 0 0 0.1908 5 5 20 
# window -0.55 3.57 1.36 0.5 0.5 0.5 1 0 0 0 5 5 20
# window 0.6 5.59 1.36 0.5 0.5 0.5 0.9890 0 0 0.1478 5 5 20
 
file=open('maps/map1.txt','r')
lines=file.readlines()
window_list=[]
for line in lines:
    #if the first word is boundary
    if line.split()[0]=='boundary':
        #get the boundary coordinates
        xmin=float(line.split()[1])
        ymin=float(line.split()[2])
        zmin=float(line.split()[3])
        xmax=float(line.split()[4])
        ymax=float(line.split()[5])
        zmax=float(line.split()[6])
        #print(xmin,ymin,zmin,xmax,ymax,zmax)
    #if the first word is window
    if line.split()[0]=='window':
        #get the window coordinates
        x=float(line.split()[1])
        y=float(line.split()[2])
        z=float(line.split()[3])
        # x,y=y,-x
        xdelta=float(line.split()[4])
        ydelta=float(line.split()[5])
        zdelta=float(line.split()[6])
        qw=float(line.split()[7])
        qx=float(line.split()[8])
        qy=float(line.split()[9])
        qz=float(line.split()[10])
        xangdelta=float(line.split()[11])
        yangdelta=float(line.split()[12])
        zangdelta=float(line.split()[13])
        #print(x,y,z,qw,qx,qy,qz)
        #append the window coordinates to the window list
        window_list.append([x,y,z,qw,qx,qy,qz])





class TelloCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        tello_thrd = Tello.Tello()
        self.tello = tello_thrd
        self.tello.streamon()
        self.frame = None
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self.tello.get_frame_read().frame
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if frame is None or np.max(frame) < 10:
                    continue
                self.frame = frame
                print("taking_images")
                time.sleep(0.2)

            except Exception as e:
                print(f"Error capturing frame: {e}")
                # tello.streamoff()
                print("restart stream")
                self.tello.streamon()
                print("stream on")
                time.sleep(0.1)
                continue
         #   time.sleep(0.6)  # Adjust the sleep duration as needed

    def stop(self):
        self._stop_event.set()
        self.tello.end()



model_window = TRTModule()
model_window.load_state_dict(torch.load('model_trt.pth'))

# model_state_dict = torch.load('model_eff_net.pth')

# #load the model
# model = SquareCoordNet()
# model.load_state_dict(model_state_dict)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_window.to(device)


# put in eval mode
model_window.eval()


# function to get point from text file path

def get_image_points(image, model, device):

    img = image.copy()

    img = cv2.resize(img,(224,224))

    image = transforms.ToTensor()(image)
    image = image.to(device)
    transform=transforms.Resize((224,224))
    image = transform(image)
    # push image to gpu
    
    #pass the image through the model
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
    
    # outputs.tolist()

    outputs_test = outputs.cpu()

    


    outputs_test = np.array(outputs_test)

    outputs_test = outputs_test.reshape(4,2)

    for j in range(len(outputs_test)):
        if j==0:
            color=(0,0,255)
        if j==1:
            color=(0,255,0)
        if j==2:
            color=(255,0,0)
        if j==3:
            color=(255,255,0)
        cv2.circle(img, (int(outputs_test[j][0]), int(outputs_test[j][1])), 5, color, -1)

    #save the image4.285714285714286
    file_name='images_test/result'+'.jpg'
    cv2.imwrite(file_name, img)

    return outputs

def window_center(tello,tello_capture_thread):
    global frame_counter
    while True:
        if tello_capture_thread.frame is None:
            continue
        frame = tello_capture_thread.frame.copy()

        frame_counter += 1

        if frame_counter<50:
            continue

        image_points=get_image_points(frame,model_window,device)
        outputs_ = image_points.tolist()
        image_points = np.array(outputs_, dtype=np.float32)
        image_points=np.reshape(image_points,(4,2))

        #find the center of the points
        center_x=(image_points[0][0]+image_points[1][0]+image_points[2][0]+image_points[3][0])/4
        center_y=(image_points[0][1]+image_points[1][1]+image_points[2][1]+image_points[3][1])/4

        #if the center is not in the center of the image(224x224), move the drone
        image_center_x=224/2
        image_center_y=224/2

        delta_x=center_x-image_center_x
        delta_y=center_y-image_center_y

        factor=0.6

        #move the drone
        tello.send_rc_control(int(delta_x*factor),0,-int(delta_y*factor),0)

        print("delta_x", delta_x)
        
        print("delta_y", delta_y)

        time.sleep(0.1)

        if abs(delta_x)<20 and abs(delta_y)<20:
            tello.send_rc_control(0,0,0,0)
            time.sleep(0.1)
            break

def get_pnp(tello_capture_thread):
    #get the image points
    image=tello_capture_thread.frame.copy()
    image_points=get_image_points(image,model_window,device)
    image_points= image_points.cpu()
    image_points = image_points.tolist()
    image_points = np.array(image_points, dtype=np.float32)
    image_points=np.reshape(image_points,(4,2))
    for j in range(len(image_points)):
            image_points[j][0]=image_points[j][0]*4.285714285714286
            image_points[j][1]=image_points[j][1]*3.2142857142857144

    rmat,tvec=pnp(image_points)
    return tvec

    
    

def get_countour_center(tello,tello_capture_thread):
    
    frame = tello_capture_thread.frame.copy()
    # save the frame as frame_left
    cv2.imwrite('images_test/frame_left.jpg', frame)
    # give velocity to right
    tello.send_rc_control(10, 0, 0, 0)
    time.sleep(0.3)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.2)
    frame2 = tello_capture_thread.frame.copy()
    cv2.imwrite('images_test/frame_right.jpg', frame2)

    frame = cv2.resize(frame, (1024, 416))
    frame2 = cv2.resize(frame2, (1024, 416))


    tenOne = torch.FloatTensor(np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(np.ascontiguousarray(frame2[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    flow = run_tello.estimate(tenTwo,tenOne)


    strout = 'check.flo'
    objOutput = open(strout, 'wb')


    np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
    np.array([ flow.shape[2], flow.shape[1] ], np.int32).tofile(objOutput)
    np.array(flow.numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)
    objOutput.close()


    file = 'check.flo'
    img = fz.convert_from_file(file)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    found=False
    threshold=255
    # Threshold the image using Otsu's thresholding
    while found==False:
        ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY )
        edges = cv2.Canny(thresh, 50, 150, apertureSize=7)

        # smooth the edges using dilation and erosion
        kernel = np.ones((6, 6), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        cnts = []
        for c in contours:
            # remove contours which is near the top of the image as well and at the bottom of the image
            # get the bounding rect of c
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) > 20000 and y > 30 and y < 300:
                print(cv2.contourArea(c))
                found=True
                cnts.append(c)

                
                break
        threshold=threshold-10

        if threshold<100:
            break





    # select the largest contour
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    larget_contour = cnts[0]

    cv2.drawContours(frame2, [larget_contour], -1, (0, 0, 255), 2)
    file_name='images_test/threshold'+'.jpg'
    cv2.imwrite(file_name, frame2)

 


    # find the center of contour
    M = cv2.moments(larget_contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cv2.circle(img, (int(cx), int(cy)), 5, (255,0,0), -1)

    file_name='images_test/result_c'+'.jpg'
    cv2.imwrite(file_name, img)


    return cx,cy


def optical_flow_center(tello,tello_capture_thread):
    while True:
        try:
            center_x,center_y=get_countour_center(tello,tello_capture_thread)

            #if the center is not in the center of the image(1024, 416), move the drone
            image_center_x=1024/2
            image_center_y=416/2

            delta_x=center_x-image_center_x
            delta_y=center_y-image_center_y

            factor_x=0.1
            factor_y=0.1

            #move the drone
            tello.send_rc_control(int(delta_x*factor_x),0,-int(delta_y*factor_y),0)

            time.sleep(0.2)
            print("delta_x", delta_x)
            print("delta_y", delta_y)

            tello.send_rc_control(0,0,0,0)
            time.sleep(0.2)

            if abs(delta_x)<150 and abs(delta_y)<20000:
                break
        except Exception as e:
            if (e == KeyboardInterrupt):
                tello.land()
                tello.end()
                tello_capture_thread.stop()
                cv2.destroyAllWindows()
                sys.exit()
            else:
                continue
    




def get_window_center(frame):
    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame_copy = frame.copy()
    
    #hsv hue sat value
    # lower_blue = np.array([97,0,0])
    # upper_blue = np.array([110,255,255])

    lower_blue = np.array([106,50,0])   #80
    upper_blue = np.array([124,255,150])

    #hsv hue sat value
    lower_pink = np.array([140,80,0])
    upper_pink= np.array([174,255,197])

    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    #find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #find contour with max area
    c = max(contours, key = cv2.contourArea)

    #find bounding rectangle
    x,y,w,h = cv2.boundingRect(c)

    #draw bounding rectangle
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #find contour center
    M = cv2.moments(c)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    #draw center
    cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)

    
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)

    frame1=frame_copy.copy()
    
    mask1 = cv2.inRange(hsv, lower_pink, upper_pink)
    res1 = cv2.bitwise_and(frame1, frame1, mask = mask1)
    # cv2.imshow('mask1', mask1)
    # cv2.imshow('res1', res1)

    #find contours
    contours1, hierarchy1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #find contour with max area
    c1 = max(contours1, key = cv2.contourArea)

    #find bounding rectangle
    x1,y1,w1,h1 = cv2.boundingRect(c1)

    # get the four corners of the rectangle from x1,y1,w1,h1, it should be a 4x2 array

    corner_pts = np.array([[x1,y1],[x1,y1+h1],[x1+w1,y1],[x1+w1,y1+h1]])

    # scale the corner points, dimension 1 is multiplied by 3.2 and dimension2 by 4.2857

    corner_pts[:,0] = corner_pts[:,0]*4.285714285714286
    corner_pts[:,1] = corner_pts[:,1]*3.2142857142857144





    #draw bounding rectangle
    cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

    #find contour center
    M1 = cv2.moments(c1)
    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])

    #draw center
    cv2.circle(frame,(cx1,cy1), 5, (0,255,0), -1)

    # cv2.imshow('frame', frame)
    save_str = 'images_test/frame' +'.jpg'
    cv2.imwrite(save_str,frame)
    # counter = counter+1

    return cx,cy,cx1,cy1, corner_pts

def dynamic_window_center(tello,tello_capture_thread):
    d_frame_counter=0
    while True:
        if tello_capture_thread.frame is None:
            continue
        d_frame_counter += 1
        if d_frame_counter<50:
            continue
        frame = tello_capture_thread.frame.copy()

        center_x,center_y,cx1,cy1, corner_points =get_window_center(frame)

        # #if the center is not in the center of the image(224x224), move the drone
        image_center_x=224/2
        image_center_y=224/2

        delta_x=center_x-image_center_x
        delta_y=center_y-image_center_y

        factor=0.3

        #move the drone
        tello.send_rc_control(int(delta_x*factor),0,-int(delta_y*factor),0)

        print("delta_x", delta_x)
        
        print("delta_y", delta_y)

        time.sleep(0.1)

        if abs(delta_x)<30 and abs(delta_y)<30:
            tello.send_rc_control(0,0,0,0)
            time.sleep(0.1)
            break

    # perform pnp on the corner points
        
    rmat,tvec=pnp(corner_points)

    return tvec






    

    



cur_location = [0,0,0] # in centimeters





if __name__ == "__main__":
    tello = Tello.Tello()
    tello.connect()
    print(tello.get_battery())
    tello_capture_thread = TelloCaptureThread()
    
    try:
        
        tello.takeoff()
        time.sleep(0.1)
        
        

        tello_capture_thread.start()

        for i in range(len(window_list)):
            #get the window coordinates
            x=window_list[i][0]
            y=window_list[i][1]
            z=window_list[i][2]
            qw=window_list[i][3]
            qx=window_list[i][4]
            qy=window_list[i][5]
            qz=window_list[i][6]
            x=int(x*100)
            y=int(y*100)
            z=int(z*100)



            #move the drone to the window
            print("xyz", x,y,z)
            print("cur_location", cur_location)
            if i==0 or i==1:
                tello.go_xyz_speed(x-170-cur_location[0],y-cur_location[1],z-cur_location[2]-50,100)  # [334, -67, 56] [873, -18, 124] go -464 -62 -28 100 
                #update the current location
                cur_location[0]= x-170
                cur_location[1]= y
                cur_location[2]= z-50 
            if i==2:
                tello.go_xyz_speed(x-145-cur_location[0],y-cur_location[1],z-cur_location[2]-50,100)
            

                #update the current location
                cur_location[0]= x-145
                cur_location[1]= y
                cur_location[2]= z-50

            if i==3:
                tello.go_xyz_speed(0,0,z-cur_location[2]-50,100)
                
                tello.rotate_clockwise(60)
                tello.go_xyz_speed(-30,0,0,100)


                #update the current location
                # cur_location[0]= x
                # cur_location[1]= y
                cur_location[2]= z-50

                frame = tello_capture_thread.frame.copy()
                # save the frame as frame_last
                cv2.imwrite('images_test/frame_last.jpg', frame)

            print("cur_location_mid", cur_location)

            if i ==0 :
                window_center(tello,tello_capture_thread)
                tvec=get_pnp(tello_capture_thread)
                tello.go_xyz_speed(int(tvec[2][0]*100)+150,int(-tvec[0][0]*100),int(-tvec[1][0]*100)-30,100)

                cur_location[0]=cur_location[0]+int(tvec[2][0]*100)+150
                cur_location[1]=cur_location[1]+int(-tvec[0][0]*100) 
                cur_location[2]=cur_location[2]+int(-tvec[1][0]*100)-30

            if i == 1:
                window_center(tello,tello_capture_thread)
                tvec=get_pnp(tello_capture_thread)
                tello.go_xyz_speed(int(tvec[2][0]*100)+100,int(-tvec[0][0]*100),int(-tvec[1][0]*100)-30,100)

                cur_location[0]=cur_location[0]+int(tvec[2][0]*100)+100
                cur_location[1]=cur_location[1]+int(-tvec[0][0]*100) 
                cur_location[2]=cur_location[2]+int(-tvec[1][0]*100)-30
                # tello.go_xyz_speed(100,0,-20,100)
                # cur_location[0]=cur_location[0]+100
                # cur_location[1] = cur_location[1]
                # cur_location[2] = cur_location[2] - 20

            if i==2:
                optical_flow_center(tello,tello_capture_thread)
                tello.go_xyz_speed(300,-12,-30,100)
                cur_location[0]=cur_location[0]+300
                cur_location[1] = cur_location[1] - 5
                cur_location[2] = cur_location[2] - 20

            if i==3:
                tvec = dynamic_window_center(tello,tello_capture_thread)
                print(tvec)
                while True:
                    frame = tello_capture_thread.frame.copy()
                    if frame is None or np.max(frame) < 10:
                        continue
                    else:
                        print("waiting")
                        center_x,center_y,cx1,cy1, corner_points =get_window_center(frame)

                        if cx1<center_x and cy1>center_y:
                            break


                tello.go_xyz_speed(300,10,-50,100)
                tello.land()

                cur_location[0]=cur_location[0]+int(tvec[2][0]*100)+100
                cur_location[1]=cur_location[1]+int(-tvec[0][0]*100) 
                cur_location[2]=cur_location[2]+int(-tvec[1][0]*100)-50
    
    except KeyboardInterrupt:
        tello.land()
        tello.streamoff()
        tello.end()
        tello_capture_thread.stop()
        cv2.destroyAllWindows()
        sys.exit()

