import cv2
import numpy as np
import imutils
import threading
import math
import time
import timeit
import DobotDllType as dType

import cv2.aruco as aruco
import os

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

api = dType.load()

def dobot_get_start():
    
    dType.SetQueuedCmdClear(api)

    dType.SetHOMEParams(api, 200, 0, 20, 30, isQueued = 1) # x, y, z, r 
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1) # velocity[4], acceleration[4]
    dType.SetPTPCommonParams(api, 200, 200, isQueued = 1) # velocityRatio(속도율), accelerationRation(가속율)
   
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)

    dType.SetQueuedCmdStartExec(api)

def dobot_connect():

    CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound", 
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):
        dobot_get_start()

def segmentaition(frame):

    img_ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(img_ycrcb)

    _, cb_th = cv2.threshold(cb, 90, 255, cv2.THRESH_BINARY_INV)
    cb_th = cv2.erode(cb_th, None, iterations=2)
    cb_th = cv2.dilate(cb_th, None, iterations=2)

    return cb_th

def get_distance(x, y, imagePoints):
    
    objectPoints = np.array([[25,-5,0],
                            [35,-5,0],
                            [35,5,0],
                            [25,5,0],],dtype = 'float32')


    fx = float(470.5961)
    fy = float(418.18176)
    cx = float(275.7626)
    cy = float(240.41246)
    k1 = float(0.06950)
    k2 = float(-0.07445)
    p1 = float(-0.01089)
    p2 = float(-0.01516)

    cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float64')
    distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')
    _,rvec,t = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    R,_ = cv2.Rodrigues(rvec)
            
    u = (x - cx) / fx
    v = (y - cy) / fy
    Qc = np.array([[u],[v],[1]])
    Cc = np.zeros((3,1))
    Rt = np.transpose(R)
    Qw = Rt.dot((Qc-t))
    Cw = Rt.dot((Cc-t))
    V = Qw - Cw
    k = -Cw[-1,0]/V[-1,0]
    Pw = Cw + k*V
    
    px = Pw[0]
    py = Pw[1]

    #print("px: %f, py: %f" %(px,py))

    return px,py

def find_ball(frame,cb_th,box_points,start_t,find_time_list):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    px = None
    py = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
            px,py = get_distance(center[0], center[1],box_points)
            
            text = " %f , %f" %(px,py)
            cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            if len(find_time_list) > 0:
                terminate_t = timeit.default_timer()
                time = terminate_t - start_t + find_time_list[-1]

                find_time_list.append(time)
                
                print("px: %f, py: %f, time: %f\n" %(px,py,time))
            else:
                print("px: %f, py: %f" %(px,py))

    return px,py

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if draw:
        cv2.aruco.drawDetectedMarkers(img,bboxs)
        #print(len(bboxs))

    if len(bboxs) > 0:
        return bboxs[0][0]
    else:
        return [0,0]

def move_dobot(of_x,of_y):

    offset_x = float(of_x * 10 + 89)#float((of_x + 6) * 10 - 5)
    offset_y = float(-1 * of_y * 10)#float(of_y * 10 - 3)

    offset_x = round(offset_x,2)
    offset_y = round(offset_y,2)
    last_index = 0
    
    length = math.sqrt(math.pow(offset_x,2) + math.pow(offset_y,2))

    if length > 50 and length < 290 and of_x > 0:

        print("offset_x: %f, offset_y: %f, length: %f \n" %(offset_x,offset_y,length))
        last_index = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, offset_x, offset_y, -10, 0, isQueued = 1)
        dType.dSleep(100)

def Predict_x(ball_path_x,ball_path_y,prediction_y = 5):

    X = ball_path_y
    y = ball_path_x

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    lin_reg.intercept_, lin_reg.coef_
    X_new=np.linspace(-3, 3, 100).reshape(100, 1)

    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    
    rx = prediction_y

    predict_x = [rx,rx**2]
    predict_x = np.reshape(predict_x,(1,-1))
    predict_y = lin_reg.predict(predict_x)

    P_result_x = predict_y

    return P_result_x

def Predict_time(ball_path_x,ball_path_y,time_list,predict_x, predict_y = 5):

    time_list = np.reshape(time_list,(-1,1))

    x = [[0] * 2] * len(ball_path_x)
    x = np.reshape(x,(-1,2))

    for i in range(len(ball_path_x)):

        x[i][0] = ball_path_x[i]
        x[i][1] = ball_path_y[i]
    
    y = time_list

    mlr = LinearRegression()
    mlr.fit(x,y)


    pd_x = float(predict_x)
    my_x = [[pd_x,predict_y]]

    #my_x = np.reshape(my_x,(-1,1))
    
    predict_time = mlr.predict(my_x)

    #print(predict_time)

    return predict_time

def calc_V(ball_path_x,ball_path_y,time_list):

    start_x = ball_path_x[0]
    start_y = ball_path_y[0]

    end_x = ball_path_x[-1]
    end_y = ball_path_y[-1]

    time = time_list[-1] - time_list[0]

    av_V = math.sqrt(math.pow(end_x - start_x,2) + math.pow(end_y - start_y,2))/time

    return av_V
    

def main():

    dobot_connect()

    cap = cv2.VideoCapture(0)

    ball_path_x = []
    ball_path_y = []
    time_list = []

    find_time_list = []

    while(1):

        start_t = timeit.default_timer()

        ret,frame = cap.read()

        box_points = findArucoMarkers(frame)
        #print(len(box_points))

        if len(box_points) > 2:
            
            cb_th = segmentaition(frame)
            px,py = find_ball(frame,cb_th,box_points,start_t,find_time_list)

            if px != None and py != None and (py > 30):

                ball_path_x.append(px)
                ball_path_y.append(py)

                terminate_t = timeit.default_timer()

                if time_count == 0:
                    time_list.append(terminate_t - start_t)
                else:
                    time_list.append(terminate_t - start_t + time_list[-1])

                time_count += 1

            elif len(ball_path_x) > 0 and px != None and py != None and ( py <= 30 and 5 < py):

                Velocity = calc_V(ball_path_x,ball_path_y,time_list)
                print("average_Velocity: %f cm/s" %Velocity)
                
                predict_x = Predict_x(ball_path_x,ball_path_y)
                predict_time = Predict_time(ball_path_x,ball_path_y,time_list,predict_x)

                print("predict-- x: %f, y: 5, time: %f" %(predict_x,predict_time))

                

                of_x = predict_x
                of_y = 5
                move_dobot(of_x,of_y)

                find_time_list.append(time_list[-1])

                ball_path_x.clear()
                ball_path_y.clear()
                time_list.clear()
                time_count = 0

            elif px == None:
                find_time_list.clear()
                ball_path_x.clear()
                ball_path_y.clear()
                time_list.clear()
                time_count = 0

        cv2.imshow('cam',frame)

        if cv2.waitKey(1) == 27:
            break

    dType.SetQueuedCmdStopExec(api)

    dType.DisconnectDobot(api)
    cap.release()
    cv2.destroyAllWindows()

            
main()
































    
