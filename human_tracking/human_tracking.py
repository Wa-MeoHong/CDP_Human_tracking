"""
Project: Human_tracking
작성자: 김민관
프로그램 내용
 - 이 프로그램은 카메라를 이용하여 사물을 감지하는 프로그램이다.
 - 모든 사물을 감지하고, person인 것을 찾는다.
 - person의 BBox 중앙에 점을 찍고, 화면의 좌우, BBox의 y축 높이에 따라 모터 출력을 제어한다.
 - 이 프로그램은 텐서 모듈과 openCV를 기반으로 제작되었다.
 - 텐서 모델은 moblienet_ssd_v2_coco 기계학슴 모듈을 사용한다.
 - 이 프로그램은 라즈베리파이에서 정상작동하도록 설계되어 있다.
 - 텐서 분석과 사용은 common2.py파일에서 진행하도록 한다.
 - GPIO설정은 tool.py에 정의되어있다.
 - 프로그램은 jiteshsaini가 진행한 AI robot프로젝트 도중에 사용된 object_tracking을 
 기반으로 작성되었으며, 원래 프로그램에서 하드웨어 가속과, 필요없는 부분들을 제거하고 최적화하였다.
"""

"""
version 1.0  - 기본적인 사물의 추적이 가능하며, 사람의 중앙에 점을 찍는다.
version 1.1  - 화면의 중앙에서 0.1크기만큼의 정지범위를 설정하고, 정지범위에서 사람이 벗어난 파라미터를 측정한다
version 1.1.1- GPIO설정이 추가되었다.
version 1.2  - 정지범위의 설정이 변경되었다.
version 1.3  - 사람의 BBox 설정에 따라 BBox의 가장 아래변을 y값으로 설정하여 정지범위를 다시 설정하였다.
version 1.4  - 정지범위의 설정이 변경되었다.
version 1.5  - GPIO설정이 수정되었다.
version 1.6  - TPU가속 모듈이 추가되었다.
latest version : 1.6.1
작성자 : 김민관
"""


import common2 as cm
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
import sys

import tool as tl

sys.path.insert(0, './')

#GPIO 이니셜라이징

cap = cv2.VideoCapture(0)
threshold = 0.2
top_k=2 #최대 오브젝트 갯수 5개
fps = 1
Stop_flag = 0
y_max = 0
x_min = 0
x_max = 0

#모델 주소
model_dir = './'
model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite' #텐서 모듈
label = 'coco_labels.txt' #라벨

stop_range = 0.1 #움직임이 시작되는 최소 범위
x_deviation = 0  #x축에서 얼마나 떨어져있는지 확인할 변수

valid_objects = 'person'
arr_track_data=[0,0,0,0,0,0]
tl.pre()
tl.init()


#------------------------------------------------------------
#원본파일에는 여기서 로봇의 모터 스피드를 초기화한다.
#------------------------------------------------------------

def track_object(objs, labels):
    global x_deviation, y_max, x_min, x_max, stop_range, arr_track_data
    y = 0
    if(len(objs)==0):
        print("no objects to trace")
        tl.stop()
        arr_track_data=[0,0,0,0,0,0]
        #GPIO 모터 정지
        return
    
    flag=0
    for obj in objs:
        object_label = labels.get(obj.id, 0) #사물의 아이디를 받아옴
        if(object_label == valid_objects): #만약 아이디가 사람이면 bbox정보를 받아옴
            x_min, y_min, x_max, y_max = list(obj.bbox)
            flag = 1
            break
    
    if(flag==0): #만약 감지된 사물중에 사람이 없다면 종료
        tl.stop()
        print("person Disappeared")
        return

    x_center = round((x_min + (x_max - x_min)/2), 3) #물제의 가로 중앙
    y_center = round((y_min + (y_max - y_min)/2), 3) #물체의 세로 중앙 =  (아래쪽 변 + 중심까지의 거리)

    x_deviation = round(0.5 - x_center, 3) #x축으로 부터 떨어진 거리

    
    
    x_right = 1-x_max
    print("right = ", x_right)
    x_left = 1-x_min
    print("left = ", x_left)
    thread1 = Thread(target = move_robot_BLDC)
    thread2 = Thread(target = move_robot_servo)
    thread1.start()
    thread2.start()
    #thread1.join()
    #thread2.join()

    arr_track_data[0]=x_center
    arr_track_data[1]=y_center
    arr_track_data[2]=x_deviation


#----------------BLDC모터를 움직이는 쓰레드-----------------------------------
def move_robot_BLDC():
    global x_deviation,  stop_range, Stop_flag, cmd, y_max, x_min, x_max
    delay = 1
    y=1-y_max #밑변이 위치하는 길이
    x_right = 1-x_max # 0이면 오른쪽에 닿음
    x_left = x_min # 1이면 왼쪽에 닿음
    print("y = ", y)
    state = 0 #1 = 전진중, 2 = 후진중 0 = 정지
    """
        1)BLDC모터를 움직이는 쓰레드
            1-1) y축이 양수 범위 에서 정지범위를 벗어남
                 - flag를 내린다.
                 - BLDC모터를 회전시켜 전진한다
            1-2) y축이 음수 범위 에서 정지범위를 벗어남
                 - flag를 올린다.
                 - BLDC모터를 회전시켜 후진한다.
            1-3) y축은 정지범위에 있다
                 - flag를 올린다.
                1-3-1) x축이 정지범위에서 벗어남
                    - BLDC모터를 회전시켜 조금 후진한다.(10cm정도)
                1-3-2) x축이 정지범위에 있음
                    - 정지
    """

    if((y>=(stop_range*2)) and state == 0): #y축이 양수 범위 에서 정지범위를 벗어남
        if(state == 2):
            time.sleep(delay)
        else:
            cmd = "forword"
            Stop_flag = 0 #flag를 내린다
            state = 1
            tl.forword()   #BLDC모터를 회전시켜 전진한다

    elif((y<=stop_range/2) and state == 0): #y축이 음수 범위 에서 정지범위를 벗어남
        if (state == 1):
            time.sleep(delay)
        else:
            cmd = "backword" 
            Stop_flag = 1   #flag를 올린다.
            #time.sleep(delay)#서보모터 조정하는 시간동안 딜레이
            state = 2
            tl.backword() #GPIO 모터 후진   #BLDC모터를 회전시켜 후진한다.

    else:  #y축은 정지범위에 있다
        cmd = "stop"
        Stop_flag = 1   #flag를 올린다.
        tl.stop() #일단 정지
        
        if((x_deviation>stop_range) or (x_deviation<-1*(stop_range))) : #x축이 정지범위에서 벗어남
            cmd = "revise" 
            time.sleep(delay)#서보모터 조정하는 시간동안 딜레이
            tl.backword() #GPIO 모터 약간 후진   #BLDC모터를 회전시켜 아주 조금 후진한다.
            time.sleep(delay*4)
            tl.stop()
        else:
            state = 0

    arr_track_data[4]=cmd
    arr_track_data[3]=y
    """
    전진하는 방법
     - 물체의 y값이 일정 범위 이상일때 물체는 앞으로 전진함
    
    정지해야될 때 
     - 물체의 y값이 일정 범위 이하일때 물체는 정지함

    서보모터가 움직이는 때
     - 물체가 전진하고 있을 때만 서보모터가 움직임
     - 물체의 중심이 일정 범위를 벗어나면 벗어난 방향으로 회전
    
    후진해야될 때
     - 물체의 y값이 0에 근접할 때 후진함
     - 정지상태에서 물체의 1-x_max값이 일정 범위를 벗어나면 후진
     - 정지상태에서 물체의 1-x_min값이 일정 범위를 벗어나면 후진

    y < 0.05 이하 일때 후진
    y > 0.05 and y < 0.25 일때 정지
    y > 0.25 에서 전진

    """
    
#-----------------서보모터를 움직이는 쓰레드------------------------------------
def move_robot_servo():
    global x_deviation, stop_range, Stop_flag
    delay = 0.5
    cmd = 0
    state = 0 #state = 1 : left ,  state = 2 : right state = 0 : normal
    """
    2)서보모터를 움직이는 쓰레드
            2-1) flag가 올라왔음
                 - 서보모터를 회전시켜 바퀴축을 11자로 만듦
            2-2) flag가 올라오지 않았음
                2-2-1) x축이 양수 범위에서 정지범위를 벗어남
                 - 서보모터를 회전시켜 바퀴를 시계방향으로 회전시킨다.
                    2-2-1-1) x축이 정지범위 안에 들어옴
                     - 서보모터를 회전시켜 바퀴를 11자로 만든다.

                2-2-1) x축이 음수 범위에서 정지범위를 벗어남
                 - 서보모터를 회전시켜 바퀴를 시계반대방향으로 회전시킨다.
                    2-2-1-1) x축이 정지범위 안에 들어옴
                     - 서보모터를 회전시켜 바퀴를 11자로 만든다.
    """

    #정지 또는 후진 상황에 사용되는 Stop_flag
    if(Stop_flag == 1): #Stop_flag가 올라왔음
        state = 0
        tl.zero#GPIO 서보모터를 원래대로 되돌림
        #time.sleep(delay)

    else:
        if(x_deviation>stop_range):
            if(state == 1):
                state = 1
                cmd = "left"
                #time.sleep(delay)
            else:
                state = 1
                cmd = "left"
                #tl.zero()#GPIO 서보모터를 회전시켜 바퀴를 11자로 만든다.
                time.sleep(delay)
                tl.left() #GPIO 서보모터를 회전시켜 바퀴를 시계방향으로 회전시킨다.
                #time.sleep(delay)

        elif(x_deviation<-1*(stop_range)):
            if(state == 2):
                state = 2
                cmd = "right"
                #time.sleep(delay)
            else:
                state = 2
                cmd = "right"
                #tl.zero()#GPIO 서보모터를 회전시켜 바퀴를 11자로 만든다.
                time.sleep(delay)
                tl.right() #GPIO 서보모터를 회전시켜 바퀴를 시계반대방향으로 회전시킨다.
                #time.sleep(delay)

        else:
            state = 0
            tl.zero()#GPIO 서보모터를 회전시켜 바퀴를 11자로 만든다.
            #time.sleep(delay)

    arr_track_data[5]=cmd


#--------------------------메인-----------------------------------------------
def tracking():
    interpreter, labels = cm.load_model(model_dir, model, label) #모델 불러오기
    
    arr_duration=[0,0,0] #[컨버트, 추론, 미리보기] 하는데 걸리는 시간
    while True:
        start_time=time.time() #시작 시간 계산 필요한가?

        #--------------------추론가능하게 이미지 변경-------------------------
        time_convert = time.time()

        ret, frame = cap.read() #정상작동하면  ret = true 실패하면 ret = false
        if not ret:
            print("someting wrong")
            break

        im = cv2.flip(frame, 0) #좌우 반전
        im = cv2.flip(im, 1) #상하 반전

        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #색을 삼원색 바탕으로 바꿈
        pil_im = Image.fromarray(im_rgb)  #이미지를 python image library 형태로 바꿈

        arr_duration[0]=time.time() - time_convert #컨버트하는데 걸리는 시간 계산
        #---------------------이미지 추론-----------------------------------
        time_inference = time.time()

        cm.set_input(interpreter, pil_im) #사진과 모델 입력
        interpreter.invoke() #연산 권한 인계
        objs = cm.get_output(interpreter, threshold, top_k) #추론 결과 받아옴
        
        arr_duration[1] = time.time() - time_inference #추론하는데 걸린 시간 측정
        #---------------------미리보기--------------------------------------
        time_preview=time.time()
        #im = cv2.flip(im, 0) #상하 반전

        track_object(objs, labels) #로봇을 움직이는 함수

        if cv2.waitKey(1) & 0xFF == ord('q'): #종료방법
            break
        #cv2.imshow('Preview', im)
        im = cm.draw_overlays(im, objs, labels, arr_duration, arr_track_data, stop_range) 
        cv2.imshow('Preview', im)

        arr_duration[2] = time.time() - time_preview #미리보기하는데 걸리는 시간 계산
        #--------------------fps계산----------------------------------------
        fps = round(1.0 / (time.time() - start_time), 1)

    tl.init()
    cap.release() #리소스 해제
    cv2.destroyAllWindows() #cv2모두 닫기 