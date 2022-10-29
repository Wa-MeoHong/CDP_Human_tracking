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
version 1.6.2 - backward()실행 시, ServoMotor가 중앙으로 오도록 tl.zero()추가 (신대홍)
version 1.6.3 - edgetpu 파일 경로를 제대로 찾을 수 있게 os라이브러리 추가 및 경로 수정 (신대홍)
version 1.6.4 - 정지 range에서 작용되는 경우를 3가지에서 4가지로 늘림 (revise 상태를 추가) (신대홍)
version 1.7.0 - 사람이 2인 이상 잡히게 되면 이전의 사람을 계속 추적할 수 있게 알고리즘을 추가
version 1.8.0 - motor_BLDC를 수정함 (Stop_flag를 먼저 수정 후, 작동하게끔 함) ( 22.10.01 )
version 1.8.1 - Stop_flag를 3가지로 만들어 반대회전을 실현시킴, stop_range를 늘림 (22.10.02)
version 1.9.0 - Semiflag를 추가하여 좀 더 세부적인 서보모터 조정을 실현시킴 (22.10.03)
version 2.0.0 - Class로 재구성 (Class Tracking) (22.10.29)

latest version : 2.0.0
작성자 : 김민관, 신대홍
"""

import common2 as cm
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
import sys, os
import keyboard

import face as fc
from tool import Motor

#------------------------------------------------------------
#                       DEFINE CONSTANT  
#------------------------------------------------------------

    #모델 주소
model_dir = os.path.dirname(os.path.realpath(__file__))          # 이 파일을 제외한 경로(realpath)를 가져옴(dirname)
model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite' # 텐서 모듈
label = 'coco_labels.txt'                                        # 라벨

    # 유효한 오브젝트
valid_objects = 'person'

    #GPIO 핀 주소
servopin = 18

    #추론 결과에서 사용되는 상수값
threshold = 0.2
top_k=2                                         # 최대 오브젝트 갯수 5개
stop_range = 0.18                               # 움직임이 시작되는 최소 범위
    
    #cmd
cmd = ""
    
    #딜레이
delay = 0.5
    # 트랙 주소 데이터 
arr_track_data=[0,0,0,0,0,0]

    # Tracking (Class)
class Tracking :
    def __init__(self):
        # 비디오 캡쳐 시작
        self.cap = cv2.VideoCapture(0)
        # flag들
        self.Stop_flag = 0                                   # 모터에서 사용하는 Flag
        self.BLDC_state = 0                                  # BLDC 상태: 0 = 정지, 1 = 전진중, 2 = 후진중 
        self.Servo_state = 0                                 # Servo 상태 : 0 = 노말, 1 = 정지
        self.Semiflag = 0

        # 화면의 좌표값들 
        self.y_max = 0
        self.x_min = 0
        self.x_max = 0
        self.x_deviation = 0                              # x축에서 얼마나 떨어져있는지 확인할 변수

        self.tl = Motor(servopin)                         # 처음 객체를 생성하게 되면 Motor를 init을 한다. 
        self.tl.inits()

    # 타겟을 찾음 ( 두개 이상의 타겟 중 하나를 선정)
    def find_target(self, obj, tempFlag):                                 
        obj_position = [0, 0, 0]                                    # 오브젝트 중심 위치
        obj_x_center = 0                                            # 오브젝트 중심 x위치 
        obj_y_center = 0                                            # 오브젝트 중심 y위치
        obj_x_min, obj_y_min, obj_x_max, obj_y_max = list(obj.bbox) # Bbox의 데이터(x_min, y_min, x_max, y_max)
        obj_x_center = round((obj_x_min + (obj_x_max - obj_x_min)/2), 3) #x축 중앙
        obj_y_center = round((obj_y_min + (obj_y_max - obj_y_min)/2), 3) #y축 중앙
        obj_position = [obj_x_center, obj_y_center, tempFlag]       # 리스트에 대입
        return obj_position                                         # obj_중심의 데이터 반환 
    def obj_area_check(self, obj, tempFlag):                              
        area = [0, 0]                                               # 넓이 리스트 
        obj_x_min, obj_y_min, obj_x_max, obj_y_max = list(obj.bbox) # 오브젝트 박스의 길이 꼭지점 정보
        width = obj_x_max - obj_x_min                               # 가로 (width)
        height = obj_y_max - obj_y_min                              # 세로 (height)
        area = [width * height, tempFlag]                           # 넓이 대입 [넓이, 순번]
        return area                                                 # 반환  # Bbox의 넓이 체크
    def resetPerson(self, objs, labels):
        tempFlag = 0
        temp_obj_center = [] #이번 프레임에서 얻은 정보 
        position_difference = []
        temp_obj_position = []

        for obj in objs:
            object_label = labels.get(obj.id, 0)                        # 사물의 아이디를 받아옴
            if(object_label == valid_objects):                          # 만약 아이디가 사람이면 bbox정보를 받아옴 
                temp_obj_center.append(self.find_target(obj, tempFlag))      # 이번 프레임에서 받아온 "person" 라벨의 중앙 위치 좌표
                temp_x_min, temp_y_min, temp_x_max, temp_y_max = list(obj.bbox)             # 일단 좌표를 받아둠
                temp_obj_position.append([temp_x_min, temp_y_min, temp_x_max, temp_y_max])  # 좌표 임시 보관
                

        for target_position in temp_obj_center:                         # 차이를 계산
            x_diff = target_position[0]- arr_track_data[0]              # x축에서의 차이
            y_diff = target_position[1] - arr_track_data[1]             # y축에서의 차이
            # 수정 ( append 추가)
            position_difference.append([x_diff + y_diff, target_position[2]]) # 차이를 더함 (이전에 따라가던 목표라면 차이를 더한값이 젤 작을 것임)

        position_difference.sort(key=lambda x:x[0])                     # 차이를 더한 값을 기준으로 오름차순 정렬
        target = position_difference[0][1]                              # tempFlag를 가지고옴
        
        #temp_x_min, temp_y_min, temp_x_max, temp_y_max
        #x_min = temp_obj_position[target][0]
        #y_min = temp_obj_position[target][1]
        #x_max = temp_obj_position[target][2]
        #y_max = temp_obj_position[target][3]
        return temp_obj_position[target]

        # x_min, y_min, x_max, y_max = list(temp_obj_position[target]) #위치 할당
        # x_center = temp_obj_center[tempFlag][0]
        # y_center = temp_obj_center[tempFlag][1]
        # #x_min, y_min, x_max, y_max = list(obj.bbox)
        # targetPosition = [x_max - x_min, y_max - y_min] #저장
        # print("Target setted")


        # return targetPosition


    """
    전진하는 방법
    - 물체의 y값이 일정 범위 이상일때 물체는 앞으로 전진함
    
    정지해야될 때 
    - 물체의 y값이 일정 범위 이하일때 물체는 정지함

    서보모터가 움직이는 때
    - 물체가 전진하고 있을 때만 서보모터가 움직임
    - 물체의 중심이 일정 범위를 벗어나면 벗어난 방향으로 회전
    
    revise (정지 상태에서 사람을 따라다니는 때)
    - 제자리 회전이 불가능하므로 전진(좌회전)하기 위해 먼저 후진을 한다.
    - 움직일 만큼 후진했다면 다시 회전을 한다. 

    후진해야될 때
    - 물체의 y값이 0에 근접할 때 후진함
    - 정지상태에서 물체의 1-x_max값이 일정 범위를 벗어나면 후진
    - 정지상태에서 물체의 1-x_min값이 일정 범위를 벗어나면 후진

    Stop_flag : 서보모터, BLDC모터를 돌릴 때, 멈추는 트리거


    y < 0.05 이하 일때 후진
    y > 0.05 and y < 0.25 일때 정지
    y > 0.25 에서 전진

    """
    """
    Stop_Flag : 서보모터, BLDC공용 ( 정지(0), 전진(1), 후진(2) )
    BLDC_state : 정지(0), 전진(1), 후진(2)
    
    1)BLDC모터를 움직이는 쓰레드
        1) Stop_flag를 수정함 (전진범위에선 0, 나머지는 1)

        2) BLDC_state를 통해 전진, 후진, 정지 상태를 보고있음
            2-1) y가 stop_range * 2 ( 전진범위 )에 있음
                - BLDC_state = 1이 아니면 1로 변경
                - tool.py의 forward()를 실행
            2-2) y가 stop_range / 2 ( 후진범위 )에 있음
                - BLDC_state = 2가 아니면 2로 변경
                - tool.py의 backward()를 실행
            2-3) y가 정지범위에 있음
                2-3-1) y는 정지범위, x는 회전범위에 있을 때(서보모터 돌아감)
                    - 서보모터는 중앙으로 바꾸기 
                    - 후진을 하여 좌/우회전 할 공간을 확보함
                    - 다시 전진하여 사람을 중앙으로 맞춘다.
                2-3-2) x, y 전부 정지범위에 있음 (서보모터가 중앙)
                    - BLDC_state를 0로 강제로 변환
                    - tool.py의 stop()을 실행
    """

    def CheckGotype(self, y):                               # Stop_Flag를 조정하는 함수
        if (y >= (stop_range * 1.2)):                     # 전진 상태 (y >= stop_range * 1.8)
            self.Stop_flag = 1   
        elif (y <= (stop_range / 1.2)):                   # 후진 상태 (y <= stop_range / 2)
            self.Stop_flag = 2                  
        else:                                             # 그 이외의 상태 (y가 정지범위)
            self.Stop_flag = 0

    def move_robot_BLDC(self):                              # BLDC 모터 동작 함수 ( 불러오는 함수 )
        global cmd
        cmd = ""

        y=1-(self.y_max)                          # 밑변이 위치하는 길이
        print("y = ", y)

        # BLDC 상태: 1 = 전진중, 2 = 후진중 0 = 정지        
        # 먼저 Stop_flag를 먼저 설정함 ( 전진할 때만 Stop_flag가 내려가고, 다른 상태는 올라감)
        
        self.CheckGotype(y)
        self.SetBLDC()

        arr_track_data[4]=cmd
        arr_track_data[3]=y
    def SetBLDC(self):                                      # BLDC 모터를 동작 결정 함수
        global cmd
        # forward ( 전진 )
        if (self.Stop_flag == 1):
            cmd = "forward"
            self.BLDC_state = 1
            self.tl.forward()
        # backward ( 후진 )
        elif (self.Stop_flag == 2):
            cmd = "backward"
            self.BLDC_state = 2
            self.tl.backward()
        # Stop range (정지 범위에 있다면)
        else:   
            self.BLDC_state = 0
            self.tl.stop()
            if((self.x_deviation > stop_range) or (self.x_deviation < -(stop_range))):
                cmd = "revise"
                self.BLDC_state = 2
                self.tl.backward()                       # BLDC_state = 2, backward   


    """
    2) 서보모터를 움직이는 쓰레드
        - Stop_flag => 정방향 (0), 역방향 (1), 고정 (2)
        - Servo_flag => 센터 (0), 좌회전 (1), 우회전 (2)
    
        2-1) Stop_flag == 0 (정방향회전) 
            1. x_deviation(bBox중앙)가 stop_range보다 클 경우 (좌회전)
            2. x_deviation(bBox중앙)가 -stop_range보다 작을 경우 ( 우회전 )
            3. 중앙점이 1, 2사이인 경우 ( 중앙 초기화 )
        2-2) Stop_flag == 1 (역방향회전, 후진할때 필요)
            1. 2-1의 1.상황일경우 우회전
            2. 2-1의 2.상황일경우 좌회전
            3. 초기화
        2-3) Stop_flag == 2 (중앙 초기화)
            - 중앙으로 초기화
    """

    def move_robot_servo(self):                             # 서보모터 동작 함수 ( 불러오는 함수 )
        x_dot = self.x_deviation

        # Stop_flag = 1 은 전진상태, 즉, 정회전
        if (self.Stop_flag == 1):
            # 얼마나 각도를 조정해야하는지 Semiflag를 설정함
            self.CheckSemiangle(x_dot)
            self.SetServo(x_dot)   

        # Stop_flag = 0 은 정지상태, 역회전
        elif (self.Stop_flag == 0):
            self.CheckSemiangle((-1*x_dot))
            self.SetServo((-1*x_dot)) 

        # Stop_flag = 2 은 후진상태, 무조건 중앙으로 와야됨
        else:
            self.Servo_state = 0
            cmd = "center"
            self.tl.init()
            time.sleep(delay)
        
        arr_track_data[5]=cmd
    def CheckSemiangle(self, x_dot):                        # 각도를 얼만큼 조절할 지 Semiflag를 조정하는함수
        if ((x_dot > (stop_range / 2)) or (x_dot < -(stop_range / 2))):
            if ((x_dot > (stop_range * 2)) or (x_dot < -(stop_range * 2))):
                self.Semiflag = 0
            elif (x_dot > (stop_range * 1.4) or (x_dot < -(stop_range * 1.4))):
                self.Semiflag = 1
            elif (x_dot > (stop_range / 1.3) or (x_dot < -(stop_range / 1.3))):
                self.Semiflag = 2
            else:
                self.Semiflag = 3
        else:
            self.Semiflag = 4
    def SetServo(self, x_dot):                              # 서보모터 동작 결정 함수
        # 좌회전
            if ((x_dot > (stop_range / 2))):
                self.Servo_state = 1
                cmd = "left"
                self.tl.left(self.Semiflag)               # 대입
                time.sleep(delay)

            # 우회전
            elif ((x_dot< -(stop_range / 2))):
                self.Servo_state = 2
                cmd = "right"
                self.tl.right(self.Semiflag)
                time.sleep(delay)
            
            # 중앙
            else:
                self.Servo_state = 0
                cmd = "center"
                self.tl.init()
                time.sleep(delay)
    

#------------------------------------------------------------
#                       main Function 
#------------------------------------------------------------

Trk = Tracking()

# 트래킹 오브젝트
def track_object( Trk, objs, labels):                       # 오브젝트를 판별하고, 발견하면 사람을 따라감
    # global x_deviation, y_max, x_min, x_max, stop_range, arr_track_data, Stop_flag
    
    # if(len(objs)==0):                                      # GPIO 모터 정지
    #     print("no objects to trace")
    #     Trk.tl.stop()
    #     arr_track_data=[0,0,0,0,0,0]
    #     return

    Person_Flag = 0                               # 물체는 잡히나 사람이 없을 때에도 정지
    for obj in objs:
        object_label = labels.get(obj.id, 0)    # 사물의 아이디를 받아옴
        if(object_label == valid_objects):      # 만약 아이디가 사람이면 bbox정보를 받아옴
            Person_Flag = Person_Flag + 1
    
    if (Person_Flag == 0 or len(objs) == 0):                # GPIO 모터 정지
        print("no person to trace")
        Trk.tl.stop()
        arr_track_data=[0,0,0,0,0,0]
        #GPIO 모터 정지
        return                                              # 리턴으로 탈출
    
    targetBoxData = Trk.resetPerson(objs, labels)
    x_min, y_min, x_max, y_max = targetBoxData
    x_center = round((x_min + (x_max - x_min)/2), 3)            # 물제의 가로 중앙
    y_center = round((y_min + (y_max - y_min)/2), 3)            # 물체의 세로 중앙 =  (아래쪽 변 + 중심까지의 거리)
    x_deviation = round(0.5 - x_center, 3)                      # x축으로 부터 떨어진 거리
    
    x_right = 1-x_max
    print("right = ", x_right)
    x_left = 1-x_min
    print("left = ", x_left)

    print ("Stop_Flag = ", Trk.Stop_flag)
    # 모터 쓰레드 시작
    thread1 = Thread(target = Trk.move_robot_BLDC)
    thread2 = Thread(target = Trk.move_robot_servo)
    thread1.start()
    thread2.start()
    #thread1.join()
    #thread2.join()

    arr_track_data[0]=x_center
    arr_track_data[1]=y_center
    arr_track_data[2]=x_deviation

#---------------------------------메인-----------------------------------------------
def tracking():                                             # Main Tracking Function
    interpreter, labels = cm.load_model(model_dir, model, label) # 모델 불러오기
    
    arr_duration=[0,0,0]                        # [컨버트, 추론, 미리보기] 하는데 걸리는 시간
    while True:
        start_time=time.time()                  # 시작 시간 계산 필요한가?

        #--------------------추론가능하게 이미지 변경-------------------------
        time_convert = time.time()

        ret, frame = Trk.cap.read()                 # 정상작동하면  ret = true 실패하면 ret = false
        if not ret:
            print("someting wrong")
            break

        im = cv2.flip(frame, 0)                     # 좌우 반전
        im = cv2.flip(im, 1)                        # 상하 반전

        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    # 색을 삼원색 바탕으로 바꿈
        pil_im = Image.fromarray(im_rgb)                # 이미지를 python image library 형태로 바꿈

        arr_duration[0]=time.time() - time_convert      # 컨버트하는데 걸리는 시간 계산
        #---------------------이미지 추론-----------------------------------
        time_inference = time.time()

        cm.set_input(interpreter, pil_im)               # 사진과 모델 입력
        interpreter.invoke()                            # 연산 권한 인계
        objs = cm.get_output(interpreter, threshold, top_k)     # 추론 결과 받아옴
        
        arr_duration[1] = time.time() - time_inference          # 추론하는데 걸린 시간 측정
        #---------------------미리보기--------------------------------------
        time_preview=time.time()
        #im = cv2.flip(im, 0)                           # 상하 반전

        track_object(objs, labels)                      # 로봇을 움직이는 함수

        if keyboard.is_pressed("q"):                    # 종료방법
            break

        # if cv2.waitKey(1) & 0xFF == ord('q'):           # 종료방법
        #     break


        # 스크린 보여주기
        # cv2.imshow('Preview', im)
        # im = cm.draw_overlays(im, objs, labels, arr_duration, arr_track_data, stop_range) 
        # cv2.imshow('Preview', im)
        # thread3 = Thread(target=fc.BomiFace)
        # thread3.start()

        arr_duration[2] = time.time() - time_preview    # 미리보기하는데 걸리는 시간 계산
        #--------------------fps계산----------------------------------------
        fps = round(1.0 / (time.time() - start_time), 1)

    Trk.tl.inits()
    del(Trk.tl)
    Trk.cap.release()                                   # 리소스 해제
    cv2.destroyAllWindows()                             # cv2모두 닫기 
