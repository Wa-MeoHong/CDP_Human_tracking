# tool.py (v1.5.1)
# 최초작성자 : 김민관 ( 2022.08.22 )
# 마지막 수정자 : 신대홍 ( 2022.10.01 )
# 수정사항 : tool.py에 vesc코드 및 Servo모터  관련  기능 추가(22.09.18)
#           LED관련 코드 주석처리 및 stop()의 위치를 옮김(22.09.21)
#           무게와 모터방향을 고려해 전,후진 수정 및 vesc포트위치 수정
#           전류값 2배 증가, 왼쪽모터 역방향회전으로 변경, 각도 미세수정(22.09.22)
#           VESC 포트 변경(반대로 바꿈), VESC 전류값 다운, backward(좌, 우) 주석처리
#           zero()를 삭제함( init으로 역할을 통합시킴 (중앙으로 돌리기)) (22.10.01)

import RPi.GPIO as GPIO
import pyvesc as VESC
import serial
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

servopin = 18
delay = 0.05

#flag
flagright = 0
flagleft = 0

#angle
left_angle = 8.1
center_angle = 9.6
right_angle = 11.1

# 초기값 설정 pre()함수
def pre():
    global p, vesc1 , vesc2
    # vesc1(왼쪽), vesc2(오른쪽)
    # vesc1 = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=0.05)
    # vesc2 = serial.Serial(port='/dev/ttyACM1', baudrate=115200, timeout=0.05)

    GPIO.setup(servopin, GPIO.OUT)

    # 초기값 설정
    p = GPIO.PWM(servopin, 60)          # 60Hz로 시작
    p.start(0)                          # duty_cycle은 0으로 설정

""" 
    BLDC 모터의 작동 형태
    1) 전진
        1-1) 직진 전진
            - 모터의 전류 상태 : 왼쪽 = 오른쪽, 서보모터 : Normal
        1-2) 좌회전 전진
            - 모터의 전류 상태 : 왼쪽 < 오른쪽, 서보모터 : left
        1-3) 우회전 전진
            - 모터의 전류 상태 : 왼쪽 > 오른쪽, 서보모터 : right
            
    2) 후진
        2-1) 직진 후진
            - 모터의 전류 상태 : 왼쪽 = 오른쪽, 서보모터 : Normal
        2-2) 좌회전 후진
            - 모터의 전류 상태 : 왼쪽 < 오른쪽, 서보모터 : left
        2-3) 우회전 후진
            - 모터의 전류 상태 : 왼쪽 > 오른쪽, 서보모터 : right
        
"""

def forward(): #빨간불
    global flagright, flagleft
    #오른쪽, 왼쪽으로 회전하는것에 따라서 모터마다 속력을 다르게해
    #좀 더 회전이 잘 되게함
        
    # 전진(죄회전)
    if (flagleft == 1 and flagright == 0):
        msgleft = VESC.SetCurrent(500)
        msgright = VESC.SetCurrent(-600)

    # 전진(우회전)
    elif (flagleft == 0 and flagright == 1):
        msgleft = VESC.SetCurrent(600)
        msgright = VESC.SetCurrent(-500)
        
    # 전진(기본)
    else:
        msgleft = VESC.SetCurrent(500)
        msgright = VESC.SetCurrent(-500)
        
    # vesc1.write(VESC.encode(msgleft))
    # vesc2.write(VESC.encode(msgright))

def backward(): #하얀불
    global flagleft, flagright
    
    #  후진 (좌회전 방향)   
    if (flagleft == 1 and flagright == 0):
        msgleft = VESC.SetCurrent(-500)
        msgright = VESC.SetCurrent(600)
    #  후진 (우회전 방향)
    elif (flagright == 1 and flagleft == 1):
        msgleft = VESC.SetCurrent(-600)
        msgright = VESC.SetCurrent(500)
    # 기본
    else:
        msgleft = VESC.SetCurrent(-500)
        msgright = VESC.SetCurrent(500)

    # vesc1.write(VESC.encode(msgleft))
    # vesc2.write(VESC.encode(msgright))
        
"""
    Servo모터 작동 형태

    1) 센터 (init, Normal)
        - flagleft, flagright는 변경X, 서보모터 상태 : Normal
    2) 우회전 (right)
        - flagright = 1, flagleft = 0, 서보모터 상태 : right
    3) 좌회전 (left )
        - flagright = 0, flagleft = 1, 서보모터 상태 : left
"""
# 좌회전, 우회전이 종료된 후, Servo 초기화

def right(): #파란불
    global flagright, flagleft

    if (flagright != 1 ):
        flagright = 1
        flagleft = 0  
        #time.sleep(delay)
    else:
        setangle(right_angle)
        #time.sleep(0.5)

def left(): #녹색불
    global flagleft, flagright

    if (flagleft != 1 ):    
        flagleft = 1
        flagright = 0
        #time.sleep(delay)
    else:
        setangle(left_angle)
        #time.sleep(0.5)

#초기값 설정
def init():
    setangle(center_angle)
    

# 정지
def stop():
    #정지
    message = VESC.SetCurrent(0)
    # vesc1.write(VESC.encode(message))
    # vesc2.write(VESC.encode(message))

def setangle(angle):
    p.ChangeDutyCycle(angle)
    # time.sleep(0.5)


#def main():
    #global flagleft, flagright
    #pre()
    #time.sleep(1)
    #right()
    #time.sleep(1)
    #left()
    #time.sleep(1)
    #init()

#if __name__ == "__main__" :
    #main()
