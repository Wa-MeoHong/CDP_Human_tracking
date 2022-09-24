# tool.py (v1.5.0)
# 최초작성자 : 김민관 ( 2022.08.22 )
# 마지막 수정자 : 신대홍 (2022.09.22)
# 수정사항 : tool.py에 vesc코드 및 Servo모터  관련  기능 추가(22.09.18)
#           LED관련 코드 주석처리 및 stop()의 위치를 옮김(22.09.21)
#           무게와 모터방향을 고려해 전,후진 수정 및 vesc포트위치 수정
#           전류값 2배 증가, 왼쪽모터 역방향회전으로 변경, 각도 미세수정(22.09.22)
#           VESC 포트 변경(반대로 바꿈), VESC 전류값 다운, backward(좌, 우) 주석처리
#               \ ChangeFrequency를 추가해봄

import RPi.GPIO as GPIO
import pyvesc as VESC
import serial
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

servopin = 18
delay = 0.05

#flag
flagstart = 0
flagright = 0
flagleft = 0

#angle
left_angle = 6.5
center_angle = 8.5
right_angle = 10.5

def pre():
    global p, vesc1 , vesc2
    # vesc1(왼쪽), vesc2(오른쪽)
    vesc1 = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=0.05)
    vesc2 = serial.Serial(port='/dev/ttyACM1', baudrate=115200, timeout=0.05)

    # GPIO.setup(17,GPIO.OUT)         # 빨간불
    # GPIO.setup(27,GPIO.OUT)         # 파란불
    # GPIO.setup(22,GPIO.OUT)         # 녹색불
    # GPIO.setup(23,GPIO.OUT)         # 하얀불
    GPIO.setup(servopin, GPIO.OUT)

    # 초기값 설정
    p = GPIO.PWM(servopin, 60)       #60Hz로 시작
    p.start(0)                    # duty_cycle은 0으로 설정
    p.ChangeFrequency(0)            # 0Hz로 초기화


def forword(): #빨간불
    global flagright, flagleft
    #오른쪽, 왼쪽으로 회전하는것에 따라서 모터마다 속력을 다르게해
    #좀 더 회전이 잘 되게함

    # GPIO.output(17,True)
    # time.sleep(delay)

    #전진(기본, global변수 왼쪽, 오른쪽 없을 때)
    if (flagleft == 0 and flagright ==0):
        msgleft = VESC.SetCurrent(-500)
        msgright = VESC.SetCurrent(500)
        vesc1.write(VESC.encode(msgleft))
        vesc2.write(VESC.encode(msgright))
        
    #전진(죄회전)
    elif (flagleft == 1 and flagright == 0):
        msgleft = VESC.SetCurrent(-450)
        msgright = VESC.SetCurrent(500)
        vesc1.write(VESC.encode(msgleft))
        vesc2.write(VESC.encode(msgright))

    #전진(우회전)
    elif (flagleft == 0 and flagright == 1):
        msgleft = VESC.SetCurrent(-500)
        msgright = VESC.SetCurrent(450)
        vesc1.write(VESC.encode(msgleft))
        vesc2.write(VESC.encode(msgright))


def backword(): #하얀불
    global flagright, flagleft
    # GPIO.output(23,True)
    # time.sleep(delay)
    
    #  후진 (기본)
    if (flagleft == 0 and flagright ==0):
        msgleft = VESC.SetCurrent(500)
        msgright = VESC.SetCurrent(-500)
        vesc1.write(VESC.encode(msgleft))
        vesc2.write(VESC.encode(msgright))
        
    # # 후진 (전진방향이 좌회전)
    # elif (flagleft == 1 and flagright == 0):
    #     msgleft = VESC.SetCurrent(450)
    #     msgright = VESC.SetCurrent(-500)
    #     vesc1.write(VESC.encode(msgleft))
    #     vesc2.write(VESC.encode(msgright))
    
    # # 후진 (전진방향이 우회전)
    # elif (flagleft == 0 and flagright == 1):
    #     msgleft = VESC.SetCurrent(500)
    #     msgright = VESC.SetCurrent(-450)
    #     vesc1.write(VESC.encode(msgleft))
    #     vesc2.write(VESC.encode(msgright)) 


# 좌회전, 우회전이 종료된 후, Servo 초기화
def zero():
    global flagleft, flagright
    # 만약 flagleft(좌회전)이 1이라면 0으로 바꾼다.
    if(flagleft == 1 ):
        flagleft = 0
        # GPIO.output(22,False)
        #time.sleep(delay)
        setangle(center_angle)

    # 만약 flagright(우회전)이 1이라면 0으로 바꿈
    if(flagright == 1):
        flagright = 0
        # GPIO.output(27,False)
        #time.sleep(delay)
        setangle(center_angle)



def right(): #파란불
    global flagright, flagleft

    # GPIO.output(27,True)

    if (flagright == 0 ):
        flagright = 1
        flagleft = 0  
        #time.sleep(delay)
        setangle(right_angle)
        #time.sleep(0.5)

def left(): #녹색불
    global flagleft, flagright
    
    # GPIO.output(22,True)

    if (flagleft == 0 ):    
        flagleft = 1
        flagright = 0
        #time.sleep(delay)
        setangle(left_angle)

        #time.sleep(0.5)

#초기값 설정
def init():

    # GPIO.output(22,False)
    # GPIO.output(17,False)
    # GPIO.output(27,False)
    # GPIO.output(23,False)
    setangle(center_angle)

# 정지
def stop():
    # GPIO.output(22,False)
    # GPIO.output(17,False)
    # GPIO.output(27,False)
    # GPIO.output(23,False) 
    #time.sleep(delay)

    #정지
    message = VESC.SetCurrent(0)
    vesc1.write(VESC.encode(message))
    vesc2.write(VESC.encode(message))

def setangle(angle):
    p.ChangeFreaquency(60)
    p.ChangeDutyCycle(angle)
    p.ChangeFreaquency(0)


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