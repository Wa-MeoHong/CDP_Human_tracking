"""
                             tool.py (v1.7.0)
 최초작성자 : 김민관 ( 2022.08.22 )
 마지막 수정자 : 신대홍 ( 2022.10.08 )
 수정사항 : tool.py에 vesc코드 및 Servo모터  관련  기능 추가(22.09.18)
           LED관련 코드 주석처리 및 stop()의 위치를 옮김(22.09.21)
           무게와 모터방향을 고려해 전,후진 수정 및 vesc포트위치 수정
           전류값 2배 증가, 왼쪽모터 역방향회전으로 변경, 각도 미세수정(22.09.22)
           VESC 포트 변경(반대로 바꿈), VESC 전류값 다운, backward(좌, 우) 주석처리
           zero()를 삭제함( init으로 역할을 통합시킴 (중앙으로 돌리기)) (22.10.01)
           Semiflag를 추가하여, 좀더 세부적인 서보모터 조정을 실현시킴 (22.10.03)
           tool.py를 클래스로 재구성하였음 (22.10.08)
"""
import RPi.GPIO as GPIO
import pyvesc as VESC
import serial
import time

# GPIO 동작 시작
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# GPIO 핀들과 delay
servopin = 18
delay = 0.05

#angle
left_angle = 8.25
center_angle = 9.75
right_angle = 11.25

class Motor :
    # __init__() : 초기값 설정 
    def __init__(self):
        # GPIO 스타트
        GPIO.setup(servopin, GPIO.OUT)
        self.p = GPIO.PWM(servopin, 60)         # 60Hz로 시작
        self.p.start(0)                         # duty_cycle은 0으로 설정
        
        # 좌우 flag변수
        self.flagright = 0
        self.flagleft = 0
        # VESC에 넣을 message
        self.msgleft = 0
        self.msgright = 0
        # 전류, 방향에 따른 추가 전류
        self.current = 500
        self.dircur = 100

        # VESC 초기 기동
        self.vesc1 = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=0.05)
        self.vesc2 = serial.Serial(port='/dev/ttyACM1', baudrate=115200, timeout=0.05)
    
    # setmsg(Cur, dir, flagleft, flagright) : flag에 따라서 메세지를 바꾼다. (인수로 Cur, dir를 넣음)
    def setmsg(self, Cur, dir, flagleft, flagright):
        if (flagleft == 1 and flagright == 0):
            self.msgleft = VESC.SetCurrent(Cur)
            self.msgright = VESC.SetCurrent(-1 * (Cur + dir))
        elif ( flagright == 1 and flagleft == 0 ):
            self.msgleft = VESC.SetCurrent(Cur + dir)
            self.msgright = VESC.SetCurrent(-1 * (Cur))
        else:
            self.msgleft = VESC.SetCurrent(Cur)
            self.msgright = VESC.SetCurrent(-1 * Cur)
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

    """
                                BLDC Motor functions
            1. go() : VESC를 통해 BLDC모터에 인코딩함 
            2. stop() : VESC에다가 전류를 0으로 집어넣어 멈춤
            3. forward() : 메세지를 left, right에 설정 후, VESC에 값을 집어넣음
            4. backward() : 전류는 반대로 돌리고, 메세지를 left, right에 설정 후, VESC에 값을 집어넣음
    """
    def go(self):                   # go() : BLDC를 움직임
        self.vesc1.write(VESC.encode(self.msgleft))
        self.vesc2.write(VESC.encode(self.msgright))
    def stop(self):                 # stop() : BLDC를 멈춤
        message = VESC.SetCurrent(0)
        self.vesc1.write(VESC.encode(message))
        self.vesc2.write(VESC.encode(message))
    def forward(self):              # forward() : 앞으로 Go
        self.current = 500          # backward()에 의해 바꾼 전류와 dircur를 원래대로 돌림
        self.dircur = 100
        self.setmsg(self.current, self.dircur, self.flagleft, self.flagright)
        self.go()
    def backward(self):             # backward() : 뒤로 Go
        self.current = -500         # 전류와 dircur를 forward의 반대로 돌림
        self.dircur = -100
        self.setmsg(self.current, self.dircur, self.flagleft, self.flagright)
        self.go()

    """
        Servo모터 작동 형태

        1) 센터 (init, Normal)
            - flagleft, flagright는 변경X, 서보모터 상태 : Normal
        2) 우회전 (right)
            - flagright = 1, flagleft = 0, 서보모터 상태 : right
        3) 좌회전 (left )
            - flagright = 0, flagleft = 1, 서보모터 상태 : left
    """
    
    """
                    Servo Motor functions
            1. setangle(angle) : 서보모터 조정 
            2. left(Semiflag) : 왼쪽으로 flag설정 및 서보모터 조정
            3. right(Semiflag) : 오른쪽으로 flag설정 및 서보모터 조정
            4. init() : 중앙이 가도록 서보모터 조정
    """
    def setangle(self, angle):
        self.p = ChangeDutyCycle(angle)
    def left(self, Semiflag):
        if (self.flagleft != 1):
            self.flagleft = 1
            self.flagright = 0
        else:
            Final_left = left_angle + (0.375 * Semiflag)
            self.setangle(Final_left)
    def right(self, Semiflag):
        if (self.flagright != 1):
            self.flagleft = 0
            self.flagright = 1
        else:
            Final_right = right_angle - (0.375 * Semiflag)
            self.setangle(Final_right)
    def init(self):
        self.setangle(center_angle)
