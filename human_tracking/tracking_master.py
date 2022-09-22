"""
file name : tracking_master
Details : main control centor of tracking module
"""

import human_tracking
from threading import Thread

"""
1. 메인 마스터로 부터 신호를 받음(flag)
2. flag 값이 0으로 내려가면 트래킹을 멈춰야됨
3. 트래킹을 멈추면 state가 0으로 내려감
4. 또 flag 값이 1로 올라가면 트래킹을 다시 실행함
5. 트래킹이 실행되면 state가 1로 올라감
"""

def tracking_control(flag, system):
    print("traking process start")
    state = 0 #if tracking == 1, else == 0

    while (system == 0):
        if(flag == 1):
            mainThread = Thread(target = human_tracking.tracking)
            mainThread.start()
            state = 1    
            return state
        else:
            mainThread.join()
            state = 0
            return state
    return

def main():
    tracking_control(1,0)   # test

if __name__ == "__main__":
    main()