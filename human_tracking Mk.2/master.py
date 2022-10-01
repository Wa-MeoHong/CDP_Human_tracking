import time, os
import sys

""" master.py ( 첫 부팅시 실행되는 파일 ) 
    설명 : 첫 부팅시 실행되는 master.py ( 이를 통해 cmd로 tracking_master.py 실행)
    프로그램 작성자 : 신대홍 (Ver.1.0.0)
    수정사항 : 1. 
"""

local_path = os.path.dirname(os.path.realpath(__file__))    # 이 파일이 있는 디렉토리경로를 얻는다
print("local_path : ", local_path)

status = sys.argv[1]                # 전원이 인가되어있는가? 를 가져온다
file_name = 'tracking_master.py'    # 파일이름은 tracking_master.py로 한다.

# 전원이 켜졌다면 tracking_master.py를 실행한다.
if (status == "1"):
    print("Starting Human tracking Module")
    cmd = "python" + local_path + "/" + file_name + " &"
    print("cmd: ", cmd)
    os.system(cmd)                  # cmd 명령을 실행하는 코드
    time.sleep(1)
    
# 만약 전원이 off되면 파일을 강제종료 시킨다.
if (status == "0"):                 
    cmd = "sudo pkill -f " + file_name
    os.system(cmd)
    
