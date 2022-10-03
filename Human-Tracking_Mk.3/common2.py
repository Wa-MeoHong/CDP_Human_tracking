"""
common2.py파일은 jiteshsaini가 AI로봇 제작을 진행하면서 작업한
object_detection프로젝트의 common1.py를 수정하고 보완하여 만들었으며, 
모든 내용은 jiteshsaini가 제작한것과 동일하게 구글이 지원하는 Object Detection example을
기반으로 설계되었다. (https://github.com/google-coral/examples-camera/tree/master/opencv)
common1.py는다음과 같은 작업을 수행한다.
1. 텐서 파일을 분석하거나 삽입한다.
2. 프로그램이 작동하는데 부가적인 역할을 수행한다.
3. BBox를 구현한다.
4. openCv2를 통한 비디오 수정작업을 진행한다.
"""
import numpy as np  #텐서파일을 분석하기 위한 넘파이
from PIL import Image #이미지 처리
import tflite_runtime.interpreter as tflite #텐서파일에서 모델을 불러오기 위한 텐서라이트
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

"""
def make_interpreter_0(model_file): #인터프리터 생성 (일반 모듈)
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(model_path=model_file)

"""
def make_interpreter_1(model_file): #인터프리터 생성 (가속 모듈)
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {}) #edeTPU 데이터 디바이스에서 받아옴
      ])


def set_input(interpreter, image, resample=Image.NEAREST):
    """인풋된 텐서플로우에서 데이터를 복사한다."""
    image = image.resize((input_image_size(interpreter)[0:2]), resample) #높이랑 너브를 기본값으로 설정,
    input_tensor(interpreter)[:, :] = image

def input_image_size(interpreter):
    """리턴된 삽입이미지를 튜플로 바꾸어 저장한다.(width, height, channels)"""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels

def input_tensor(interpreter):
    """리턴된 인풋 텐서를 numpy array형태로 다시 저장한다.(height, width, 3)"""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def output_tensor(interpreter, i):
    """한번 양자화된 데이터라면 양자화를 해제한다."""

    output_details = interpreter.get_output_details()[i] #모델출력 텐서의 세부 정보를 가지고옴
    output_data = np.squeeze(interpreter.tensor(output_details['index'])()) #실제로 사용중인 배열의 차원수를 줄여줌
    """만약 3차원 배열일때, 반드시 n차원일 필요가 없다면 명시적으로 분석하기 편한 n-x (x<n) 차원배열로 바꿀수 있다.
    ex) [[1,2,3]] => [1,2,3] (2차원배열이지만 굳이 2차원배열일 필요가 없기에 1차원 배열로 바꾸었다)
    """

    if 'quantization' not in output_details: #만약 세부정보가 양자화 되지 않았다면 바로 데이터를 반환
        return output_data

    scale, zero_point = output_details['quantization'] #양자화된 데이터의 scale(범위)와 zero_point(영점)을 받아온다.
    if scale == 0: #만약 범위가 양자화를 하나 마나 똑같다면 
        return output_data - zero_point #데이터를 영점에서 뺀 값을 돌려줌

    return scale * (output_data - zero_point) #위의 상황이 아니라면 범위와 영점을 이용하여 원래 값으로 양자화 해제한다.

import time
def time_elapsed(start_time,event):
        """연산되는 시간을 계산"""
        time_now=time.time()
        duration = (time_now - start_time)*1000
        duration=round(duration,2)
        print (">>> ", duration, " ms (" ,event, ")")


import os
def load_model(model_dir,model, lbl):
    """모델을 불러옴"""
    print('Loading from directory: {} '.format(model_dir))
    print('Loading Model: {} '.format(model))
    print('Loading Labels: {} '.format(lbl))
    
    model_path=os.path.join(model_dir,model) #모델이 저장된 주소
    labels_path=os.path.join(model_dir,lbl) #라벨이 저장된 주소
    
    interpreter = make_interpreter_1(model_path)  #가속모듈사용
    
    interpreter.allocate_tensors() #텐서를 초기화 시킴

    labels = load_labels(labels_path)  #라벨을 받아옴

    return interpreter, labels #받아온 인터프리터(기계학습 모델)과 라벨을 반환
    
import re
def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)') # 정규식으로 반환된거 내용을 아래에서 match함수로 긁어올 예정
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines()) #라벨 폴더에서 한줄씩 긁어옴
       return {int(num): text.strip() for num, text in lines} #한줄씩 긁어온 내용의 복사본을 반환(딕셔너리)
       



#----------------------------------------------------------------------
import collections
"""
튜플 서브 클래스를 사용하기 위하여 collections를 import
튜플 서브 클래스 : namedtuple(튜플 이름, [원소]) 형태로 사용
튜플처럼 쓸수도 있고, 클래스 처럼 쓸수 도있음
자료의 열거형을 마치 클래스를 사용하는 것 처럼 쓸수 있음
예를 들어 namedtuple을 이용하여 x와 y를 표현하고 싶다면 하드코딩하는 대신 
이름.x 또는 이름.y와 같이 표현할 수 있음
"""

Object = collections.namedtuple('Object', ['id', 'score', 'bbox']) #오브젝트 라는 튜플 서브 클래스

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """테두리가 있는 박스를 클래스화 (BBox = Boundary Box)
    openCV를 통해 비디오에서 오브젝트를 감싸는데 사용될것
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = output_tensor(interpreter, 0)
    class_ids = output_tensor(interpreter, 1)
    scores = output_tensor(interpreter, 2)
    count = int(output_tensor(interpreter, 3))
    #박스의 크기, 오브젝트의 아이디(사물의 이름), 얼마나 비슷한지, 몇개 인지

    def make(i): #박스의 크기 제단한 내용을 리턴할 것인데 아래의 make 함수를 재귀하여 사용
        #(재귀하기 때문에 지역변수 사용을 위해 get_output 함수 안에 작성)
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold] #재귀해서 리턴시킴
#--------------------------------------------------------------------


import cv2

def draw_overlays(cv2_im, objs, labels, arr_dur, arr_track_data, stop_range1):
    height, width, channels = cv2_im.shape
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    stop_range = stop_range1

    #위쪽 검은색 배젤-------------------------------------------------------------------
    cv2_im = cv2.rectangle(cv2_im, (0,0), (width, 24), (0,0,0), -1)

    #작동시간 체크----------------------------------------------------------------------
    cam=round(arr_dur[0]*1000.0)
    inference=round(arr_dur[1]*1000,0)
    preview=round(arr_dur[2]*1000,0)

    text_dur = 'Camera: {}ms    Inference: {}ms   other: {}ms'.format(cam,inference,preview)
    cv2_im = cv2.putText(cv2_im, text_dur, (int(width/4)-30, 16),font, 0.4, (255,255,255), 1)
    
    #FPS작성---------------------------------------------------------------------------
    total_duration=cam+inference+preview
    fps=round(1000/total_duration,1)

    text1 = 'FPS: {}'.format(fps)
    cv2_im = cv2.putText(cv2_im, text1, (10,20), font, 0.7, (150, 150, 255), 2)

    #아래쪽 검은색 배젤------------------------------------------------------------------
    cv2_im = cv2.rectangle(cv2_im, (0,height-24), (width, height), (0,0,0), -1)
    cv2.imshow('Preview', cv2_im)
    #허용범위와 오차범위 작성-------------------------------------------------------------
    str_tol='Tol : {}'.format(stop_range)
    cv2_im = cv2.putText(cv2_im, str_tol, (10, height-8), font, 0.55, (150, 150, 255), 2)
    
    #오차에 따른 오차 표현 변화-----------------------------------------------------------
    x_dev=arr_track_data[2]
    str_x='X: {}'.format(x_dev) # x축으로 떨어진 거리
    if(abs(x_dev)<stop_range):
        color_x=(0,255,0) #녹색
    else:
        color_x=(0,0,255) #빨간색
    cv2_im = cv2.putText(cv2_im, str_x, (110, height-8),font, 0.55, color_x, 2)

    y_dev=arr_track_data[3]
    str_y='Y: {}'.format(y_dev) # y축으로 떨어진 거리
    if(abs(y_dev)<stop_range):
        color_y=(0,255,0) #녹색
    else:
        color_y=(0,0,255) #빨간색
    cv2_im = cv2.putText(cv2_im, str_y, (220, height-8),font, 0.55, color_y, 2)

    #상태 표시--------------------------------------------------------------------------
    cmd = arr_track_data[4]
    if(cmd==0):
        str1="Missing Human"
    elif(cmd=='stop'):
        str1='Stop'
    elif(cmd=='forward'):
        str1='Forward'
    elif(cmd=='backward'):
        str1='Backward'
    elif(cmd=='revise'):
        str1='Revise'
    else:
        str1='something wrong'
    cv2_im = cv2.putText(cv2_im, str1, (width-140, 18),font, 0.7, (0, 255, 255), 2)


    #격자-------------------------------------------------------------------------------
    cv2_im = cv2.rectangle(cv2_im, (0,int(height/2)-1), (width, int(height/2)+1), (255,0,0), -1) #가로축
    cv2_im = cv2.rectangle(cv2_im, (int(width/2)-1,0), (int(width/2)+1,height), (255,0,0), -1)  #세로축

    #물체의 정중앙에 빨간점 찍기----------------------------------------------------------
    cv2_im = cv2.circle(cv2_im, (int(arr_track_data[0]*width),int(arr_track_data[1]*height)), 7, (0,0,255), -1)

    #정지범위 사각형---------------------------------------------------------------------
    # x축 정지범위
    cv2_im = cv2.rectangle(cv2_im, (int(width/2-stop_range*width),0), (int(width/2+stop_range*width),height), (0,255,0), 2)
    # y축 정지범위
    cv2_im = cv2.rectangle(cv2_im, (0,int(height-(stop_range/2)*width)), (width,int(height/2+(stop_range*2)*width)), (0,255,0), 2)

    #BBox-------------------------------------------------------------------------------
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        
        box_color, text_color, thickness=(0,150,255), (0,255,0),2
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), box_color, thickness)
        cv2_im = cv2.rectangle(cv2_im, (x0, y1), (x1, y1), (0,0,255), thickness)

    return cv2_im