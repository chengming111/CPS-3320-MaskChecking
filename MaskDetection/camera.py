#检测视频或者摄像头中的人脸
import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import dlib
from PIL import Image
model = load_model('/Users/chengming/MaskDetection/mask_and_unmask_small_1.h5')
detector = dlib.get_frontal_face_detector()
video=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
def rec(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dets=detector(gray,1)
    if dets is not None:
        for face in dets:
            left=face.left()
            top=face.top()
            right=face.right()
            bottom=face.bottom()
            cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
            img1=cv2.resize(img[top:bottom,left:right],dsize=(150,150))
            img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            img1 = np.array(img1)/255.
            img_tensor = img1.reshape(-1,150,150,3)
            prediction =model.predict(img_tensor)    
            if prediction[0][0]>0.5:
                result='unmask'
            else:
                result='mask'
            cv2.putText(img, result, (left,top), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video', img)
while video.isOpened():
    res, img_rd = video.read()
    if not res:
        break
    rec(img_rd)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
