import cv2
import time
from collections import deque
from win10toast import ToastNotifier

# 加载OpenCV的人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
#在这里也可以使用人体检测方法

# 打开摄像头
cap = cv2.VideoCapture(1)


# 初始化时间和缓冲区
start_time = time.time()
buffer = deque(maxlen=4)  # 存储前4帧的人脸检测结果，防止检测错误
toaster = ToastNotifier()  # windows消息提醒

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 在灰度图像上检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 绘制人脸边界框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 更新缓冲区
    buffer.append(len(faces) > 0)

    # 如果检测到人且连续检测到人脸且停留时间超过0.3秒，执行相应操作
    if sum(buffer) >= len(buffer) and (time.time() - start_time) > 0.3:
        print("Detected person for more than 1 second!")
        # windows弹窗提醒
        toaster.show_toast("提醒", "有人出现！", duration=5)

        # 重置计时器
        start_time = time.time()

    # 显示帧
    cv2.imshow('frame', frame)

    # q退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
