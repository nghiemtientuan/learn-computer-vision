import cv2
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
mpDaw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

cap = cv2.VideoCapture('videos/1.mp4')
pTime = 0

while True:
    success, img = cap.read()
    # Anh truoc khi cho vao xu ly tim khuon mat can chuyen sang mau RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)

    # fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
