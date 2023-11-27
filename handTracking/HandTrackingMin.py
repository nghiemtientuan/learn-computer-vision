import cv2
import mediapipe as mp
import time

# 0/1 all can use
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        # results.multi_hand_landmarks mang chua nhieu tay neu phat hien thay
        for handLms in results.multi_hand_landmarks:
            # handLms.landmark mang chua 21 diem tren ban tay
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # tinh chieu rong, chieu cao
                # print(id, cx, cy)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255 ), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # tinh va chen fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(
        img,
        str(int(fps)),
        (10, 70), # position
        cv2.FONT_HERSHEY_PLAIN, # font
        3,
        (255, 0, 255), # color
        3 # do day chu
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
