import cv2
import mediapipe as mp
import time


class poseDetector:
    def __init__(self, mode=False, complexityMode=1, smoothLandmarks=True, enableSegmentation=False,
                 smoothSegmentation=True, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.complexityMode = complexityMode
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexityMode, self.smoothLandmarks,
                                     self.enableSegmentation, self.smoothSegmentation, self.detectionConfidence,
                                     self.trackingConfidence)

    def findPose(self, img, drawConnection=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and drawConnection:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, drawConnection=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if drawConnection:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture('videos/1.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)

        # fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (100, 150), cv2.FONT_HERSHEY_PLAIN, 20, (255, 0, 0), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
