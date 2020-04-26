import cv2
import numpy as np
import time

ESCAPE_KEY = 27


def empty(x):
    pass

def timeDiff(t,showFPS):
    ms = time.perf_counter() - t
    fps = round(1000/ms)
    if not showFPS:
        return str(round(ms)) + "ms"
    return str(round(ms)) + "ms," + str(fps) + "fps"

def drawCalcTime(img, t,name, line, showFPS=False):
    cv2.putText(img, name + ": " +timeDiff(t,  showFPS),(10,20+20*line),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)

def createUI():
    print("Starting UI...")
    webcam_image = np.zeros((200, 150, 3), np.uint8)
    webcam_image_rgb = np.zeros((200, 150, 3), np.uint8)  # BGR -> RGB for pytorch

    # Create window and UI
    cv2.namedWindow('3D Human Pose Estimation')
    cv2.createTrackbar('Model', '3D Human Pose Estimation', 0, 6, empty)

    print("Connecting to Webcam")
    cam = cv2.VideoCapture(0)  # Opens the default camera

    time_part = time.perf_counter()
    time_all = time.perf_counter()

    while True:
        # Read Webcam data
        time_part = time.perf_counter()
        _, webcam_image = cam.read()
        webcam_image = cv2.flip(webcam_image, 1) #Mirror
        drawCalcTime(webcam_image,time_part,"Webcam",1)
        time_part = time.perf_counter()



        # Convert to RGB and analyse
        webcam_image_rgb = webcam_image #cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
        # TODO analyse the webcam image
        drawCalcTime(webcam_image,time_part,"Calc",2)

        cv2.circle(webcam_image, center=(50, 50), radius=50, color=(255, 0, 255), thickness=3)
        drawCalcTime(webcam_image,time_all,"All", 3,True)
        time_all = time.perf_counter()
        # Draw
        cv2.imshow('3D Human Pose Estimation', webcam_image)

        if cv2.waitKey(1) == ESCAPE_KEY:
            break  # esc to quit

    cv2.destroyAllWindows()
