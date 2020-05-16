import os


import cv2
import numpy as np
import time
import _datetime as datetime

import dataloader

ESCAPE_KEY = 27
WINDOW_TITLE = '3D Human Pose Estimation'
MODEL_FOLDER = 'models/'
MODEL_COUNT = 0
MODELS = []


def empty(x):
    pass


# Returns the difference to time s and maybe shows the FPS
def timeDiff(t, showFPS = False):
    s = time.perf_counter() - t  # difference time in seconds
    fps = round(1 / s)
    if not showFPS:
        return str(round(s * 1000)) + "ms"
    return str(round(s * 1000)) + "ms, " + str(fps) + "fps"


# Draws the time difference to t into img with the given name
def drawCalcTime(img, t, name, line, showFPS=False):
    cv2.putText(img, name + ": " + timeDiff(t, showFPS), (10, 20 + 20 * line), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (255, 255, 255), 2)


def getModels():
    global MODELS
    if len(MODELS) != 0:
        return MODELS

    files = []
    for file in os.listdir(MODEL_FOLDER):
        if file.endswith('.pth'):
            files.append(file)

    MODELS = files

    if len(MODELS) == 0:
        print("Error: No models in folder ./models or folder does not exist")
        print("Please download the models with 'sh downloadModels.sh'")
        raise Exception("No models found :(")

    return MODELS


def createUI():
    global MODEL_COUNT
    usedModel = 0
    time_part = time.perf_counter()
    time_all = time.perf_counter()

    print("Starting UI... Press 'Esc' to exit")
    webcam_image = np.zeros((200, 150, 3), np.uint8)
    webcam_image_rgb = np.zeros((200, 150, 3), np.uint8)  # BGR -> RGB for pytorch

    MODEL_COUNT = len(getModels())
    print("Found " + str(MODEL_COUNT) + " models to use")
    dataloader.loadModel(MODEL_FOLDER + getModels()[usedModel])

    # Create window and UI
    cv2.namedWindow(WINDOW_TITLE)
    cv2.createTrackbar('Model', WINDOW_TITLE, 0, MODEL_COUNT - 1, empty)
    cv2.createTrackbar('Screenshot', WINDOW_TITLE, 0, 1, empty)

    print("Connecting to Webcam (this may take a few seconds...)")
    cam = cv2.VideoCapture(0)  # Opens the default camera

    print("Running")

    while True:
        # Read Webcam data
        time_part = time.perf_counter()
        _, webcam_image = cam.read()
        webcam_image = cv2.flip(webcam_image, 1)  # Mirror
        drawCalcTime(webcam_image, time_part, "Webcam", 1)
        time_part = time.perf_counter()

        # Convert to RGB and analyse
        # TODO analyse the webcam image
        dataloader.analyse(webcam_image)

        drawCalcTime(webcam_image, time_part, "Calc", 2)

        # TODO draw skeleton
        cv2.circle(webcam_image, center=(320, 240), radius=50, color=(255, 0, 255), thickness=1)

        cv2.putText(webcam_image, "Model: " + getModels()[usedModel], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .6,
                    (192, 192, 192), 2)

        drawCalcTime(webcam_image, time_all, "All", 3, True)
        time_all = time.perf_counter()

        # Draw to screen

        cv2.imshow(WINDOW_TITLE, webcam_image)

        # Model Change Event
        model = cv2.getTrackbarPos('Model', WINDOW_TITLE)
        if not usedModel == model:
            file = MODEL_FOLDER + getModels()[model]
            dataloader.loadModel(file)
            usedModel = model

        # Screenshot event
        # FIXME This event gets called twice but the position is reset inside the block
        if cv2.getTrackbarPos('Screenshot', WINDOW_TITLE) == 1:
            if not os.path.exists('output'):  # Create folder
                os.makedirs('output')
            dt = datetime.datetime.today().strftime('%Y%m%d-%H.%M.%S')
            cv2.imwrite('output/img_' + dt + '.png', webcam_image)  # Save to folder
            cv2.setTrackbarPos('Screenshot', WINDOW_TITLE, 0)  # Reset Trackbar
            print("Screenshot saved")
            time.sleep(1)  # User can preview the saved frame

        # Exit
        if cv2.waitKey(1) == ESCAPE_KEY:
            break  # esc to quit

    cv2.destroyAllWindows()
