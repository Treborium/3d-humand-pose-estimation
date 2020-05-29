import colorsys
import os
import sys

import cv2
import numpy as np
import time
import _datetime as datetime

import chart
import dataloader

from modules.input_reader import VideoReader

ESCAPE_KEY = 27
WINDOW_TITLE = '3D Human Pose Estimation'
MODEL_FOLDER = 'models/'
MODEL_COUNT = 0
MODELS = []

# BODY EDGES FROM LHPE3D DEMO
body_edges = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],  # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 6], [6, 7], [7, 8],  # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle


def empty(x):
    pass


# Returns the difference to time s and maybe shows the FPS
def timeDiff(t, showFPS=False):
    s = time.perf_counter() - t  # difference time in seconds
    fps = round(1 / s)
    if not showFPS:
        return str(round(s * 1000)) + "ms"
    return str(round(s * 1000)) + "ms, " + str(fps) + "fps"


# Draws the time difference to t into img with the given name
def drawCalcTime(img, t, name, line, showFPS=False):
    cv2.putText(img, name + ": " + timeDiff(t, showFPS), (10, 20 + 20 * line), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (15, 196, 241), 2)


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
    cv2.createTrackbar('Model', WINDOW_TITLE, usedModel, max(MODEL_COUNT - 1, 1), empty)
    cv2.createTrackbar('Height', WINDOW_TITLE, 256, 512, empty)
    cv2.createTrackbar('FX', WINDOW_TITLE, 0, 50, empty)
    cv2.createTrackbar('Screenshot', WINDOW_TITLE, 0, 1, empty)
    cv2.createTrackbar('Sync Draw', WINDOW_TITLE, 0, 1, empty)

    is_using_video_file = False
    if len(sys.argv) > 1:
        print("Using the provided video file:", sys.argv[1])
        video_reader = VideoReader(sys.argv[1])
        is_using_video_file = True
        video_iter = iter(video_reader)
    else:
        print("Connecting to Webcam (this may take a few seconds...)")
        cam = cv2.VideoCapture(0)  # Opens the default camera

    print("Running")

    while True:
        # Get image data
        time_part = time.perf_counter()

        if is_using_video_file:
            try:
                webcam_image = next(video_iter)
            except StopIteration as e:
                video_iter = iter(video_reader)
                webcam_image = next(video_iter)
        else:
            _, webcam_image = cam.read()

        webcam_image = cv2.flip(webcam_image, 1)  # Mirror
        drawCalcTime(webcam_image, time_part, "WEBCAM", 1)
        time_part = time.perf_counter()

        # Read variables
        height = cv2.getTrackbarPos('Height', WINDOW_TITLE)
        if height < 16:
            height = 16
            cv2.setTrackbarPos('Height', WINDOW_TITLE, 16)
        fx = cv2.getTrackbarPos('FX', WINDOW_TITLE)
        if fx == 0:
            fx = -1

        # Prepare Image
        image, input_scale, fx = dataloader.prepareImage(webcam_image, height, fx)
        drawCalcTime(webcam_image, time_part, "CV2", 2)
        time_part = time.perf_counter()

        # Get Poses
        pose_3d, pose_2d = dataloader.calcPoses(image, input_scale, fx)
        drawCalcTime(webcam_image, time_part, "POSE", 3)
        time_part = time.perf_counter()

        # Draw Poses
        for pid in range(len(pose_2d)):
            # all array elements
            # into ?x3 array
            # reverse dimensions
            pose = np.array(pose_2d[pid][0:-1]) \
                .reshape((-1, 3)) \
                .transpose()
            has_pose = pose[2, :] > 0
            for eid in range(len(body_edges)):  # Go through all defined edges
                edge = body_edges[eid]
                if has_pose[edge[0]] and has_pose[edge[1]]:  # If we have both "points" -> Draw line
                    color = colorsys.hsv_to_rgb(eid / 17.0, 1, 1)  # Use HSL color space to use different colors
                    color = [e * 256 for e in color]  # convert [0,1] to [0,256] for ocv
                    cv2.line(webcam_image, tuple(pose[0:2, edge[0]].astype(int)), tuple(pose[0:2, edge[1]].astype(int)),
                             color, 4, cv2.LINE_AA)

        sync = cv2.getTrackbarPos('Sync Draw', WINDOW_TITLE)
        chart.updateData(pose_3d,sync)

        cv2.putText(webcam_image, "Model: " + getModels()[usedModel], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .6,
                    (192, 192, 192), 2)

        drawCalcTime(webcam_image, time_part, "DRAW", 4)
        time_part = time.perf_counter()
        drawCalcTime(webcam_image, time_all, "All", 5, True)
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
            exit(0)
            break  # esc to quit

    cv2.destroyAllWindows()
