import cv2
import numpy as np;

ESCAPE_KEY = 27


def empty(x):
    pass


def createUI():
    webcam_image = np.zeros((200, 150, 3), np.uint8)
    webcam_image_rgb = np.zeros((200, 150, 3), np.uint8)  # BGR -> RGB for pytorch

    # Create window and UI
    cv2.namedWindow('3D Human Pose Estimation')
    cv2.createTrackbar('Model', '3D Human Pose Estimation', 0, 6, empty)

    cam = cv2.VideoCapture(0)  # Opens the default camera

    while True:
        # Read Webcam data
        _, webcam_image = cam.read()
        #   image = cv2.flip(image, 1) Mirroring needed?

        # Convert to RGB and analyse
        webcam_image_rgb = webcam_image #cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
        # TODO analyse the webcam image

        cv2.circle(webcam_image, center=(50, 50), radius=50, color=(255, 0, 255), thickness=3)

        cv2.imshow('3D Human Pose Estimation', webcam_image)

        if cv2.waitKey(1) == ESCAPE_KEY:
            break  # esc to quit

    cv2.destroyAllWindows()
