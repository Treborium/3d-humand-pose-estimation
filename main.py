# pip install opencv-python matplotlib


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading

import ui

ESCAPE_KEY = 27

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)  # Opens the default camera
    print("Press Escape to quit the webcam window")

    while True:
        _, image = cam.read()  # fetch next frame
        if mirror: 
            image = cv2.flip(image, 1)

        cv2.circle(image, center=(50, 50), radius=50, color=(255, 0, 255), thickness=3)

        cv2.imshow('Webcam Stream', image)
        
        if cv2.waitKey(1) == ESCAPE_KEY: 
            break  # esc to quit
        
    cv2.destroyAllWindows()


def show_3d_coordinate_system():
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = [2,3]
    x = [2,2]
    y = [1,4]
    ax.plot(x, y, z, label='A Leg (?)')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    webcamThread = threading.Thread(target=ui.createUI)
    webcamThread.start()
    show_3d_coordinate_system()
