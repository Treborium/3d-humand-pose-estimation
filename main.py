# pip install opencv-python matplotlib


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    print("Press Escape to quit the webcam window")
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)

        cv2.imshow('Camera Data', img)
        # Funktioniert noch nicht
        cv2.circle(img, center=(50, 50), radius=50, color=(255, 0, 255), thickness=5)
        # cv2.line(img=img, pt1=(10, 10), pt2=(300, 10), color=(255, 0, 0), thickness=50)
        
        if cv2.waitKey(1) == 27: 
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
    webcamThread = threading.Thread(target=show_webcam)
    webcamThread.start()
    show_3d_coordinate_system()
