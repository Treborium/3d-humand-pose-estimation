# 3D Human Pose Estimation

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import threading

import ui


def show_3d_coordinate_system():
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = [2, 3]
    x = [2, 2]
    y = [1, 4]
    ax.plot(x, y, z, label='A Leg (?)')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    webcam_thread = threading.Thread(target=ui.createUI)
    webcam_thread.start()
    show_3d_coordinate_system()
