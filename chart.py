import threading

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import data
from data import R

fig = None
axis = None


# Creates an chart
def createChart():
    print("Initlialising plot")
    global fig, axis

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    axis = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = [2, 3]
    x = [2, 2]
    y = [1, 4]
    axis.plot(x, y, z, label='A Leg (?)')
    axis.legend()
    plt.show()


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


# Updates data in the chart
def __updateData(pose_3d):
    global axis

    axis.clear()  # Clear all data

    if not len(pose_3d):  # if we have data
        return
    pose_3d = rotate_poses(pose_3d, data.R, data.t)
    pose_3d_copy = pose_3d.copy()
    x = pose_3d_copy[:, 0::4]
    y = pose_3d_copy[:, 1::4]
    z = pose_3d_copy[:, 2::4]
    pose_3d[:, 0::4], pose_3d[:, 1::4], pose_3d[:, 2::4] = -z, x, -y
    pose_3d = pose_3d.reshape(pose_3d.shape[0], 19, -1)[:, :, 0:3]

    for sid in range(len(data.SKELETON_EDGES)):
        firstNr = data.SKELETON_EDGES[sid][0]
        secondNr = data.SKELETON_EDGES[sid][1]

        for human in range(len(pose_3d)):  # Go through all humans
            first = pose_3d[human][firstNr]
            second = pose_3d[human][secondNr]

            x = [first[0], second[0]]
            y = [first[1], second[1]]
            z = [first[2], second[2]]
            axis.plot(x, y, z, label=str(human) + "-" + str(firstNr) + "." + str(secondNr))
    plt.draw()

def updateData(pose_3d, sync = False):
    if sync:
        __updateData(pose_3d)
    else:
        pose3d_copy = pose_3d.copy()
        t = threading.Thread(target=__updateData, args=[pose3d_copy])
        t.start()

