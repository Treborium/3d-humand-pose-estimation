# 3D Human Pose Estimation
import threading

import chart
import ui





if __name__ == '__main__':

    webcamThread = threading.Thread(target=ui.createUI)
    webcamThread.start()
    chart.createChart()
