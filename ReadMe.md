# 3D Human Pose Estimation

This project is based on the following [repository](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch).

## Requirements

- Python >=3.5
- CMake >=3.1
- C++ Compiler
- OpenCV >=4.0

## Setup

1. Create a virtual environment and activate it

```
python -m venv venv
source venv/bin/activate
```

2. Make sure to install the required python packages 

```
pip install -r requirements.txt
```

3. Build the `pose_extractor` module

```
python setup.py build_ext
```

4. Add build folder to the `PYTHONPATH`

```
export PYTHONPATH=pose_extractor/build/:$PYTHONPATH
```

### Setup on the Jetson Nano

The steps are theoretically the same as above, however you **will** have issues installing the required dependencies.
Here is a list with some useful links and tips: 

- Sometimes dependencies need to be installed in a certain order
- [Installing PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-5-0-now-available/72048)
- `Matplotlib` requires *FreeType2* to be installed on the system: `sudo apt install libfreetype6-dev`
- `OpenCV` and `Torchvision` can't be installed directly inside a virtual environment.
  Install both globally on the system via `pip3 install --user <python_package>` and then copy them from the system dependency directory to the virtual environment dependency directory. Something like this:

  ```
  cp -r /usr/local/lib/python3.6/dist-packages/torchvision ~/3d-human-pose-estimation/venv/lib/python3.6/site-packages/
  ```


## Usage

Simply execute `main.py` with python:

```
python main.py
```

You can also use a video file instead of the webcam. Simply pass the path to the file as an argument: 

```
python main.py /path/to/video-file/
```

## UI Controls
#### Model
Selects the model from the 'model' folder (*.pict). The model name is drawn onscreen.
#### Height
Sets the image width and height. Lower resolutions will increase performance but decrease the quality of the
 recognision. 256 seems to be a good starting value for most use-cases
#### FX
Sets the camera lens focual length. If unsure (you likely are), "0" is automatic match.
#### Screenshot
Takes a screenshot in the output folder. Acts like a slider component
#### Sync Draw
Draws the 3d poses aynchronous (=0) or synchronous (=1). Asynchron draw could lead to a better performance
#### Frame Buffer
 Determines how many frames the network will process concurrently. A value of "0" means that no frame buffer is used.
 A higer frame buffer value leads to a higher performance but increases lag and jitter. Values around 3-6 are profen
  to be a good tradeoff and adds around 50-70% better performance.
  
### On Screen Drawing
#### Webcam
The time to get the image from the webcam (or video/image) file.

#### CV2
The time CV2 uses to tranform the image to a usable format
#### POSE
The pose estimation
#### DRAW
On-Screen Drawing
#### ALL
The accumulated stats for this single(!) frame. If using a buffer this value could be quite high because the GPU
 calculates multiple images concurrently
#### Buffer AVG
The draw time and FPS, averaged on the last 10 frames. This is the value you want to check how good the network is
 performing.

## Troubleshooting

> UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.

If you encounter the error above, you need to install some additional dependencies on your machine. Please see:

- [StackOverflow Thread](https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so)
- [Arch Linux Forum Thread](https://bbs.archlinux.org/viewtopic.php?pid=1885317#p1885317)
