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


## Usage

Simply execute `main.py` with python:

```
python main.py
```

## Troubleshooting

> UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.

If you encounter the error above, you need to install some additional dependencies on your machine. Please see:

- [StackOverflow Thread](https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so)
- [Arch Linux Forum Thread](https://bbs.archlinux.org/viewtopic.php?pid=1885317#p1885317)
