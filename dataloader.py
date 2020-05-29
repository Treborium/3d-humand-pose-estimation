import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.inference_engine_trt import InferenceEngineTRT
from modules.parse_poses import parse_poses

net = None


## Load a model with the given nam
## This method is called from the ui (1) on startup and (2) when a user selects a different model
def loadModel(model):
    global net
    print("Loading model " + model)

    cuda_active = torch.cuda.is_available
    if not cuda_active:
        print("Warning: Running on CPU")
    else:
        print("CUDA is available")

    if model.endswith("_opt.pth"):
        print("Loading optimized RTR model")
        net = InferenceEngineTRT(model, "GPU" if cuda_active else "CPU")
    else:
        net = InferenceEnginePyTorch(model, "GPU" if cuda_active else "CPU")


def prepareImage(webcam_image, height, fx):
    input_scale = height / webcam_image.shape[0]
    if fx < 0:  # Focal length is unknown
        fx = np.float32(0.8 * webcam_image.shape[1])
    image = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)  # Set TO RGB
    image = cv2.resize(image, dsize=None, fx=input_scale, fy=input_scale)  # Resize
    return image, input_scale, fx


## Analyse the webcam image
def calcPoses(image, input_scale, fx):
    stride = 8
    inference_result = net.infer(image)
    poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, True)

    return poses_3d, poses_2d


class WebcamDataset(torch.utils.data.Dataset):
    def __init__(self, pil_image, images_class: int, transform):
        self.pil_image = pil_image
        self.transform = transform
        self.images_class: int = images_class

    def __getitem__(self, index):
        self.transform(self.pil_image, self.images_class)  # Transform
