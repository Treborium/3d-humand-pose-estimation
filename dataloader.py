import importlib
import re

import PIL
import torch
import torchvision.transforms as transforms

## Load a model with the given nam
## This method is called from the ui (1) on startup and (2) when a user selects a different model
def loadModel(model):
    print("Loading model " + model)

    typeStart = model.index("/")+1
    typeEnd = model.index(".p")
    type = "Selec" + model[typeStart:typeEnd]
    print("Loading type '" + type + "'")

    model_module = importlib.import_module('model_selecsls')
    net = model_module.Net(nClasses=1000, config=type)  # Type of kind 'SelecSLS60'
    net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    cuda_active = torch.cuda.is_available
    if not cuda_active:
        print("Warning: Running on CPU")
    else:
        print("Starting CUDA device")

    device = torch.device("cuda:0" if cuda_active else "cpu")  # Use CUDA on nVIDIA GPUs, else CPU

## Analyse the webcam image
def analyse(webcam_image):
    pil_image = PIL.Image.fromarray(webcam_image)  # convert from ndarray to pil, see https://discuss.pytorch.org/t/use-cv2-to-load-images-in-custom-dataset/67196

    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm_transform
    ])

#TODO Load dataset and create dataloader

    pass  # TODO Implement
