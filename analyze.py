import importlib
import torch
import torch.nn as nn


def loadModel(weightPath):
    model_module = importlib.import_module('model_selecsls')
    net = model_module.Net(nClasses=1000, config='SelecSLS60')
    net.load_state_dict(torch.load(weightPath, map_location=lambda storage, loc: storage))

    if not torch.cuda.is_available():
        print("Warning! Running on CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def analyse(webcam_image):
    pass  # TODO
