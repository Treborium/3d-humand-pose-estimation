import importlib
import re

import PIL
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

net = None

## Load a model with the given nam
## This method is called from the ui (1) on startup and (2) when a user selects a different model
def loadModel(model):
    global net
    print("Loading model " + model)

    # Get the model type name selecSLS expects
    model_typename_start = model.index("/") + 1
    model_typename_end = model.index(".p")
    model_type = "Selec" + model[model_typename_start:model_typename_end]
    model_type = model_type.replace("_old", "")
    print("Loading type '" + model_type + "'")

    model_module = importlib.import_module('model_selecsls')
    net = model_module.Net(nClasses=1000, config=model_type)  # Type of kind 'SelecSLS60'
    net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
    net.eval()
    cuda_active = torch.cuda.is_available
    if not cuda_active:
        print("Warning: Running on CPU")
    else:
        print("Starting CUDA device")

    device = torch.device("cuda:0" if cuda_active else "cpu")  # Use CUDA on nVIDIA GPUs, else CPU
    print()


## Analyse the webcam image
def analyse(webcam_image):
    webcam_image_rgb = webcam_image  # cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)



    pil_image = PIL.Image.fromarray(
        webcam_image)  # convert from ndarray to pil, see https://discuss.pytorch.org/t/use-cv2-to-load-images-in-custom-dataset/67196

    # Use transformations
    # Resize and crop the image
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm_transform
    ])

    tensor_image = transform(pil_image).float().unsqueeze_(0)

    #  We dont use multiple images
    dataset = WebcamDataset(pil_image, 0, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    output = net.forward(tensor_image)
    print(str(output))
    print(str(output.shape))

    #index = output.data.numpy().argmax()
    #print(str(index))


class WebcamDataset(torch.utils.data.Dataset):
    def __init__(self, pil_image, images_class: int, transform):
        self.pil_image = pil_image
        self.transform = transform
        self.images_class: int = images_class

    def __getitem__(self, index):
        self.transform( self.pil_image, self.images_class) # Transform
