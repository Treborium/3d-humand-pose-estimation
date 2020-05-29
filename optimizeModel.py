# Optimizes a model
# Relies on the platform => Must be compiled for every device
# The first command lin parameter must be the path to the old model
import sys

import torch
# import torch2trt

from models.with_mobilenet import PoseEstimationWithMobileNet
from torch2trt.torch2trt import *


MODEL_NAME = sys.argv[1]
NEW_MODEL_NAME = MODEL_NAME.replace(".pth", "_opt.pth")
WIDTH = 224
HEIGHT = 224

net = PoseEstimationWithMobileNet()
net.load_state_dict(torch.load(sys.argv[1]), strict=False)
net.eval().cuda()

data = torch.ones((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = torch2trt(net, [data], fp16_mode=True, max_workspace_size=1 << 25)
torch.save(model_trt.state_dict(), NEW_MODEL_NAME)
