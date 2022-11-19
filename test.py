import torch
import random
import math
from torch.autograd import Variable as V
from torchvision import transforms
import os
import numpy as np
import cv2
from time import time
from networks.UNet import UNet
from framework import MyFrame
from loss import dice_bce_loss
from data_preprocessing import ImageFolder
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

test_transformer=transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor()
])
# img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
# img = V(torch.Tensor(img).to(self.device))




test_set=cv2.imread("training/images/satImage_099.png")
test = cv2.resize(test_set, (384,384))
img = np.array(test, np.float32)/255.0 * 3.2 - 1.6
# test_set=test_transformer(test_set)
# test_set=test_set.unsqueeze(0)
# test_set=test_set.to('cuda')
img = V(torch.Tensor(img).to("cuda"))
img =img.permute(2,0,1).unsqueeze(0)
print(img.shape)

# #
# #
model = UNet()
model.load_state_dict(torch.load('weights/UNet.pth'))
# print(model)
model.to('cuda')
model.eval()
# # print(model)
pred=model(img)
pred=torch.squeeze(pred)
pred=pred.cpu()
pred=torch.argmax(pred.permute(1,2,0), axis=-1).detach().numpy()
plt.imshow(pred)
plt.show()
print(pred.shape)

# #
# pred=pred.cpu()
# pred=pred.permute(1,2,0).numpy()
#
# pred=pred*255.0
# pred=np.array(pred)
# pred=np.sum(pred)
#
# # cv2.imwrite('test.jpg',pred)
# print(pred)