import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image #Used to load a sample image
import cv2

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#Load a flower image from sklearn.datasets, crop it to shape 1 X 3 X 256 X 256:
img = cv2.imread(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\Dataset\TrainData\images\00041.png')
print(img.shape)
print('Starting')
I = torch.tensor(img).permute(2,0,1).unsqueeze(0).type(dtype)
print('adsf')
kernel_size = 16
stride = kernel_size//2 
I_unf = F.unfold(I, kernel_size, stride=stride)

print(I_unf.shape)
I_f = F.fold(I_unf,I.shape[-2:],kernel_size,stride=stride)
norm_map = F.fold(F.unfold(torch.ones(I.shape).type(dtype),kernel_size,stride=stride),I.shape[-2:],kernel_size,stride=stride)
I_f /= norm_map
print(I_f.shape)


plt.imshow(I[0,...].permute(1,2,0).cpu()/255)
plt.figure()
plt.imshow(I[0,...].permute(1,2,0).cpu()/255)
plt.show()
