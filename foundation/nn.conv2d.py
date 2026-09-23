import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import  DataLoader

dataset=torchvision.datasets.CIFAR10("/foundation\\dataset",
                                     train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)

    def forward(self,x):
       x = self.conv1(x)
       return x

tudui=Tudui()
print(tudui)

for data in dataloader:
    imgs,target=data
    output=tudui(imgs)
    print(imgs.shape)
    print(output.shape)