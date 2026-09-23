
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("/foundation\\dataset",
                                     train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

tudui=Tudui() #创建神经网络实例

writer=SummaryWriter("../logs_maxpool")
step=0

for data in dataloader:
    imgs,target=data
    writer.add_images("input",imgs,step)
    output=tudui(imgs)
    writer.add_images("output", output, step)
    step=step+1

writer.close()