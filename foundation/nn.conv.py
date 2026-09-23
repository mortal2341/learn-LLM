
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("/foundation\\dataset",
                                     train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)



kernel=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])

input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3))

output=F.conv2d(input,kernel,stride=1)
print(output)
