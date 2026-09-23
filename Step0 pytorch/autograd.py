import torch
from torch.utils.tensorboard import SummaryWriter

#确保CUDA可用
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#生成数据
inputs = torch.rand(100,3)
weights = torch.tensor([[1.1],[2.2],[3.3]])
bias=torch.tensor(4.4)
targets = inputs@weights + bias +0.1*torch.randn(100,1)

#创建SummaryWriter实例
writer = SummaryWriter(log_dir="C:\\Users\\1\\Desktop\\learn-torch\\autograd")

#初始化参数，并且使用GPU加速
w=torch.rand((3,1),requires_grad=True,device=device)
b=torch.rand((1,),requires_grad=True,device=device)
#将生成的数据也转移到gpu上
inputs = inputs.to(device)
targets = targets.to(device)
#设置超参数
epoch=10000
lr=0.003

#进入训练循环
for i in range(epoch):
    outputs = inputs@w + b
    loss =  torch.mean(torch.square(outputs - targets))
    print("loss:",loss.item())
    # 记录loss，三个参数分别：tag，loss值，第几步
    writer.add_scalar("loss/train", loss.item(), i)

    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    w.grad.zero_()
    b.grad.zero_()

print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)


