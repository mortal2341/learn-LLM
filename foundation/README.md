# foundation · PyTorch 底层复现

**这个阶段解决的问题**：`nn.Conv2d(...)` 一行代码背后发生了什么？在用 PyTorch 之前，先把最核心的抽象亲手写一遍。

## 内容清单

| 文件 | 实现内容 |
|---|---|
| `nn,module.py` | 手写 `nn.Module` 抽象：参数注册、前向传播契约 |
| `nn,seq.py` | 手写序列容器（对应 `nn.Sequential`） |
| `nn.conv.py` / `nn.conv2d.py` | 手写二维卷积：滑窗计算与通道间聚合 |
| `nn,maxpool.py` | 手写最大池化的下采样逻辑 |
| `nn.relu.py` | 手写 ReLU 激活 |
| `read_data.py` | 数据读取与预处理流水线 |
| `dataset/cifar-10-batches-py/` | CIFAR-10 测试数据 |

## 学习方式

不继承框架的 `nn.Module`，而是从零实现参数管理与前向计算的绑定关系，确保理解：

- `Module` 为什么能自动追踪参数（参数注册机制）
- 卷积核如何在输入上滑动并聚合通道
- 池化如何完成空间下采样

后续所有 Step 的手写实现（注意力、LoRA、MoE 路由）都建立在这一层的理解之上。
