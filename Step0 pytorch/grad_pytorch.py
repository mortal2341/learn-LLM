import torch

# 1. 数据准备：转换为 PyTorch 张量
# 注意：PyTorch通常要求数据是浮点型(float)
X_data = [[10, 3], [20, 3], [25, 3], [28, 2.5], [30, 2], [35, 2.5], [40, 2.5]]
y_data = [60, 85, 100, 120, 140, 145, 163]

# 将列表转换为 Tensor
X = torch.tensor(X_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1) # 变成列向量方便计算

# 2. 初始化参数：关键点是设置 requires_grad=True
# 这告诉 PyTorch：“请帮我追踪这个变量的计算过程，我要算它的梯度”
w = torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32, requires_grad=True)

# 为了包含截距 w0，我们在 X 左边加一列 1 (Bias Trick)
# 这样 X 变成了 [[1, 10, 3], [1, 20, 3]...]，w 还是 [w0, w1, w2]
ones = torch.ones(X.shape[0], 1)
X_with_bias = torch.cat((ones, X), dim=1)

lr = 0.0001
num_iterations = 10000

# 3. 训练循环
for i in range(num_iterations):
    # --- 前向传播 (和原来逻辑一样，只是写法不同) ---
    # 矩阵乘法：X(7x3) * w(3x1) = y_pred(7x1)
    y_pred = torch.mm(X_with_bias, w)

    # 计算 Loss (均方误差)
    # PyTorch 里可以直接写公式，或者用内置函数
    loss = ((y_pred - y) ** 2).mean()

    # --- 反向传播 (替代你手写的梯度公式) ---
    # 如果 loss 已经有值了，先清空上一步的梯度，否则会累加
    if w.grad is not None:
        w.grad.zero_()

    # 自动求导！这一行代码自动完成了你原来第15-17行的工作
    loss.backward()

    # --- 参数更新 ---
    # 注意：更新参数时，我们不想让 PyTorch 追踪这个“减法”操作，
    # 所以要用 torch.no_grad() 或者直接操作 .data
    with torch.no_grad():
        w -= lr * w.grad

    # 打印
    if i % 1000 == 0:
        print(f"Iteration {i}: Loss = {loss.item():.4f}")

# 4. 输出结果
# .data 取出数值，不需要梯度信息
print(f"Final parameters: w0 = {w[0].item():.2f}, w1 = {w[1].item():.2f}, w2 = {w[2].item():.2f}")