import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self,file_path):
        self.images,self.labels = self._read_file(file_path)

    def _read_file(self,file_path):
        images=[]
        labels=[]
        with open(file_path,'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                line=line.rstrip('\n')
                items=line.split(',')
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels

    def __getitem__(self, index):
        image,label = self.images[index],self.labels[index]
        image = torch.tensor(image)
        image = image/ 255.0
        image = (image - 0.1307) / 0.3081
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.images)


batch_size = 64
train_dataset = MNISTDataset("C:\\Users\\1\\Desktop\\learn-torch\\data\\mnist_test.csv\\mnist_test.csv")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset("C:\\Users\\1\\Desktop\\learn-torch\\data\\mnist_train.csv\\mnist_train.csv")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
learning_rate = 0.03
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# 模型、损失函数、优化器
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#训练过程
model.train()
for epoch in range (num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        #计算损失并更新参数
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #计算准确率
        preds = torch.argmax(outputs,dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        #计算并打印本轮结果
    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

#测试过程
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs,dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
