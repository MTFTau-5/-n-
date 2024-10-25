import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义超参数
BATCH_SIZE = 16  # 每批处理的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用 cpu 还是 gpu
EPOCHS = 50  # 训练次数

# 构建 pipeline，对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 正则化，降低模型复杂度
])

# 下载数据集
train_set = datasets.MNIST('data', train=True, download=True, transform=pipeline)
test_set = datasets.MNIST('data', train=False, download=False, transform=pipeline)
# 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 构建模型
class Dight(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)  # batch_size   ×1×28×28
        x = self.conv1(x)  # s\输出 batch*1*28*28,输出×10×24×24(28-5+1=24)
        x = F.relu(x)  # 保持 shape 不变
        x = F.max_pool2d(x, 2, 2)  # 输入：10×24×24  输出：batch×10×12×12
        x = self.conv2(x)  # 10*12*12  20*10*10
        x = F.relu(x)
        x = x.view(input_size, -1)  # 拉平，自动计算维度 20×10×10=2000
        x = self.fc1(x)  # 2000 500
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # 数字概率找最大的
        return output

# 定义优化器
model = Dight().to(DEVICE)
optimizer = optim.Adam(model.parameters())

# 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到 GPU
        data = data.to(device)
        target = target.to(device)
        # 梯度初始化为 0
        optimizer.zero_grad()
        # 预测结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)
        # 找到概率
        pred = output.max(1, keepdim=True)
        # 损失函数
        loss.backward()
        # 反向传播
        optimizer.step()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print('Train Epoch : () \t Loss:{:.6f}'.format(epoch, loss.item()))

# 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            # 部署到 device 上
            data = data.to(device)
            target = target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 概率
            pred = output.max(1, keepdim=True)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test - Average loss : (:.4f),Accuracy:{:.3f}\n'.format(
        test_loss, 100.0 * correct / len(test_loader.dataset)))

# 调用，输出
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)
