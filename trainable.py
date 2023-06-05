import torch.nn as nn
import torch.optim as optim
import models


# 定义网络模型
class Net(nn.Module):
    def __init__(self, ...):
        super(Net, self).__init__()


    def forward(self, x):

        return

num_epochs=3
trainloader=3
momentum=0.9

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 定义Adam优化器，并设置超参数
lr = 0.1 # 学习率
beta1 = 0.1 # 一阶矩估计的指数衰减率
beta2 = 0.1 # 二阶矩估计的指数衰减率
eps = 0.01 # 防止除0操作的增量
num_classes=10
feature_dim=10

params=models.parameters()

optimizer = optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=eps)

# 初始化可训练原型向量
prototype = torch.randn(num_classes, feature_dim, requires_grad=True)

# 在训练循环中更新可训练原型
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # 将输入数据传入模型进行前向计算，得到特征向量
        features = models(inputs)

        # 计算损失函数
        loss = criterion(features, labels)

        # 将当前批次输入数据的特征向量平均池化得到一个向量
        pooled_features = features.mean(dim=0)

        # 更新可训练原型
        prototype[labels] = prototype[labels] * momentum + (1 - momentum) * pooled_features

        # 清空梯度并反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新网络参数和可训练原型
        optimizer.step()
        prototype.data = prototype.clamp(min=0, max=1)
