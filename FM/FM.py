import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义 FM 模型
class FM(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(FM, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # 线性部分
        self.linear = nn.Linear(num_features, 1)
        
        # 嵌入层，为每个特征分配embedding_dim长度的emb
        self.embedding = nn.Embedding(num_features, embedding_dim)
        
    def forward(self, x):
        # 线性部分
        linear_part = self.linear(x)
        
        # 计算交叉项
        # 1. 将特征值 x 与嵌入向量相乘
        x_embeds = x.unsqueeze(-1) * self.embedding.weight  # 形状: (batch_size, num_features, embedding_dim)
        # 2. 计算交叉项
        # cross_term = 1/2((∑ViXi)^2-∑(ViXi^2))
        sum_square = torch.sum(x_embeds, dim=1) ** 2  # (batch_size, embedding_dim)
        square_sum = torch.sum(x_embeds ** 2, dim=1)   # (batch_size, embedding_dim)
        cross_term = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)
        
        # FM模型的输出
        output = linear_part + cross_term
        
        return output

# 生成模拟数据
num_samples = 40  # 样本数量
num_features = 10   # 特征数量
emb_nums = 32

X = torch.randn(num_samples, num_features)  # 特征数据
y = torch.randint(0, 2, (num_samples, 1)).float()  # 标签数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 初始化模型
model = FM(num_features, emb_nums)
model = model.to(device)
# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失（带 logits）
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

# 训练参数
num_epochs = 5

# 训练循环
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)  # 计算损失 -1/n∑[y * log(σ(x)) + (1 - y) * log(1 - σ(x))]
        # 反向传播和优化
        optimizer.zero_grad()  # 梯度清0
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
    # 打印每个 epoch 的损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")