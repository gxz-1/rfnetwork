import torch.nn as nn


# 添加1D残差块
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()

        # 主路径
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 捷径连接（shortcut connection）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)  # 残差连接
        out = self.relu(out)

        return out


class EmbeddingNet_res(nn.Module):
    """
    功能：基础特征提取网络，将输入图像映射到特征空间
    结构：
        卷积部分：两层卷积+激活+池化
        全连接部分：三层全连接网络
        输出：2维特征向量
    应用：作为其他网络的基础组件，独立使用可直接进行特征提取
    """

    def __init__(self, in_channels=2, embedding_dim=128):
        """
        参数：
        - in_channels: 输入通道数（实部和虚部，默认为2）
        - embedding_dim: 嵌入向量维度
        """
        super(EmbeddingNet_res, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # 残差块1
        self.res_block1 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=1), ResidualBlock1D(64, 64)
        )

        # 残差块2
        self.res_block2 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2), ResidualBlock1D(128, 128)
        )

        # 残差块3
        self.res_block3 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2), ResidualBlock1D(256, 256)
        )

        # 自适应池化，确保输出固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNet_cov(nn.Module):
    """
    功能：基础特征提取网络，将输入图像映射到特征空间
    结构：
        卷积部分：两层卷积+激活+池化
        全连接部分：三层全连接网络
        输出：2维特征向量
    应用：作为其他网络的基础组件，独立使用可直接进行特征提取
    """

    def __init__(self, in_channels=2, embedding_dim=128):
        """
        参数：
        - in_channels: 输入通道数（实部和虚部，默认为2）
        - embedding_dim: 嵌入向量维度
        """
        super(EmbeddingNet_cov, self).__init__()

        # 卷积块1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # 卷积块4
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # 自适应平均池化，确保输出固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


# 未使用的类 - 添加L2归一化的嵌入网络
# class EmbeddingNetL2(EmbeddingNet_cov):
#     """
#     功能：在EmbeddingNet基础上添加L2归一化
#     特点：输出的特征向量被归一化为单位长度
#     """
#     def __init__(self):
#         super(EmbeddingNetL2, self).__init__()
#
#     def forward(self, x):
#         output = super(EmbeddingNetL2, self).forward(x)
#         output /= output.pow(2).sum(1, keepdim=True).sqrt()
#         return output
#
#     def get_embedding(self, x):
#         return self.forward(x)


# 未使用的类 - 分类网络
# class ClassificationNet(nn.Module):
#     """
#     功能：在嵌入网络基础上添加分类层
#     结构：
#         使用任意嵌入网络提取特征
#         非线性激活后连接全连接分类层
#     输出：类别概率分布(对数softmax)
#     """
#     def __init__(self, embedding_net, n_classes):
#         super(ClassificationNet, self).__init__()
#         self.embedding_net = embedding_net
#         self.n_classes = n_classes
#         self.nonlinear = nn.PReLU()
#         self.fc1 = nn.Linear(2, n_classes)
#
#     def forward(self, x):
#         output = self.embedding_net(x)
#         output = self.nonlinear(output)
#         scores = F.log_softmax(self.fc1(output), dim=-1)
#         return scores
#
#     def get_embedding(self, x):
#         return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    """
    功能：孪生网络，处理成对样本
    特点：
        使用相同的嵌入网络处理两个输入
        权重共享确保相似输入映射到相似特征
    """

    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        """
        获取单个输入的嵌入向量
        """
        return self.embedding_net(x)


class TripletNet(nn.Module):
    """
    功能：三元组网络，处理三元组样本
    输入：锚点样本(anchor)、正样本(positive)、负样本(negative)
    特点：
        使用相同的嵌入网络处理三个输入
        权重共享确保相似输入映射到相似特征
    """

    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
