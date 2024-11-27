import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.relu1 = nn.ReLU()

    def __call__(self, x):
        return self.relu1(self.bn1(self.conv1(x)))


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=num_filters)
        self.relu2 = nn.ReLU()

    def __call__(self, x):
        residual = x
        temp = self.relu1(self.bn1(self.conv1(x)))
        output = self.relu2(self.bn2(self.conv2(temp)) + residual)
        return output


class ValueHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(64, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.tanh1 = nn.Tanh()

    def __call__(self, x):
        temp1 = self.relu1(self.bn1(self.conv1(x)))
        view = temp1.view(temp1.shape[0], 64)
        temp2 = self.tanh1(self.fc2(self.relu2(self.fc1(view))))
        return temp2


class PolicyHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=2)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(128, 4608)

    def __call__(self, x):
        temp = self.relu1(self.bn1(self.conv1(x)))
        view = temp.view(temp.shape[0], 128)
        temp = self.fc1(view)
        return temp


class AlphaZeroNet(nn.Module):
    def __init__(self, num_blocks, num_filters):
        super().__init__()
        self.convBlock1 = ConvBlock(16, num_filters)
        residual_blocks = []
        for i in range(num_blocks):
            residual_blocks.append(ResidualBlock(num_filters))
        self.residualBlocks = nn.ModuleList(residual_blocks)
        self.valueHead = ValueHead(num_filters)
        self.policyHead = PolicyHead(num_filters)
        self.softmax1 = nn.Softmax(dim=1)
        self.mseLoss = nn.MSELoss()
        self.crossEntropyLoss = nn.CrossEntropyLoss()

    def __call__(self, x, value_target=None, policy_target=None, policy_mask=None):
        x = self.convBlock1(x)
        for block in self.residualBlocks:
            x = block(x)
        value, policy = self.valueHead(x), self.policyHead(x)

        if self.training:
            value_loss = self.mseLoss(value, value_target)
            policy_target = policy_target.view(policy_target.shape[0])
            policy_loss = self.cross_entropy_loss(policy, policy_target)
            return value_loss, policy_loss
        else:
            policy_mask = policy_mask.view(policy_mask.shape[0], -1)
            policy_exp = torch.exp(policy)
            policy_exp *= policy_mask.type(torch.float32)
            policy_exp_sum = torch.sum(policy_exp, dim=1, keepdim=True)
            policy_softmax = policy_exp / policy_exp_sum
            return value, policy_softmax
