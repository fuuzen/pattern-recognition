import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout_rate = dropout_rate
        self.equal_in_out = (in_planes == out_planes)
        self.shortcut = (not self.equal_in_out) and nn.Conv2d(in_planes, out_planes,
                                                              kernel_size=1, stride=stride,
                                                              padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_in_out else self.shortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, dropout_rate=0.3, num_classes=10):
        self.depth=depth
        self.widen_factor=widen_factor
        self.dropout_rate=dropout_rate
        self.num_classes=num_classes

        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # 三个残差块组
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        
        # 全局平均池化和分类器
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def save(self, filepath, save_config=True, optimizer=None, iter=None, loss=None):
        """保存模型"""
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 准备保存的数据
        save_data = {
            'model_state_dict': self.state_dict(),
        }
        
        # 保存模型配置
        if save_config:
            save_data['config'] = {
                'depth': self.depth,
                'widen_factor': self.widen_factor,
                'dropout_rate': self.dropout_rate,
                'num_classes': self.num_classes
            }
        
        # 保存优化器状态
        if optimizer is not None:
            save_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # 保存训练信息
        if iter is not None:
            save_data['iteration'] = iter
        if loss is not None:
            save_data['loss'] = loss
        
        # 保存到文件
        torch.save(save_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath, load_optimizer=None, device=None):
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载模型文件
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # 加载模型参数
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if load_optimizer is not None and 'optimizer_state_dict' in checkpoint:
            load_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 返回训练信息
        info = {}
        if 'iter' in checkpoint:
            info['iter'] = checkpoint['iter']
        if 'loss' in checkpoint:
            info['loss'] = checkpoint['loss']
        if 'config' in checkpoint:
            info['config'] = checkpoint['config']
        
        print(f"模型已从 {filepath} 加载")
        return info