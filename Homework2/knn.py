import torch
import numpy as np
from collections import Counter

class KNN:
  def __init__(self, k=3):
    """
    初始化KNN分类器
    
    参数:
      k (int): 最近邻居的数量，默认为3
    """
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.k = k
    self.X_train = None
    self.y_train = None
    
  def fit(self, X_train, y_train):
    """
    训练KNN分类器
    
    参数:
      X_train (torch.Tensor 或 numpy.ndarray): 训练数据特征
      y_train (torch.Tensor 或 numpy.ndarray): 训练数据标签
    """
    # 转换为torch张量
    if not isinstance(X_train, torch.Tensor):
      X_train = torch.from_numpy(np.array(X_train)).float()
    X_train = X_train.float().to(self.device)
    if not isinstance(y_train, torch.Tensor):
      y_train = torch.from_numpy(np.array(y_train))
    y_train = y_train.float().to(self.device)
      
    self.X_train = X_train
    self.y_train = y_train
    
  def predict(self, X_test):
    """
    预测测试数据的标签
    
    参数:
      X_test (torch.Tensor 或 numpy.ndarray): 测试数据特征
      
    返回:
      torch.Tensor: 预测的标签
    """
    if not isinstance(X_test, torch.Tensor):
      X_test = torch.from_numpy(np.array(X_test)).float()
      
    # 计算L2距离(欧氏距离)
    distances = torch.cdist(X_test, self.X_train)
    
    # 获取k个最近邻居的索引
    _, indices = torch.topk(distances, self.k, largest=False)
    
    # 获取这些邻居的标签
    k_nearest_labels = self.y_train[indices]
    
    # 投票决定预测标签
    predictions = []
    for labels in k_nearest_labels:
      # 使用PyTorch的unique_consecutive和bincount实现投票
      unique_labels, counts = torch.unique_consecutive(
        labels.sort().values, 
        return_counts=True
      )
      most_common = unique_labels[torch.argmax(counts)]
      predictions.append(most_common)
      
    return torch.tensor(predictions)
  
  def evaluate(self, X_test, y_test):
    """
    评估模型在测试数据上的准确率
    
    参数:
      X_test (torch.Tensor 或 numpy.ndarray): 测试数据特征
      y_test (torch.Tensor 或 numpy.ndarray): 测试数据真实标签
      
    返回:
      float: 准确率
    """
    predictions = self.predict(X_test)
    
    if not isinstance(y_test, torch.Tensor):
      y_test = torch.from_numpy(np.array(y_test))
      
    accuracy = (predictions == y_test).float().mean()
    return accuracy.item()