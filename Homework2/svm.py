from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

class SVM:
  def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
    """
    初始化 SVM 分类器
    :param kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
    :param C: 正则化参数
    :param gamma: 'scale', 'auto' 或浮点数
    """
    self.svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=False)
    self.scaler = StandardScaler()
    self.is_trained = False
  
  def fit(self, X_train, y_train):
    """
    训练 SVM 模型
    :param X_train: 训练特征 (numpy array)
    :param y_train: 训练标签 (numpy array)
    """
    if isinstance(X_train, torch.Tensor):
      X_train = X_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
      y_train = y_train.cpu().numpy()
    
    # 数据标准化
    X_train = self.scaler.fit_transform(X_train)
    
    # 训练 SVM
    self.svm.fit(X_train, y_train)
    self.is_trained = True
  
  def evaluate(self, X_test, y_test):
    """
    评估模型准确率
    :param X_test: 测试特征 (numpy array)
    :param y_test: 测试标签 (numpy array)
    :return: 准确率
    """
    if isinstance(X_test, torch.Tensor):
      X_test = X_test.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
      y_test = y_test.cpu().numpy()
    
    if not self.is_trained:
      raise RuntimeError("Model not trained yet. Call train() first.")
    
    # 数据标准化
    X_test = self.scaler.transform(X_test)
    
    # 预测并计算准确率
    y_pred = self.svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy
  
  def predict(self, X_test):
    """
    使用训练好的模型进行预测
    :param X: 输入特征 (numpy array)
    :return: 预测结果 (numpy array)
    """
    if not self.is_trained:
      raise RuntimeError("Model not trained yet. Call train() first.")
    
    # 数据标准化
    X_test = self.scaler.transform(X_test)
    
    return self.svm.predict(X_test)

  