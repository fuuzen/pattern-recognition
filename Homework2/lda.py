import torch

def lda(X, y, n_components=None):
  """
  线性判别分析(LDA)的PyTorch GPU实现
  
  参数:
  X : torch.Tensor, 形状为 (n_samples, n_features)
    输入数据矩阵
  y : torch.Tensor, 形状为 (n_samples,)
    类别标签
  n_components : int 或 None
    要保留的维度数量。如果为None，则保留最多min(n_features, n_classes-1)个维度
  device : str
    使用的设备 ('cuda' 或 'cpu')
    
  返回:
  W : torch.Tensor, 形状为 (n_features, n_components)
    投影矩阵
  explained_variance_ratio : torch.Tensor, 形状为 (n_components,)
    每个维度的解释方差比例
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # 将数据移动到指定设备
  if not isinstance(X, torch.Tensor):
    X = torch.from_numpy(X).float().to(device)
  else:
    X = X.float().to(device)
  
  if not isinstance(y, torch.Tensor):
    y = torch.from_numpy(y).float().to(device)
  else:
    y = y.float().to(device)
  
  # 获取类别信息
  classes = torch.unique(y)
  n_classes = len(classes)
  n_samples, n_features = X.shape
  
  # 如果没有指定n_components，则设置为最大可能值
  if n_components is None:
    n_components = min(n_features, n_classes - 1)
  elif n_components > min(n_features, n_classes - 1):
    n_components = min(n_features, n_classes - 1)
    print(f"警告: n_components 不能超过 min(n_features, n_classes-1)={n_components}，已自动调整")
  
  # 1. 计算总体均值
  overall_mean = torch.mean(X, dim=0)
  
  # 2. 计算类内散度矩阵(S_w)和类间散度矩阵(S_b)
  S_w = torch.zeros((n_features, n_features), device=device)  # 类内散度矩阵
  S_b = torch.zeros((n_features, n_features), device=device)  # 类间散度矩阵
  
  for c in classes:
    # 获取当前类别的样本
    X_c = X[y == c]
    # 计算当前类别的均值
    mean_c = torch.mean(X_c, dim=0)
    # 计算类内散度
    X_c_centered = X_c - mean_c
    S_w += X_c_centered.T @ X_c_centered
    # 计算类间散度
    n_c = X_c.shape[0]
    mean_diff = (mean_c - overall_mean).reshape(-1, 1)
    S_b += n_c * (mean_diff @ mean_diff.T)

  # 3. 求解广义特征值问题 S_b * W = lambda * S_w * W
  # 使用伪逆矩阵避免S_w奇异的问题
  S_w_pinv = torch.linalg.pinv(S_w)
  mat = S_w_pinv @ S_b
  
  # 计算特征值和特征向量
  eigenvalues, eigenvectors = torch.linalg.eig(mat)
  
  # 确保特征值和特征向量是实数(对于实对称矩阵应该是实数，但数值计算可能产生微小虚部)
  eigenvalues = eigenvalues.real
  eigenvectors = eigenvectors.real
  
  # 4. 按特征值降序排序
  sorted_indices = torch.argsort(eigenvalues, descending=True)
  sorted_eigenvalues = eigenvalues[sorted_indices]
  sorted_eigenvectors = eigenvectors[:, sorted_indices]
  
  # 5. 选择前 n_components 个特征向量
  W = sorted_eigenvectors[:, :n_components]
  
  # 计算解释方差比例
  explained_variance_ratio = sorted_eigenvalues[:n_components] / torch.sum(sorted_eigenvalues)
  
  return W, explained_variance_ratio