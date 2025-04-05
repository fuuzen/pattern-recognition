import torch

def pca(X, n_components=None):
  """
  主成分分析 (PCA) 的 PyTorch GPU 实现
  
  参数:
    X: torch.Tensor, 形状为 (n_samples, n_features)
    n_components: int, 主成分数量，如果为None，则返回所有主成分
    device: str, 使用的设备 ('cuda' 或 'cpu')
    
  返回: 
    sorted_eigenvectors: 排序后的特征向量 (n_features, n_components)
    explained_variance_ratio: 每个主成分的方差比率 (n_components,)
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 确保输入是 PyTorch Tensor 并移动到指定设备
  if not isinstance(X, torch.Tensor):
    X = torch.from_numpy(X).float().to(device)
  else:
    X = X.float().to(device)
  
  # 中心化数据
  X_centered = X - torch.mean(X, dim=0)
  
  # 计算协方差矩阵 (使用矩阵乘法更高效)
  cov_matrix = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
  
  # 特征分解
  eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
  
  # 按特征值降序排序
  sorted_indices = torch.argsort(eigenvalues, descending=True)
  sorted_eigenvalues = eigenvalues[sorted_indices]
  sorted_eigenvectors = eigenvectors[:, sorted_indices]
  
  # 选择前n_components个主成分
  if n_components is not None:
    sorted_eigenvectors = sorted_eigenvectors[:, :n_components]
    sorted_eigenvalues = sorted_eigenvalues[:n_components]
  
  # 计算解释方差比例
  explained_variance_ratio = sorted_eigenvalues / torch.sum(sorted_eigenvalues)
  
  return sorted_eigenvectors, explained_variance_ratio