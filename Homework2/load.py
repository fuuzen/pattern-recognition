from scipy import io
import torch

def load_and_split_data(device='cuda'):
  """
  读取数据集并划分训练测试集，返回PyTorch Tensor
  
  参数:
  device : str
    使用的设备 ('cuda' 或 'cpu')
      
  返回:
  train_data : torch.Tensor, 形状为 (n_train_samples, input_dim)
  test_data : torch.Tensor, 形状为 (n_test_samples, input_dim)
  train_label : torch.Tensor, 形状为 (n_train_samples,)
  test_label : torch.Tensor, 形状为 (n_test_samples,)
  """
  # 读取MAT文件
  x = io.loadmat('Yale_64x64.mat')
  
  # 定义常量
  ins_perclass, class_number, train_test_split = 11, 15, 9
  input_dim = x['fea'].shape[1]
  
  # 重塑数据
  feat = x['fea'].reshape(-1, ins_perclass, input_dim)
  label = x['gnd'].reshape(-1, ins_perclass)
  
  # 划分训练集和测试集
  train_data = feat[:, :train_test_split, :].reshape(-1, input_dim)
  test_data = feat[:, train_test_split:, :].reshape(-1, input_dim)
  train_label = label[:, :train_test_split].reshape(-1)
  test_label = label[:, train_test_split:].reshape(-1)
  
  # 转换为PyTorch Tensor并移动到指定设备
  train_data = torch.from_numpy(train_data).float()
  test_data = torch.from_numpy(test_data).float()
  train_label = torch.from_numpy(train_label).long()  # 标签通常用long类型
  test_label = torch.from_numpy(test_label).long()
  
  return train_data, test_data, train_label, test_label