import numpy as np
import torch
import torch.nn.functional as F


def mixup_data(x, y, alpha=1.0):
    """执行Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def sharpen(p, T=0.5):
    """温度缩放锐化预测"""
    return (p ** (1/T)) / (p ** (1/T)).sum(dim=1, keepdim=True)


def mixup(X, Y, U, model, T=0.5, K=2, alpha=0.75):
    """对无标签数据和有标签数据 mixup 得到混合数据"""
    model.eval()
    
    # 为无标签数据生成伪标签
    with torch.no_grad():
        # 对无标签数据进行K次增强预测
        U_predictions = []
        for k in range(K):
            # 这里应该对U进行不同的数据增强，简化起见直接使用原数据
            pred = F.softmax(model(U), dim=1)
            U_predictions.append(pred)
        
        # 平均预测结果并锐化
        avg_pred = torch.stack(U_predictions).mean(dim=0)
        q_b = sharpen(avg_pred, T)
    
    model.train()
    
    # 构造增强标签数据
    X_hat = X
    Y_hat = Y
    
    # 构造增强无标签数据
    U_hat = U
    q_hat = q_b
    
    # 合并所有数据
    W = torch.cat([X_hat, U_hat], dim=0)
    labels = torch.cat([Y_hat, q_hat], dim=0)
    
    # 随机打乱
    idx = torch.randperm(W.size(0))
    W = W[idx]
    labels = labels[idx]
    
    # 分别对有标签和无标签数据进行MixUp
    X_size = X_hat.size(0)
    
    # 有标签数据MixUp
    X_mixed, labels_a_X, labels_b_X, lam_X = mixup_data(W[:X_size], labels[:X_size], alpha)
    
    # 无标签数据MixUp
    U_mixed, labels_a_U, labels_b_U, lam_U = mixup_data(W[X_size:], labels[X_size:], alpha)
    
    return X_mixed, (labels_a_X, labels_b_X, lam_X), U_mixed, (labels_a_U, labels_b_U, lam_U)
