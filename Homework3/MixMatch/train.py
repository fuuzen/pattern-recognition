import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .mixmatch import mixup_criterion, mixmatch


def train(evaluate, model, labeled_loader, unlabeled_loader, test_loader, 
          num_iters=20000, lambda_u=75, lr=0.002, device='cuda',
          eval_iter=1000):
    """训练MixMatch模型"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
    
    labeled_criterion = nn.CrossEntropyLoss()
    unlabeled_criterion = nn.MSELoss()
    
    train_losses = []
    test_accuracies = []
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    total_loss = 0
    labeled_loss_total = 0
    unlabeled_loss_total = 0
    
    for iteration in range(num_iters):
        # 获取有标签数据
        labeled_batch = next(labeled_iter)
        labeled_data, labeled_targets = labeled_batch
        labeled_data, labeled_targets = labeled_data.to(device), labeled_targets.to(device)
        
        # 获取无标签数据
        unlabeled_batch = next(unlabeled_iter)
        (unlabeled_data1, unlabeled_data2), _ = unlabeled_batch
        unlabeled_data = unlabeled_data1.to(device)  # 使用第一个增强版本
        
        # 转换标签为one-hot编码
        labeled_targets_oh = F.one_hot(labeled_targets, num_classes=10).float()
        
        # 执行MixMatch
        X_mixed, (labels_a_X, labels_b_X, lam_X), U_mixed, (labels_a_U, labels_b_U, lam_U) = \
            mixmatch(labeled_data, labeled_targets_oh, unlabeled_data, model)
        
        # 前向传播
        logits_X = model(X_mixed)
        logits_U = model(U_mixed)
        
        # 计算有标签损失
        labeled_loss = mixup_criterion(labeled_criterion, logits_X, 
                                        labels_a_X.argmax(dim=1), labels_b_X.argmax(dim=1), lam_X)
        
        # 计算无标签损失
        probs_U = F.softmax(logits_U, dim=1)
        unlabeled_loss = lam_U * unlabeled_criterion(probs_U, labels_a_U) + \
                        (1 - lam_U) * unlabeled_criterion(probs_U, labels_b_U)
        
        # 总损失
        loss = labeled_loss + lambda_u * unlabeled_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        labeled_loss_total += labeled_loss.item()
        unlabeled_loss_total += unlabeled_loss.item()
        
        scheduler.step()
        
        # 评估
        if iteration % eval_iter == 0:
            model.save('./model/latest_model.pth', optimizer=optimizer, iter=iteration, loss=loss)
            
            test_acc = evaluate(model, test_loader, device)
            train_losses.append(loss.item())
            test_accuracies.append(test_acc)
            
            print(f'Iteration {iteration}/{num_iters}:')
            print(f'  Total Loss: {loss.item():.4f}')
            print(f'  Labeled Loss: {labeled_loss.item():.4f}')
            print(f'  Unlabeled Loss: {unlabeled_loss.item():.4f}')
            print(f'  Test Accuracy: {test_acc:.4f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
    
    return train_losses, test_accuracies
