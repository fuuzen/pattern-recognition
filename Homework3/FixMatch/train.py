import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(evaluate, model, labeled_loader, unlabeled_loader, test_loader, 
          num_iters=20000, lambda_u=1.0, threshold=0.95, lr=0.03, device='cuda',
          eval_step=1000):
    """训练FixMatch模型"""
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=0)
    
    labeled_criterion = nn.CrossEntropyLoss()
    unlabeled_criterion = nn.CrossEntropyLoss(reduction='none')
    
    train_losses = []
    test_accuracies = []

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    
    model.train()
    
    for iteration in range(num_iters):
        # 获取有标签数据
        labeled_data, labeled_targets = next(labeled_iter)
        labeled_data, labeled_targets = labeled_data.to(device), labeled_targets.to(device)
        
        # 获取无标签数据  
        (weak_unlabeled, strong_unlabeled), _ = next(unlabeled_iter)
        weak_unlabeled, strong_unlabeled = weak_unlabeled.to(device), strong_unlabeled.to(device)
        
        batch_size = labeled_data.shape[0]
        
        # 合并所有数据进行前向传播
        all_inputs = torch.cat([labeled_data, weak_unlabeled, strong_unlabeled], dim=0)
        all_logits = model(all_inputs)
        
        # 分离输出
        labeled_logits = all_logits[:batch_size]
        weak_unlabeled_logits = all_logits[batch_size:batch_size + weak_unlabeled.shape[0]]
        strong_unlabeled_logits = all_logits[batch_size + weak_unlabeled.shape[0]:]
        
        # 计算有标签损失
        labeled_loss = labeled_criterion(labeled_logits, labeled_targets)
        
        # 生成伪标签（使用弱增强的预测）
        with torch.no_grad():
            weak_probs = F.softmax(weak_unlabeled_logits, dim=1)
            max_probs, pseudo_labels = torch.max(weak_probs, dim=1)
            mask = max_probs >= threshold
        
        # 计算无标签损失（只对高置信度样本）
        if mask.sum() > 0:
            unlabeled_loss = (unlabeled_criterion(strong_unlabeled_logits, pseudo_labels) * mask).mean()
        else:
            unlabeled_loss = torch.tensor(0.0).to(device)
        
        # 总损失
        loss = labeled_loss + lambda_u * unlabeled_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 定期评估和打印
        if iteration % eval_step == 0:
            test_acc = evaluate(model, test_loader, device)
            train_losses.append(loss.item())
            test_accuracies.append(test_acc)
            
            print(f'Iteration {iteration}/{num_iters}:')
            print(f'  Total Loss: {loss.item():.4f}')
            print(f'  Labeled Loss: {labeled_loss.item():.4f}')
            print(f'  Unlabeled Loss: {unlabeled_loss.item():.4f}')
            print(f'  Mask Ratio: {mask.float().mean().item():.4f}')
            print(f'  Test Accuracy: {test_acc:.4f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
    
    return train_losses, test_accuracies
