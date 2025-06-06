import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(evaluate, model, labeled_loader, unlabeled_loader, test_loader, 
          num_epochs=20000, lambda_u=1.0, threshold=0.95, lr=0.03, device='cuda'):
    """训练FixMatch模型"""
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    
    labeled_criterion = nn.CrossEntropyLoss()
    unlabeled_criterion = nn.CrossEntropyLoss(reduction='none')
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        labeled_loss_total = 0
        unlabeled_loss_total = 0
        mask_prob_total = 0
        
        # 创建迭代器
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        num_batches = max(len(labeled_loader), len(unlabeled_loader))
        
        for batch_idx in range(num_batches):
            # 获取有标签数据
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)
            
            labeled_data, labeled_targets = labeled_batch
            labeled_data, labeled_targets = labeled_data.to(device), labeled_targets.to(device)
            
            # 获取无标签数据
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            (weak_unlabeled, strong_unlabeled), _ = unlabeled_batch
            weak_unlabeled = weak_unlabeled.to(device)
            strong_unlabeled = strong_unlabeled.to(device)
            
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
            
            total_loss += loss.item()
            labeled_loss_total += labeled_loss.item()
            unlabeled_loss_total += unlabeled_loss.item()
            mask_prob_total += mask.float().mean().item()
        
        scheduler.step()
        
        # 评估
        if epoch % 10 == 0:
            test_acc = evaluate(model, test_loader, device)
            train_losses.append(total_loss / num_batches)
            test_accuracies.append(test_acc)
            
            print(f'Epoch {epoch}/{num_epochs}:')
            print(f'  Total Loss: {total_loss/num_batches:.4f}')
            print(f'  Labeled Loss: {labeled_loss_total/num_batches:.4f}')
            print(f'  Unlabeled Loss: {unlabeled_loss_total/num_batches:.4f}')
            print(f'  Mask Ratio: {mask_prob_total/num_batches:.4f}')
            print(f'  Test Accuracy: {test_acc:.4f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
    
    return train_losses, test_accuracies
