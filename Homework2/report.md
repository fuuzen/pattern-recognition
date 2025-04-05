<div class="cover" style="page-break-after:always;font-family:方正公文仿宋;width:100%;height:100%;border:none;margin: 0 auto;text-align:center;">
    <div style="width:50%;margin: 0 auto;height:0;padding-bottom:10%;">
        </br>
        <img src="../sysu-name.png" alt="校名" style="width:100%;"/>
    </div>
    </br></br>
    <div style="width:40%;margin: 0 auto;height:0;padding-bottom:40%;">
        <img src="../sysu.png" alt="校徽" style="width:100%;"/>
    </div>
		</br></br></br>
    <span style="font-family:华文黑体Bold;text-align:center;font-size:20pt;margin: 10pt auto;line-height:30pt;">本科生实验报告</span>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
        <tr style="font-weight:normal;"> 
            <td style="width:20%;text-align:center;">课程名称</td>
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">模式识别</td>
      </tr>
        <tr style="font-weight:normal;"> 
            <td style="width:20%;text-align:center;">实验名称</td>
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">基于 PCA/LDA 和 KNN 的人脸识别</td>
      </tr>
        <tr style="font-weight:normal;"> 
            <td style="width:20%;text-align:center;">专业名称</td>
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">计算机科学与技术</td>
      </tr>
        <tr style="font-weight:normal;"> 
            <td style="width:20%;text-align:center;">学生姓名</td>
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">李世源</td>
      </tr>
        <tr style="font-weight:normal;"> 
            <td style="width:20%;text-align:center;">学生学号</td>
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">22342043</td>
      </tr>
        <tr style="font-weight:normal;"> 
            <td style="width:20%;text-align:center;">实验成绩</td>
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"></td>
      </tr>
      <tr style="font-weight:normal;"> 
            <td style="width:20%;text-align:center;">报告时间</td>
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">2025年04月05日</td>
      </tr>
    </tbody>              
    </table>
</div>

<!-- 注释语句：导出PDF时会在这里分页，使用 Typora Newsprint 主题放大 125% -->


## 一、实验目的

1. 掌握 PCA（主成分分析）和 LDA（线性判别分析）的基本原理，并实现数据降维算法。
2. 实现 KNN（K-最近邻）分类器，并利用降维后的数据进行分类。
3. 分析不同降维方法对分类准确率的影响。

---

## 二、关键数学原理

### 1. PCA（主成分分析）

PCA 通过线性变换将高维数据投影到低维空间，目标是保留数据的主要方差。其步骤如下：

1. **中心化数据**：  
   计算数据的均值向量，并将数据减去均值：
   $$
   X_{\text{centered}} = X - \mu, \quad \mu = \frac{1}{N} \sum_{i=1}^N x_i.
   $$

2. **计算协方差矩阵**：  
   协方差矩阵描述数据的分布情况：
   $$
   \text{Cov} = \frac{1}{N-1} X_{\text{centered}}^T X_{\text{centered}}.
   $$

3. **特征分解**：  
   对协方差矩阵进行特征分解，得到特征值和特征向量：
   $$
   \text{Cov} \cdot v = \lambda v.
   $$

4. **选择主成分**：  
   按特征值降序排序，选择前 \(k\) 个特征向量作为投影矩阵 \(W\)。

5. **降维**：  
   将数据投影到低维空间：
   $$
   X_{\text{reduced}} = X_{\text{centered}} \cdot W.
   $$

### 2. LDA（线性判别分析）

LDA 是一种监督降维方法，目标是最大化类间散度与类内散度的比值。其步骤如下：

1. **计算类内散度矩阵 \(S_w\)**：  
   描述同一类别内数据的分散程度：
   $$
   S_w = \sum_{c=1}^C \sum_{x_i \in c} (x_i - \mu_c)(x_i - \mu_c)^T.
   $$

2. **计算类间散度矩阵 \(S_b\)**：  
   描述不同类别之间的分散程度：
   $$
   S_b = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T.
   $$

3. **求解广义特征值问题**：  
   求解 \(S_b \cdot W = \lambda S_w \cdot W\)，得到投影矩阵 \(W\)。

4. **降维**：  
   将数据投影到低维空间：
   $$
   X_{\text{reduced}} = X \cdot W.
   $$

### 3. KNN（K-最近邻）

KNN 是一种基于距离的分类算法，其步骤如下：

1. 计算测试样本与所有训练样本的距离（如欧氏距离）。
2. 选择距离最近的 \(k\) 个训练样本。
3. 根据这 \(k\) 个样本的标签进行投票，预测测试样本的类别。

---

## 三、关键代码解析

### 1. PCA 实现

```python
def pca(X, n_components=None):
    # 中心化数据
    X_centered = X - torch.mean(X, dim=0)
    
    # 计算协方差矩阵
    cov_matrix = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
    
    # 特征分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # 按特征值降序排序
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 选择前 n_components 个主成分
    if n_components is not None:
        sorted_eigenvectors = sorted_eigenvectors[:, :n_components]
    
    return sorted_eigenvectors
```

### 2. LDA 实现

```python
def lda(X, y, n_components=None):
    # 计算总体均值
    overall_mean = torch.mean(X, dim=0)
    
    # 初始化散度矩阵
    S_w = torch.zeros((n_features, n_features))
    S_b = torch.zeros((n_features, n_features))
    
    # 计算类内和类间散度矩阵
    for c in classes:
        X_c = X[y == c]
        mean_c = torch.mean(X_c, dim=0)
        S_w += (X_c - mean_c).T @ (X_c - mean_c)
        S_b += len(X_c) * (mean_c - overall_mean).reshape(-1, 1) @ (mean_c - overall_mean).reshape(1, -1)
    
    # 求解广义特征值问题
    S_w_pinv = torch.linalg.pinv(S_w)
    mat = S_w_pinv @ S_b
    eigenvalues, eigenvectors = torch.linalg.eig(mat)
    
    # 选择前 n_components 个特征向量
    W = eigenvectors[:, :n_components]
    return W
```

### 3. KNN 实现

```python
class KNN:
    def predict(self, X_test):
        # 计算欧氏距离
        distances = torch.cdist(X_test, self.X_train)
        
        # 获取 k 个最近邻居的索引
        _, indices = torch.topk(distances, self.k, largest=False)
        
        # 投票决定预测标签
        k_nearest_labels = self.y_train[indices]
        predictions = []
        for labels in k_nearest_labels:
            unique_labels, counts = torch.unique_consecutive(labels.sort().values, return_counts=True)
            most_common = unique_labels[torch.argmax(counts)]
            predictions.append(most_common)
        
        return torch.tensor(predictions)
```

### 4. 存储降维数据

我发现每次使用 LDA 进行降维都需要不少时间，一次 LDA 降维大概在我的机器上需要 40~50秒， 而其他计算时间都非常短。我希望不同维度、不同分类器测试比较时每次能直接使用已经降维好的数据，避免每次都降维带来不必要的开销和等待时间，所以我增加了降维数据的存储，存到 `reduced_data` 目录下，采用 HDF5 格式，因为该格式适合按不同维度进行存储。代码如下：

```python
with h5py.File("reduced_data/pca.h5", "w") as f:
  for dim in reduced_dims:
    pca_components, _ = pca(train_data, n_components=dim)
    train_pca = (train_data - torch.mean(train_data, axis=0)) @ pca_components
    test_pca = (test_data - torch.mean(train_data, axis=0)) @ pca_components
    grp = f.create_group(f"dim_{dim}")
    grp.create_dataset("train", data=train_pca.cpu())
    grp.create_dataset("test", data=test_pca.cpu())

print("PCA reduced data saved to reduced_data/pca.h5")

with h5py.File("reduced_data/lda.h5", "w") as f:
  for dim in reduced_dims:
    lda_components, _ = lda(train_data, train_label, n_components=dim)
    train_pca = train_data @ lda_components
    test_pca = test_data @ lda_components
    grp = f.create_group(f"dim_{dim}")
    grp.create_dataset("train", data=train_pca.cpu())
    grp.create_dataset("test", data=test_pca.cpu())

print("LDA reduced data saved to reduced_data/lda.h5")
```

---

## 四、实验结果与分析

### 1. 投影图像

PCA 前 8 个主成分对应的特征脸如图所示，反映了数据的主要变化方向 ：

<img src="images/PCA Eigenfaces (First 8 Components).png" alt="PCA Eigenfaces (First 8 Components)" style="zoom:40%;" />

LDA 前 8 个判别向量如图所示，更注重类别间的区分：

<img src="images/LDA Fisherfaces (First 8 Components).png" alt="LDA Fisherfaces (First 8 Components)" style="zoom:40%;" />


### 2. 2D 降维可视化

PCA 和 LDA 分别将测试数据降至 2 维后的可视化图如下：

<img src="images/PCA Projection (Training Data, 2D).png" alt="PCA Projection (Training Data, 2D)" style="zoom:25%;" /><img src="images/LDA Projection (Training Data, 2D).png" alt="LDA Projection (Training Data, 2D)" style="zoom:25%;" />

PCA 和 LDA 分别将测试数据降至 2 维后的可视化图如下：

<img src="images/PCA Projection (Test Data, 2D).png" alt="PCA Projection (Test Data, 2D)" style="zoom:25%;" /><img src="images/LDA Projection (Test Data, 2D).png" alt="LDA Projection (Test Data, 2D)" style="zoom:25%;" />

这些可视化图体现了 PCA 更关注全局方差，而 LDA 更关注类别分离的特点。

### 3. KNN 分类准确率

降维至 1-8 维后，用 PCA 和 LDA 降维训练和测试数据评估的 KNN 分类准确率如下所示：

<img src="images/Accuracies evaluated by KNN - PCA&LCA reduced dims.png" alt="Accuracies evaluated by KNN - PCA&LCA reduced dims" style="zoom:50%;" />

随着降维维度的增加，准确率逐渐提升。LDA 在低维时表现更好，而 PCA 在高维时逐渐接近甚至超过 LDA 的性能。

## 五、选做部分

我利用 scipy-learn 库提供的 SVM，在 KNN 的基础上，增加了和 SVM 分类器对比的测试，得到用 PCA 和 LDA 降维训练和测试数据评估的 KNN、SVM 分类准确率如下：

<img src="images/Accuracies evaluated by KNN & SVM - PCA&LCA reduced dims.png" alt="Accuracies evaluated by KNN & SVM - PCA&LCA reduced dims" style="zoom:50%;" />

可以看到整体上用 LDA 降维训练和测试数据评估的 SVM 分类准确率最好，LDA 降维后评估出来的准确率整体都比 PCA 降维的要好，尤其是维度较低部分。

---

## 六、实验总结

1. PCA 和 LDA 是两种有效的降维方法，PCA 适用于无监督任务，LDA 适用于有监督任务。
2. 分类器的性能受降维方法影响较大，LDA 在分类任务中表现更优。
3. 实验结果表明，选择合适的降维方法和维度对提高分类准确率至关重要。
