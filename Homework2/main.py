import torch
from pca import pca
from lda import lda
from load import load_and_split_data
from plot import plot_eigenfaces, plot_2d_data

# 1. 加载数据
train_data, test_data, train_label, test_label = load_and_split_data()
train_data = train_data.cuda()
test_data = test_data.cuda()

# # 2. 学习PCA投影矩阵
# pca_components, pca_ratio = pca(train_data, n_components=8)

# # 显示PCA特征脸
# plot_eigenfaces(pca_components, "PCA Eigenfaces (First 8 Components)")

# # 3. 学习LDA投影矩阵
# lda_components, lda_ratio = lda(train_data, train_label, n_components=8)

# # 显示LDA Fisher脸
# plot_eigenfaces(lda_components, "LDA Fisherfaces (First 8 Components)")

# 4. 降维到2维并可视化
# PCA降维
pca_components_2d, _ = pca(train_data, n_components=2)
train_pca_2d = (train_data - torch.mean(train_data, axis=0)) @ pca_components_2d
test_pca_2d = (test_data - torch.mean(train_data, axis=0)) @ pca_components_2d

# LDA降维
lda_components_2d, _ = lda(train_data, train_label, n_components=2)
train_lda_2d = train_data @ lda_components_2d
test_lda_2d = test_data @ lda_components_2d

# 可视化
print("PCA Projection (Training Data, 2D)")
plot_2d_data(train_pca_2d, train_label, "PCA Projection (Training Data, 2D)")
print("PCA Projection (Test Data, 2D)")
plot_2d_data(test_pca_2d, test_label, "PCA Projection (Test Data, 2D)")
print("LDA Projection (Training Data, 2D)")
plot_2d_data(train_lda_2d, train_label, "LDA Projection (Training Data, 2D)")
print("LDA Projection (Test Data, 2D)")
plot_2d_data(test_lda_2d, test_label, "LDA Projection (Test Data, 2D)")