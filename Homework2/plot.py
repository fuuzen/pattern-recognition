import numpy as np
import matplotlib.pyplot as plt

# 可视化函数


def plot_eigenfaces(eigenvectors, title, n_faces=8, img_shape=(64, 64)):
  if hasattr(eigenvectors, 'numpy'):
    eigenvectors = eigenvectors.cpu().numpy()
  
  plt.figure(figsize=(12, 6))
  plt.suptitle(title, fontsize=16)
  for i in range(n_faces):
    plt.subplot(2, 4, i+1)
    eigenface = eigenvectors[:, i].reshape(img_shape)
    plt.imshow(eigenface, cmap='gray')
    plt.title(f'Component {i+1}')
    plt.axis('off')
  plt.tight_layout()
  plt.show()

def plot_2d_data(X, y, title, n_classes=15):
  if hasattr(X, 'numpy'):
    X = X.cpu().numpy()
  if hasattr(y, 'numpy'):
    y = y.cpu().numpy()
  plt.figure(figsize=(10, 8))
  colors = plt.cm.get_cmap('tab20', n_classes)
  for i in range(n_classes):
    plt.scatter(
      X[y == i+1, 0],
      X[y == i+1, 1],
      color=colors(i),
      label=f'Class {i+1}',
      alpha=0.7
    )
  plt.title(title)
  plt.xlabel('First Component')
  plt.ylabel('Second Component')
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid()
  plt.tight_layout()
  plt.show()


def plot_accuracy_reduced_dim(
  reduced_dims, 
  accuracies_list, 
  labels=None, 
  title="Accuracy - Reduced Dimension", 
  xlabel="Reduced Dimension", 
  ylabel="Accuracy"
):
  plt.figure(figsize=(8, 5))
  
  # Default labels if not provided
  if labels is None:
    labels = [f"Method {i+1}" for i in range(len(accuracies_list))]
  
  # Colors and markers for differentiation
  colors = plt.cm.tab10.colors  # Use a color map for distinct colors
  markers = ['o', 's', '^', 'D', 'v', 'p', '*']  # Different markers
  
  for i, (accuracies, label) in enumerate(zip(accuracies_list, labels)):
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(
      reduced_dims, 
      accuracies, 
      marker=marker, 
      linestyle='-', 
      color=color, 
      label=label,
      markersize=8,
      linewidth=2
    )
  
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend(loc="best")
  
  # Set integer x-ticks if dimensions are integers
  if all(isinstance(dim, int) for dim in reduced_dims):
    plt.xticks(reduced_dims)
  
  plt.tight_layout()
  plt.show()