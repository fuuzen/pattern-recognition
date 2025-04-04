import cv2
import numpy as np

def compute_hog(img, kps, win_size=(32, 32)):
  """
  计算HOG特征描述子
  参数:
    img: 输入图像
    kps: 关键点列表
    win_size: HOG窗口大小
  返回:
    hog_features: HOG特征列表
  """
  hog = cv2.HOGDescriptor(win_size, (16,16), (8,8), (8,8), 9)
  return [
    hog.compute(
      cv2.resize(
        img[
          max(0, y - 16) : y + 16,
          max(0, x - 16) : x + 16
        ],
        win_size
      )
    ) for (x,y) in [
      map(int, kp.pt) for kp in kps
    ]
  ]

def hog_match(img1, img2, kp1, kp2):
  """
  HOG特征匹配算法实现
  参数:
    img1: 输入图像1
    img2: 输入图像2
    kp1: 图像1的关键点列表
    kp2: 图像2的关键点列表
  返回:
    [DMatch]: 匹配的特征点列表
  """
  des1 = np.squeeze(compute_hog(img1, kp1))
  des2 = np.squeeze(compute_hog(img2, kp2))
  bf = cv2.BFMatcher(cv2.NORM_L2)  # 欧几里得距离
  match = bf.match(des1, des2)
  return sorted(match, key=lambda x: x.distance)[:50]  # 取前50最佳匹配
