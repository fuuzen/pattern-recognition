import cv2
import numpy as np

def sift_match(img1, img2, kp1, kp2):
  """
  SIFT特征匹配算法实现
  参数:
    img1: 输入图像1
    img2: 输入图像2
    kp1: 图像1的关键点列表
    kp2: 图像2的关键点列表
  返回:
    [DMatch]: 匹配的特征点列表
  """
  sift = cv2.SIFT.create()
  _, des1 = sift.compute(img1, kp1)
  _, des2 = sift.compute(img2, kp2)
  bf = cv2.BFMatcher(cv2.NORM_L2)  # 欧几里得距离
  matches = bf.match(des1, des2)
  return sorted(matches, key=lambda x: x.distance)[:50]  # 取前50最佳匹配
