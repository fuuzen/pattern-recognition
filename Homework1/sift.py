import cv2
import numpy as np
from harris import harris_corner_detection

# 读取两幅图像
img1 = cv2.imread('images/uttower1.jpg')
img2 = cv2.imread('images/uttower2.jpg')

# 获取关键点
kp1 = harris_corner_detection(img1)
kp2 = harris_corner_detection(img2)

# ================= SIFT描述子 =================
sift = cv2.SIFT.create()
_, des1_sift = sift.compute(img1, kp1)
_, des2_sift = sift.compute(img2, kp2)

# SIFT特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2)  # 欧几里得距离
matches = bf.match(des1_sift, des2_sift)
matches = sorted(matches, key=lambda x: x.distance)  # 按距离排序

# 绘制SIFT匹配结果
res = cv2.drawMatches(
  img1, kp1, img2, kp2, 
  matches[:50], None,  # 显示前50个最佳匹配
  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite("uttower_match_sift.png", res)