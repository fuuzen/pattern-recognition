import cv2
import numpy as np
from harris import harris_corner_detection

# 读取两幅图像
img1 = cv2.imread('images/uttower1.jpg')
img2 = cv2.imread('images/uttower2.jpg')

# 获取关键点
kp1 = harris_corner_detection(img1)
kp2 = harris_corner_detection(img2)

# ================= HOG描述子 =================
def compute_hog(img, kps, win_size=(32, 32)):
  hog = cv2.HOGDescriptor(win_size, (16,16), (8,8), (8,8), 9)
  return [hog.compute(cv2.resize(img[max(0,y-16):y+16, max(0,x-16):x+16], win_size)) 
          for (x,y) in [map(int, kp.pt) for kp in kps]]

des1 = np.squeeze(compute_hog(img1, kp1))
des2 = np.squeeze(compute_hog(img2, kp2))

bf = cv2.BFMatcher(cv2.NORM_L2).match(des1, des2)
matches = sorted(bf, key=lambda x: x.distance)[:50]  # 取前50最佳匹配

res = cv2.drawMatches(
  img1, kp1, img2, kp2,
  matches[:50], None,  # 显示前50个最佳匹配
  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite('result/uttower_match_hog.png', res)