import cv2
import numpy as np

def harris_corner_detection(img, window_size=3, k=0.04, threshold=0.01):
  """
  Harris角点检测算法实现
  
  参数:
    image: 输入图像
    window_size: Sobel算子窗口大小
    k: Harris角点检测方程中的经验常数(通常0.04-0.06)
    threshold: 角点响应阈值(0-1之间的小数)
    
  返回:
    [KeyPoint]: 检测到的角点坐标列表
  """
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 1. 计算图像梯度
  Ix = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=window_size)
  Iy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=window_size)
  
  # 2. 计算梯度的乘积
  Ix2 = Ix * Ix
  Iy2 = Iy * Iy
  Ixy = Ix * Iy
  
  # 3. 使用高斯滤波对梯度乘积进行加权求和
  window = np.ones((window_size, window_size), dtype=np.float64)
  Sx2 = cv2.filter2D(Ix2, -1, window)
  Sy2 = cv2.filter2D(Iy2, -1, window)
  Sxy = cv2.filter2D(Ixy, -1, window)
  
  # 4. 计算每个像素的角点响应函数R
  det = (Sx2 * Sy2) - (Sxy ** 2)
  trace = Sx2 + Sy2
  R = det - k * (trace ** 2)
  
  # 5. 应用非极大值抑制和阈值处理
  keypoints = []
  R_max = np.max(R)
  height, width = img_gray.shape
  
  for y in range(height):
    for x in range(width):
      # 阈值处理
      if R[y, x] > threshold * R_max:
        # 非极大值抑制 - 检查是否是局部最大值
        is_local_max = True
        for dy in [-1, 0, 1]:
          for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
              continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
              if R[ny, nx] > R[y, x]:
                is_local_max = False
                break
          if not is_local_max:
            break
        
        if is_local_max:
          # 创建KeyPoint对象
          # 参数: x, y, size, angle, response, octave, class_id
          # 这里我们主要设置位置(x,y)和响应值(response)
          kp = cv2.KeyPoint(
            x=float(x), 
            y=float(y), 
            size=window_size,  # 特征点邻域直径
            angle=-1,         # 方向未计算设为-1
            response=float(R[y, x]),  # 角点响应值
            octave=0,         # 金字塔层数
            class_id=-1       # 未分类
          )
          keypoints.append(kp)
  
  return keypoints
