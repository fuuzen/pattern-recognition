import cv2
import numpy as np

from harris import harris_detection
from sift import sift_match
from hog import hog_match


def ransac_stitch(images, HOG=False):
  """
  拼接多个图像的函数
  
  参数:
    images: 要拼接的图像列表(按顺序从左到右或从上到下)
    HOG: 是否使用HOG特征进行匹配, 默认为False, 使用SIFT特征
  
  返回:
    拼接后的全景图像
  """
  if len(images) < 2:
    raise ValueError("至少需要2张图像进行拼接")
  
  # 初始化，先拼接前两张图像
  panorama = images[0]
  
  for i in range(1, len(images)):
    # 获取当前要拼接的图像
    next_img = images[i]
    
    # 检测特征点和匹配
    kp1 = harris_detection(panorama)
    kp2 = harris_detection(next_img)
    if HOG:
      matches = hog_match(panorama, next_img, kp1, kp2)
    else:
      matches = sift_match(panorama, next_img, kp1, kp2)
    
    if len(matches) < 10:
      print(f"警告: 图像{i}匹配点不足({len(matches)}), 跳过拼接")
      continue
    
    # 准备RANSAC的输入数据
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # 使用RANSAC求解单应性矩阵
    matrix, _ = cv2.findHomography(
      src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0
    )
    
    if matrix is None:
      print(f"警告: 无法计算图像{i}的单应性矩阵, 跳过拼接")
      continue
    
    # 计算拼接后图像大小
    h1, w1 = panorama.shape[:2]
    h2, w2 = next_img.shape[:2]
    
    # 获取图像的四个焦点
    corners1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    corners2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    
    # 变换第二幅图像的焦点
    warped_corners = cv2.perspectiveTransform(corners2, matrix)
    
    # 计算拼接后图像的大小
    all_corners = np.concatenate((corners1, warped_corners), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 计算平移矩阵
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([
      [1, 0, translation_dist[0]], 
      [0, 1, translation_dist[1]], 
      [0, 0, 1]
    ])
    
    # 应用变换
    warped_img = cv2.warpPerspective(
      next_img, 
      H_translation.dot(matrix), 
      (x_max-x_min, y_max-y_min)
    )
    
    # 创建全景图的扩展版本
    expanded_panorama = np.zeros((y_max-y_min, x_max-x_min, 3), dtype=np.uint8)
    
    # 计算全景图在新坐标系中的位置
    pano_x_start = translation_dist[0]
    pano_y_start = translation_dist[1]
    pano_x_end = pano_x_start + w1
    pano_y_end = pano_y_start + h1
    
    # 确保不超出边界
    pano_x_start = max(0, pano_x_start)
    pano_y_start = max(0, pano_y_start)
    pano_x_end = min(expanded_panorama.shape[1], pano_x_end)
    pano_y_end = min(expanded_panorama.shape[0], pano_y_end)
    
    # 调整全景图尺寸以匹配
    pano_h = pano_y_end - pano_y_start
    pano_w = pano_x_end - pano_x_start
    
    if pano_h > 0 and pano_w > 0:
      # 调整全景图尺寸
      panorama_resized = cv2.resize(panorama, (pano_w, pano_h))
      expanded_panorama[pano_y_start:pano_y_end, pano_x_start:pano_x_end] = panorama_resized
    
    # 混合图像
    overlap_mask = (expanded_panorama > 0).any(axis=2) & (warped_img > 0).any(axis=2)
    warped_only_mask = (warped_img > 0).any(axis=2) & ~overlap_mask

    # 处理重叠区域 - 使用加权平均
    if overlap_mask.any():
      expanded_panorama[overlap_mask] = expanded_panorama[overlap_mask]//2 + warped_img[overlap_mask]//2
    
    # 处理只有新图像有像素的区域 - 直接使用新图像
    if warped_only_mask.any():
      expanded_panorama[warped_only_mask] = warped_img[warped_only_mask]
    
    panorama = expanded_panorama
  
  return panorama