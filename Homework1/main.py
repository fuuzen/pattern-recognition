from harris import harris_corner_detection
from arg import parse
import cv2

def main():
  args = parse()
  # 读取图像并转换为灰度图
  image = cv2.imread(args.input)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # 执行Harris角点检测
  corners = harris_corner_detection(gray, window_size=3, k=0.04, threshold=0.01)

  # 在图像上绘制角点
  for (x, y) in corners:
    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

  cv2.imwrite(args.output, image)
  
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main())