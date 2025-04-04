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
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">全景图拼接</td>
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
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">2025年04月04日</td>
      </tr>
    </tbody>              
    </table>
</div>
<!-- 注释语句：导出PDF时会在这里分页，使用 Typora Newsprint 主题放大 125% -->

## 实验目的

1. 熟悉 Harris 角点检测器的原理和基本使用
2. 熟悉 RANSAC 抽样一致方法的使用场景
3. 熟悉 HOG 描述子的基本原理
4. 实现基于特征点的图像拼接流程

## 实验原理

### Harris角点检测

Harris角点检测基于图像的自相关函数，通过计算图像局部区域的灰度变化来检测角点。其数学原理如下：

1. 计算图像在x和y方向的梯度Ix和Iy
2. 计算梯度的乘积矩阵M：
   ```
   M = ∑[Ix² IxIy
        IxIy Iy²]
   ```
3. 计算角点响应函数R：
   ```
   R = det(M) - k·trace(M)²
   ```
   其中k为经验常数(0.04-0.06)

4. 通过阈值和非极大值抑制筛选角点

### HOG特征描述子

HOG(Histogram of Oriented Gradients)通过统计局部区域内的梯度方向直方图来描述特征：

1. 将图像划分为小的细胞单元(cell)
2. 计算每个cell内像素的梯度方向和大小
3. 将梯度方向量化为9个bin(0-180°)
4. 将相邻的cell组合成块(block)，进行归一化处理

### SIFT特征描述子

SIFT(Scale-Invariant Feature Transform)通过构建尺度空间和关键点方向来实现尺度不变性：

1. 构建高斯金字塔和DOG金字塔
2. 检测极值点作为候选关键点
3. 为关键点分配主方向
4. 生成128维的特征描述向量

### RANSAC算法

RANSAC的核心思想是通过反复随机抽样来估计数学模型参数。对于图像拼接任务，我需要估计的是单应性矩阵H（3×3的透视变换矩阵）。算法流程如下：

1. 随机从匹配点对中选取4个样本点（求解单应性矩阵的最小样本数）
2. 用这4个点计算初始的单应性矩阵H
3. 用H测试所有其他匹配点，计算重投影误差
4. 统计误差小于阈值的点（内点）数量
5. 重复上述过程若干次，保留内点最多的模型
6. 用所有内点重新估计最优的单应性矩阵

这种方法的优势在于能够有效剔除错误的匹配点（外点），即使初始匹配中有大量错误匹配，仍能获得正确的变换关系。

## 实验步骤

### 1. Harris角点检测实现

关键代码(`harris.py`)：
```python
def harris_detection(img, window_size=3, k=0.04, threshold=0.01):
    # 计算图像梯度
    Ix = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=window_size)
    Iy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=window_size)
    
    # 计算梯度的乘积
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # 高斯滤波
    Sx2 = cv2.filter2D(Ix2, -1, np.ones((window_size, window_size)))
    Sy2 = cv2.filter2D(Iy2, -1, np.ones((window_size, window_size)))
    Sxy = cv2.filter2D(Ixy, -1, np.ones((window_size, window_size)))
    
    # 计算焦点响应函数R
    det = (Sx2 * Sy2) - (Sxy ** 2)
    trace = Sx2 + Sy2
    R = det - k * (trace ** 2)
    
    # 非极大值抑制和阈值处理
    keypoints = []
    R_max = np.max(R)
    for y in range(height):
        for x in range(width):
            if R[y, x] > threshold * R_max:
                # 检查是否是局部最大值
                is_local_max = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if R[y+dy, x+dx] > R[y, x]:
                            is_local_max = False
                            break
                if is_local_max:
                    keypoints.append(cv2.KeyPoint(x, y, window_size))
```

### 2. 特征描述与匹配

首先是 SIFT 的实现，我首先用 Harris 检测到的关键点，然后在每个关键点周围计算梯度方向直方图。代码中，我使用OpenCV的SIFT实现：

```python
sift = cv2.SIFT.create()  # 创建SIFT检测器
_, des1 = sift.compute(img1, kp1)  # 计算图像1的描述子
_, des2 = sift.compute(img2, kp2)  # 计算图像2的描述子
bf = cv2.BFMatcher(cv2.NORM_L2)  # 使用欧氏距离进行匹配
matches = bf.match(des1, des2)  # 获取匹配结果
```

SIFT会为每个关键点生成一个128维的特征向量，描述关键点周围区域的梯度分布。

相比之下，HOG（方向梯度直方图）则采用了不同的思路。它不关注特定的关键点，而是统计图像局部区域的梯度方向分布。我的实现是对每个Harris关键点周围的32×32像素区域计算HOG特征：

```python
hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)  # 定义HOG参数
des1 = [hog.compute(get_patch(img1, kp)) for kp in kp1]  # 计算图像1的描述子
des2 = [hog.compute(get_patch(img2, kp)) for kp in kp2]  # 计算图像2的描述子
bf = cv2.BFMatcher(cv2.NORM_L2)  # 同样使用欧氏距离
matches = bf.match(np.squeeze(des1), np.squeeze(des2))  # 获取匹配
```

HOG将图像分成了多个小的细胞单元(cell)，在每个cell内计算梯度方向的直方图，然后将这些直方图连接起来形成最终的特征向量。这种方法对物体的局部形状描述很有效，特别是在有重复纹理的区域，比如建筑物的窗户墙面，HOG能找到很多SIFT可能忽略的匹配点。

### 3. RANSAC图像拼接(`ransac.py`)

关键步骤：
1. 检测两幅图像的特征点
2. 计算特征描述子并进行匹配
3. 使用RANSAC估计单应性矩阵
4. 计算拼接后图像大小并进行透视变换
5. 混合两幅图像

```python
def ransac_stitch(images, HOG=False):
    panorama = images[0]
    for next_img in images[1:]:
        # 特征检测与匹配
        kp1 = harris_detection(panorama)
        kp2 = harris_detection(next_img)
        matches = hog_match(panorama, next_img, kp1, kp2) if HOG else sift_match(panorama, next_img, kp1, kp2)
        
        # RANSAC计算单应性矩阵
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 计算拼接后图像大小
        h1, w1 = panorama.shape[:2]
        h2, w2 = next_img.shape[:2]
        corners = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, matrix)
        
        # 创建拼接图像
        all_corners = np.concatenate((np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2), warped_corners))
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # 应用变换并混合图像
        translation = np.array([[1,0,-x_min], [0,1,-y_min], [0,0,1]])
        warped_img = cv2.warpPerspective(next_img, translation.dot(matrix), (x_max-x_min, y_max-y_min))
        
        expanded_panorama = np.zeros((y_max-y_min, x_max-x_min, 3), dtype=np.uint8)
        expanded_panorama[-y_min:h1-y_min, -x_min:w1-x_min] = panorama
        
        # 图像混合
        overlap = (expanded_panorama > 0) & (warped_img > 0)
        expanded_panorama[overlap] = expanded_panorama[overlap]//2 + warped_img[overlap]//2
        expanded_panorama[warped_img > 0 & ~overlap] = warped_img[warped_img > 0 & ~overlap]
        
        panorama = expanded_panorama
    return panorama
```

## 实验结果与分析

### 角点检测效果

对`sudoku.png`进行Harris角点检测，结果如图所示：

<img src="results/sudoku_keypoints.png" alt="sudoku_keypoints" style="zoom:72%;" />

对`uttower1.jpg`进行Harris角点检测，结果如图所示：

<img src="results/uttower1_keypoints.png" alt="uttower1_keypoints" style="zoom:72%;" />

对`uttower2.jpg`进行Harris角点检测，结果如图所示：

<img src="results/uttower2_keypoints.png" alt="uttower2_keypoints" style="zoom:72%;" />

### 特征匹配效果

对`uttower1.jpg`和`uttower2.jpg`进行HOG特征匹配结果如下：

![uttower_match_hog](results/uttower_match_hog.png)

对`uttower1.jpg`和`uttower2.jpg`进行SIFT特征匹配结果如下：

![uttower_match_sift](results/uttower_match_sift.png)

从实际匹配效果来看，SIFT的匹配准确率更高，这是因为SIFT的描述子包含了更丰富的空间结构信息。但是HOG的计算速度更快，在处理大图像时有优势。另外我还注意到，当图像有旋转时，SIFT仍然能保持良好的匹配性能，而HOG的匹配质量会明显下降，这是因为HOG没有做方向归一化的处理。

在光照变化方面，两种方法都表现不错，但HOG对强烈的阴影变化适应性稍好一些。这是因为HOG只关心梯度方向，不依赖梯度幅值。而在尺度变化方面，SIFT明显优于HOG，这是因为它本来就是为解决尺度问题而设计的。

### 拼接效果展示

HOG 匹配拼接 uttower 如下：

![uttower_stitching_hog](results/uttower_stitching_hog.png)

SIFT 匹配拼接 uttower 如下：

![uttower_stitching_sift](results/uttower_stitching_sift.png)

SIFT 匹配拼接 yosemite 四张图片如下：

![yosemite_stitching](results/yosemite_stitching.png)

## 实验总结

Harris角点检测在纹理丰富的区域效果良好，但在低纹理区域可能检测不到足够特征点。

SIFT 特征计算复杂度更高，在图像匹配中表现优于 HOG 特征，主要体现在：_
- 对旋转和尺度变化具有不变性
- 匹配准确度更高

RANSAC 算法能有效去除错误匹配点，提高单应性矩阵估计的准确性。自然的图像融合也很重要，在我的实现中是重叠区域各取一半透明度融合。

本实验完整实现了基于Harris角点检测、SIFT/HOG特征描述和RANSAC算法的全景图拼接流程，成功实现了全景图拼接。