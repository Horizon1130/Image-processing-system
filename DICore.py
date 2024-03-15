# coding=utf-8

import os
from math import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QImage


# numpy矩阵转QImage
def arrayToImage(img):
    if len(img.shape) > 2:  # RGB
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height, depth = result.shape
        result = QImage(result.data, height, width, height * depth, QImage.Format_RGB888)
        return result
    else:  # 灰度
        width, height = img.shape
        result = QImage(img.data, height, width, height, QImage.Format_Grayscale8)
        return result


# 计算直方图
def histograme(image):
    hist = np.array([0] * 256)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            hist[image[x, y]] += 1

    # 绘制柱状图
    plt.figure()
    plt.title("Gray Level Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.bar(np.arange(256), hist, color='blue')  # 第一个参数为柱状图的横坐标,第二个参数为柱状图高度,第三个参数为颜色

    save_path = "__tmp_file_hist.png"
    plt.savefig(save_path)

    result = cv2.imread(save_path)
    os.remove(save_path)
    return result

# 缩放(height_scale_percent为高度的变换比例,width_scale_percent为宽度的变换比例)
def zoom(image, height_scale_percent, width_scale_percent):
    # 获取图像的宽度和高度
    width = int(image.shape[1] * width_scale_percent)
    height = int(image.shape[0] * height_scale_percent)

    if width == 0 or height == 0:
        raise ValueError("Invalid width or height scale. Cannot scale to 0.")

    if len(image.shape) == 2:  # 单通道
        # 缩放图像
        resized_image = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                x = int(i / height * image.shape[0])
                y = int(j / width * image.shape[1])
                resized_image[i, j] = image[x, y]
    else:  # 彩图
        resized_image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    x = int(i / height * image.shape[0])
                    y = int(j / width * image.shape[1])
                    resized_image[i, j, k] = image[x, y, k]

    return resized_image

# 旋转1,旋转出画布
def rotate_image1(image, angle):
    if angle in [0, 360, -360]:
        return image
    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 计算旋转角度的弧度值
    angle_rad = radians(angle)
    rotate = np.array([[cos(angle_rad), -sin(angle_rad)],
                       [sin(angle_rad), cos(angle_rad)]])
    rotate = np.transpose(rotate)

    rotated_image = np.zeros(image.shape, dtype=np.uint8)  # 旋转后的图像
    center = np.array([[height // 2], [width / 2]])
    if len(image.shape) == 2:
        for x in range(height):
            for y in range(width):
                a = np.array([[x], [y]])
                tmp = np.round(rotate @ (a - center) + center)
                if 0 <= tmp[0, 0] < height and 0 <= tmp[1, 0] < width:
                    rotated_image[x, y] = image[int(tmp[0, 0]), int(tmp[1, 0])]
    else:
        for x in range(height):
            for y in range(width):
                for k in range(3):
                    a = np.array([[x], [y]])
                    tmp = np.round(rotate @ (a - center) + center)
                    if 0 <= tmp[0, 0] < height and 0 <= tmp[1, 0] < width:
                        rotated_image[x, y] = image[int(tmp[0, 0]), int(tmp[1, 0])]
    return rotated_image

# 转置
def transpose_image(image):
    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    if len(image.shape) == 2:  # 单通道
        transposed_image = np.zeros((width, height), dtype=np.uint8)
        # 对每个像素进行转置操作
        for i in range(height):
            for j in range(width):
                transposed_image[j, i] = image[i, j]
    else:
        # 创建一个新的空白图像，宽度和高度交换
        transposed_image = np.zeros((width, height, 3), dtype=np.uint8)

        # 对每个像素进行转置操作
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    transposed_image[j, i, k] = image[i, j, k]

    return transposed_image

# 镜像(0为水平,1为垂直)
def mirror(image, side):
    if side == 0:
        # 水平镜像
        horizontal_mirror = image[:, ::-1].copy()  # 第一个冒号表示选择所有行,第二个冒号表示选择所有列
        return horizontal_mirror
    elif side == 1:
        # 垂直镜像
        vertical_mirror = image[::-1, :].copy()
        return vertical_mirror

# 图像灰度反转
def GrayReveral(img):
    Eimg = 255 - img
    Eimg = np.round(Eimg).astype(np.uint8)
    return Eimg

# 图像的对比度调整
def ContrastAdj(img, low_out=0, high_out=1):
    Eimg = img
    low_in = np.min(img)
    high_in = np.max(img)

    if high_in == low_in:
        raise Exception()
    Eimg = (high_out - low_out) / (high_in - low_in) * (Eimg - low_in) + low_out

    Eimg[Eimg > 255] = 255
    Eimg[Eimg < 0] = 0
    Eimg = np.round(Eimg).astype(np.uint8)
    return Eimg


def LOGadj(img, c=1):  # 对数运算
    Eimg = img.astype(np.float32)
    Eimg = np.log(Eimg + 1)
    if Eimg.max() != Eimg.min():
        Eimg = (Eimg - Eimg.min()) / (Eimg.max() - Eimg.min())
    else:
        Eimg[:] = 1
    Eimg *= 255 * c
    Eimg[Eimg > 255] = 255
    Eimg[Eimg < 0] = 0
    Eimg = np.round(Eimg).astype(np.uint8)

    return Eimg

# 图像的幂次变换
def ChangePow(img, c=1, p=1):
    Eimg = img.astype(np.float32)
    Eimg = Eimg ** p
    Eimg *= c
    Eimg[Eimg > 255] = 255
    Eimg[Eimg < 0] = 0
    Eimg = np.round(Eimg).astype(np.uint8)
    return Eimg


# 灰度拉伸
def grayStretch(img, start=1, end=255, sout=1, eout=255):
    # start 为分段区间的起点 end 为分段区间终点
    # sout 为映射区间起点 eout 为映射区间终点
    if (start == end) | (0 == eout):
        raise ZeroDivisionError
    Eimg = img
    H, W = Eimg.shape[:2]
    Rimg = np.zeros(Eimg.shape)
    # 变换函数Y=aX+b计算各个区间的斜率a和数b
    k1 = sout / eout

    k2 = (eout - sout) / (end - start)
    c2 = sout - k2 * start

    k3 = (255 - eout) / (255 - end + np.spacing(1))
    c3 = 255 - k3 * 255

    # 构建映射表
    Slist = [np.int64(x) for x in range(256)]
    for i in range(1, 256):
        if i < start:
            Slist[i] = i * k1
        elif i > end:
            Slist[i] = (i * k3 + c3)
        else:
            Slist[i] = (i * k2 + c2)
    # 查询映射表来实现对比拉伸
    for i in range(0, H):
        for j in range(0, W):
            Rimg[i][j] = Slist[Eimg[i][j]]
    Rimg[Rimg > 255] = 255
    Rimg[Rimg < 0] = 0
    Rimg = np.round(Rimg).astype(np.uint8)

    return Rimg


# 直方图均衡化
def equalizehist(image):
    # 计算直方图(将数组平铺成一维)
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # 计算累积分布函数
    cdf = hist.cumsum()

    cdf_normalized = cdf / cdf.max()  # 进行归一化
    cdf_normalized = cdf_normalized * 256
    cdf_normalized = np.round(cdf_normalized)  # 四舍五入
    cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)

    # 进行直方图均衡化
    equalized_image = np.zeros((image.shape[0], image.shape[1]))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            equalized_image[x, y] = cdf_normalized[image[x, y]]
    equalized_image = equalized_image.astype(np.uint8)

    return equalized_image


# 椒盐噪声(p1为盐的概率,p2为椒的概率)
def add_salt_pepper(image, p1, p2):
    if p1 == 0 and p2 == 0:
        return image

    num_salt = int(np.ceil(p1 * image.size))
    num_pepper = int(np.ceil(p2 * image.size))
    result = image

    if num_salt > 0:
        salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
        if len(image.shape) == 2:
            for x, y in np.nditer([salt_coords[0], salt_coords[1]]):
                result[x, y] = 255
        else:
            for channel in range(image.shape[2]):
                for x, y in np.nditer([salt_coords[0], salt_coords[1]]):
                    result[x, y, channel] = 255

    if num_pepper > 0:
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
        if len(image.shape) == 2:
            for x, y in np.nditer([pepper_coords[0], pepper_coords[1]]):
                result[x, y] = 0
        else:
            for channel in range(image.shape[2]):
                for x, y in np.nditer([pepper_coords[0], pepper_coords[1]]):
                    result[x, y, channel] = 0

    return result


# 高斯噪声(mean为噪声均值,var为噪声标准差)
def add_gaussian(image, mean, stddev):
    new_image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, stddev, size=(image.shape[0], image.shape[1]))
    noisy_image = image
    if len(image.shape) == 2:
        noisy_image = cv2.add(new_image, noise)
    else:
        for channel in range(image.shape[2]):
            noisy_image[:, :, channel] = cv2.add(new_image[:, :, channel], noise)
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = noisy_image * 255
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


# 均匀噪声（min_value为噪声的最小值,max_value为噪声的最大值）
def add_even(image, min_value, max_value):
    new_image = np.array(image / 255, dtype=float)
    # 生成均匀噪声
    noise = np.random.rand(image.shape[0], image.shape[1])
    noise = min_value + (max_value - min_value) * noise
    noisy_image = image
    if len(image.shape) == 2:
        noisy_image = cv2.add(new_image, noise)
    else:
        for channel in range(image.shape[2]):
            noisy_image[:, :, channel] = cv2.add(new_image[:, :, channel], noise)
    noisy_image = noisy_image / noisy_image.max()
    noisy_image = noisy_image * 255
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


# (超限像素)邻域平均法
def neighbourAverage(img, size=3, sigma=0):
    # sigma为是否使用超限像素邻域平均法
    # 0默认不使用,大于0为使用
    H, W = img.shape[:2]
    Eimg = img.copy()
    Rimg = img.copy()
    for i in range(0, H - size + 1):
        for j in range(0, W - size + 1):
            s = (Eimg[i:i + size, j:j + size])
            s[size // 2][size // 2] = 0
            s = np.sum(s)
            s = s / (size * size - 1)
            if fabs(Eimg[i][j] - s) > sigma:
                Rimg[i + (size - 1) // 2][j + (size - 1) // 2] = s
    # 边界填充
    Rimg = Rimg[(size - 1) // 2:H - (size - 1) // 2, (size - 1) // 2:W - (size - 1) // 2]
    Rimg = cv2.copyMakeBorder(Rimg, top=(size - 1) // 2, bottom=(size - 1) // 2, left=(size - 1) // 2,
                              right=(size - 1) // 2, borderType=cv2.BORDER_REFLECT)
    Rimg[Rimg > 255] = 255
    Rimg[Rimg < 0] = 0
    Rimg = np.round(Rimg).astype(np.uint8)
    return Rimg

# 均值滤波法
def averageFilter(img, size=3):
    H, W = img.shape[:2]
    Eimg = img.copy()
    Rimg = img.copy()
    for i in range(0, H - size + 1):
        for j in range(0, W - size + 1):
            s = np.sum(Eimg[i:i + size, j:j + size])
            Rimg[i + (size - 1) // 2][j + (size - 1) // 2] = s / (size * size)
    Rimg = Rimg[(size - 1) // 2:H - (size - 1) // 2, (size - 1) // 2:W - (size - 1) // 2]
    Rimg = cv2.copyMakeBorder(Rimg, top=(size - 1) // 2, bottom=(size - 1) // 2, left=(size - 1) // 2,
                              right=(size - 1) // 2, borderType=cv2.BORDER_REFLECT)
    Rimg[Rimg > 255] = 255
    Rimg[Rimg < 0] = 0
    Rimg = np.round(Rimg).astype(np.uint8)
    return Rimg


def midFilter(img, size=3):  # 中值滤波
    Eimg = img
    H, W = Eimg.shape[:2]
    Rimg = np.zeros((H, W))
    for i in range(0, H - size + 1):
        for j in range(0, W - size + 1):
            s = np.sort(Eimg[i:i + size, j:j + size])
            s = s[size // 2][size // 2]
            Rimg[i + (size - 1) // 2][j + (size - 1) // 2] = s
    Rimg = Rimg[(size - 1) // 2:H - (size - 1) // 2, (size - 1) // 2:W - (size - 1) // 2]
    Rimg = cv2.copyMakeBorder(Rimg, top=(size - 1) // 2, bottom=(size - 1) // 2, left=(size - 1) // 2,
                              right=(size - 1) // 2, borderType=cv2.BORDER_REFLECT)
    Rimg[Rimg > 255] = 255
    Rimg[Rimg < 0] = 0
    Rimg = np.round(Rimg).astype(np.uint8)
    return Rimg

# 自适应中值滤波过程
def AMF(src, i, j, Smin, Smax):
    filter_size = Smin
    kernalsize = filter_size
    window = src[i - kernalsize:i + kernalsize + 1, j - kernalsize:j + kernalsize + 1]  # 窗口矩阵
    Zmin = np.min(window)
    Zmax = np.max(window)
    Zmed = np.median(window)
    Zxy = src[i, j]
    if (Zmed > Zmin) and (Zmed < Zmax):  # 步骤A
        if (Zxy > Zmin) and (Zxy < Zmax):  # 步骤B
            return Zxy
        else:
            return Zmed
    else:  # 增大窗口尺寸
        filter_size = filter_size + 1
        if filter_size <= Smax:  # 未超过则再进行滤波
            return AMF(src, i, j, filter_size, Smax)
        else:  # 超过则返回Zxy
            return Zxy

# 自适应中值滤波
def AdpMedianFilt(img, Smin=3, Smax=9):
    # Smin为初始模板大小
    # Smax为最大模板大小
    bordersize = Smax
    src = cv2.copyMakeBorder(img, bordersize, bordersize, bordersize, bordersize, cv2.BORDER_REFLECT)
    # 寻找每一个像素点
    for i in range(bordersize, src.shape[0] - bordersize):
        for j in range(bordersize, src.shape[1] - bordersize):
            src[i, j] = AMF(src, i, j, Smin, Smax)
    Eimg = src[bordersize:bordersize + img.shape[0], bordersize:bordersize + img.shape[1]]
    Eimg[Eimg > 255] = 255
    Eimg[Eimg < 0] = 0
    Eimg = np.round(Eimg).astype(np.uint8)
    return Eimg


# 一阶微分锐化
# Roberts算子锐化(0为边缘,1为叠加)
def robertsFilter(image, action):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    roberts_x_grad = cv2.filter2D(image, -1, roberts_x)
    roberts_y_grad = cv2.filter2D(image, -1, roberts_y)
    roberts_grad = roberts_x_grad + roberts_y_grad
    if action == 0:
        return roberts_grad
    elif action == 1:
        result = image + roberts_grad
        return result


# Sobel算子锐化
def sobelFilter(image, action):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x_grad = cv2.filter2D(image, -1, sobel_x)
    sobel_y_grad = cv2.filter2D(image, -1, sobel_y)
    sobel_grad = sobel_x_grad + sobel_y_grad
    if action == 0:
        return sobel_grad
    elif action == 1:
        result = sobel_grad + image
        return result


# Prewitt算子锐化
def prewittFilter(image, action):
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x_grad = cv2.filter2D(image, -1, prewitt_x)
    prewitt_y_grad = cv2.filter2D(image, -1, prewitt_y)
    prewitt_grad = prewitt_x_grad + prewitt_y_grad
    if action == 0:
        return prewitt_grad
    elif action == 1:
        result = prewitt_grad + image
        return result


# 二阶微分锐化
# 拉普拉斯
def laplacianFilter(image, size, action):
    def get_kernel(size):
        kernel = np.ones((size, size))
        if action == 0:
            kernel[size // 2, size // 2] = -size ** 2 + 1
            return kernel
        elif action == 1:
            kernel[size // 2, size // 2] = -size ** 2
            kernel = -kernel
            return kernel

    # 定义卷积核
    kernel = get_kernel(size)
    # 进行卷积操作
    result = cv2.filter2D(image, -1, kernel)
    return result


# 阈值分割
def thresholdcut(image, threshold):
    result = image
    result[result >= threshold] = 255
    result[result < threshold] = 0
    return result


# 基本全局阈值分割
def basicGlobalThreshold(image, deltaT):
    T = np.mean(image)
    m1 = np.mean(image[image > T])
    m2 = np.mean(image[image <= T])
    T2 = (m1 + m2) / 2
    while abs(T - T2) > deltaT:
        mm1 = np.mean(image[image > T2])
        mm2 = np.mean(image[image <= T2])
        T = T2
        T2 = (mm1 + mm2) / 2
    result = thresholdcut(image, T2)
    return result


def calc_gray_hist(image):# 计算灰度直方图
    rows, cols = image.shape[:2]
    gray_hist = np.zeros([256], np.uint64)
    for i in range(rows):
        for j in range(cols):
            gray_hist[image[i][j]] += 1
    return gray_hist

def otsuThreshold(img):
    H,W = img.shape[:2]
    gh = calc_gray_hist(img)# 计算直方图
    norm_hist = gh / float(H*W)# 归一化直方图

    P1 = np.zeros([256], dtype=np.float32)# 累计和
    P2 = np.zeros([256], dtype=np.float32)#
    m = np.zeros([256],dtype=np.float32)# 累计均值
    mG = 0 # 全局灰度均值
    for i in range(0,256):
        if i == 0:
            P1[i] = norm_hist[i]
            P2[i] = 1 - P1[i]
            m[i] = 0
            mG = 0
        else:
            P1[i] = norm_hist[i] + P1[i-1]
            P2[i] = 1 - P1[i]
            m[i] = m[i-1] + i*norm_hist[i]
            mG += i*norm_hist[i]
    sitaB2 = np.zeros([256], dtype=np.float32)
    for i in range(0,256):
        if (P1[i] == 0) | (P1[i] == 1):
            sitaB2[i] = 0
        else:
            sitaB2[i] = (mG*P1[i] - m[i])*(mG*P1[i] - m[i]) / (P1[i]*P2[i] + 1e-10)
    k=0
    maxk = np.max(sitaB2)
    for i in range(0,256):
        if sitaB2[i] == maxk:
            k = i
    thresh_img = img.copy()
    thresh_img[thresh_img > k] = 255
    thresh_img[thresh_img <= k] = 0

    thresh_img = thresh_img.astype(np.uint8)
    return thresh_img

def getHsI(img):  # 传入uint8的hsi图像
    H, S, I = cv2.split(img.copy())
    return np.array([H, S, I], dtype='uint8')  # 一次返回H、S、I


def getRGB(img):  # 拆分uint8的BGR图像
    B, G, R = cv2.split(img.copy())
    return np.array([R, G, B], dtype='uint8')


def _R2H(R, G, B):  # 一次仅处理单个像素
    # 归一化至0~1
    R /= 255
    G /= 255
    B /= 255
    eps = 1e-8
    H, S, I = 0, 0, 0
    RGBsum = R + G + B
    Min = min(R, G, B)
    S = 1 - 3 * Min / (RGBsum + eps)
    H = np.arccos((0.5 * (R + R - G - B)) / np.sqrt((R - G) * (R - G) + (R - B) * (G - B) + eps))
    if B > G:
        H = 2 * np.pi - H
    H = H / (2 * np.pi)
    if S == 0:
        H = 0
    I = RGBsum / 3
    return np.array([H, S, I], dtype='float')  # 保存为float类型


def ShowHSI(img):  # 获得uint8分量
    H, S, I = cv2.split(img.copy())
    h = np.round(H * 255).astype(np.uint8)
    s = np.round(S * 255).astype(np.uint8)
    i = np.round(I * 255).astype(np.uint8)
    return np.array([h, s, i], dtype='uint8')


def R2H(img):  # RGB图像转换为HSI图像
    HSI = np.zeros(img.shape, dtype='float')
    wid, hig = img.shape[:2]
    for w in range(wid):
        for h in range(hig):
            HSI[w, h, :] = _R2H(img[w, h, 0], img[w, h, 1], img[w, h, 2])
    H, S, I = ShowHSI(HSI)
    HSI = cv2.merge((H, S, I))
    HSI[HSI > 255] = 255
    return HSI


def _H2R(H, S, I):
    # 归一化
    H *= 2 * np.pi / 255
    S /= 255
    I /= 255
    if H >= 0 and H < 2 * np.pi / 3:
        B = I * (1 - S)
        R = I * (1 + S * np.cos(H) / np.cos(pi / 3 - H))
        G = 3 * I - (R + B)
    elif H >= 2 * np.pi / 3 and H < 4 * np.pi / 3:
        H = H - 2 * np.pi / 3
        R = I * (1 - S)
        G = I * (1 + S * np.cos(H) / np.cos(pi / 3 - H))
        B = 3 * I - (R + G)
    else:
        H = H - 4 * np.pi / 3
        G = I * (1 - S)
        B = I * (1 + S * np.cos(H) / np.cos(pi / 3 - H))
        R = 3 * I - (B + G)
    R *= 255
    G *= 255
    B *= 255
    if R > 255:
        R = 255
    if G > 255:
        G = 255
    if B > 255:
        B = 255
    return np.array([R, G, B], dtype=np.uint8)


def H2R(img):  # HSI图像转换为RGB图像
    rgb = np.zeros(img.shape, dtype=np.uint8)
    wid, hig = img.shape[:2]
    for w in range(wid):
        for h in range(hig):
            rgb[w, h, :] = _H2R(img[w, h, 0], img[w, h, 1], img[w, h, 2])
    return rgb


def HSIhisteq(img):  # 传入RHB图像进行HSI直方图均衡化
    img = R2H(img)
    H, S, I = cv2.split(img)
    i = equalizehist(I)
    i[i > 255] = 255
    hsi = cv2.merge((H, S, i))
    rgb = H2R(hsi)
    rgb[rgb > 255] = 255
    return rgb  # 返回均衡化之后的RGB图像


def RGB2GRAY(img):  # 彩图转灰度图
    rgb = getRGB(img)
    result = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    result = np.round(result).astype(np.uint8)
    return result


# 拍照
def photo():
    # 参数0为电脑的摄像头
    cap = cv2.VideoCapture(0)
    # 创建图像显示窗口
    cv2.namedWindow('photo')
    # 判断是否有拍照
    flag = False

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('photo', frame)  # 显示当前帧图像
            action = cv2.waitKey(1)
            if action == 13:  # 按下回车键开始p
                flag = True
                break
            if action == ord('q'):  # 按下 'q' 键结束录制
                frame = None
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return flag, frame
