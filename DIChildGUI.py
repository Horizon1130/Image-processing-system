# coding=utf-8

from DICore import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class HistWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle('直方图')
        main_layout = QHBoxLayout(self)

        # 加载组件
        self.original_label = QLabel()
        save_basic = QPushButton('保存原图像直方图')
        save_basic.clicked.connect(lambda: self.saveFig(self.original_label))
        save_basic.setCursor(Qt.PointingHandCursor)

        original_box = QGroupBox('原图像直方图')
        original_layout = QVBoxLayout(original_box)
        original_layout.addWidget(self.original_label)
        original_layout.addWidget(save_basic)

        self.processed_label = QLabel()
        save_current = QPushButton('保存处理后图像直方图')
        save_current.clicked.connect(lambda: self.saveFig(self.processed_label))
        save_current.setCursor(Qt.PointingHandCursor)

        processed_box = QGroupBox('处理后图像直方图')
        processed_layout = QVBoxLayout(processed_box)
        processed_layout.addWidget(self.processed_label)
        processed_layout.addWidget(save_current)

        main_layout.addWidget(original_box)
        main_layout.addWidget(processed_box)

    def loadHist(self):
        # 获取直方图
        original = arrayToImage(histograme(self.main_window.basic_img))
        processed = arrayToImage(histograme(self.main_window.current_img))
        # 设置显示的图像
        self.original_label.setPixmap(QPixmap.fromImage(original))
        self.processed_label.setPixmap(QPixmap.fromImage(processed))

    def saveFig(self, label):
        path, _ = QFileDialog.getSaveFileName(self, '保存为', './untitled', 'JPEG(*.jpg;*.jpeg);;PNG(*.png);;BMP(*.bmp);;TIFF(*.tif;*.tiff)')
        if len(path) > 0:
            try:
                label.pixmap().save(path)
            except:
                QMessageBox.critical(self, '错误', '保存失败')


class BaseColorPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        main_layout = QHBoxLayout(self)  # 获取主布局

        # 镜像
        box = QGroupBox('镜像')
        main_layout.addWidget(box)
        layout = QVBoxLayout(box)

        button = QPushButton('水平镜像')
        button.clicked.connect(lambda: main_window.commonDeal(mirror, 0))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('垂直镜像')
        button.clicked.connect(lambda: main_window.commonDeal(mirror, 1))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        # 转置
        box = QGroupBox('转置')
        main_layout.addWidget(box)
        layout = QVBoxLayout(box)

        button = QPushButton('图像转置')
        button.clicked.connect(lambda: main_window.commonDeal(transpose_image))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        # 缩放
        box = QGroupBox('缩放')
        main_layout.addWidget(box)
        layout = QVBoxLayout(box)

        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.01, 5)
        width_spin.setValue(1)
        width_spin.setSingleStep(0.01)
        width_spin.setPrefix('宽缩放倍数: ')

        height_spin = QDoubleSpinBox()
        height_spin.setRange(0.01, 5)
        height_spin.setValue(1)
        height_spin.setSingleStep(0.01)
        height_spin.setPrefix('高缩放倍数: ')

        button = QPushButton('确定缩放')
        button.clicked.connect(lambda: main_window.commonDeal(zoom, height_spin.value(), width_spin.value()))
        button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(width_spin)
        layout.addWidget(height_spin)
        layout.addWidget(button)

        # 旋转
        box = QGroupBox('旋转')
        main_layout.addWidget(box)
        layout = QVBoxLayout(box)

        rotate_spin = QDoubleSpinBox()
        rotate_spin.setRange(-360, 360)
        rotate_spin.setSingleStep(1)
        rotate_spin.setValue(0)
        rotate_spin.setPrefix('旋转角度(顺时针): ')

        button = QPushButton('旋转')
        button.clicked.connect(lambda: main_window.commonDeal(rotate_image1, -rotate_spin.value()))
        button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(rotate_spin)
        layout.addWidget(button)


class BaseGrayPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        main_layout = QGridLayout(self)  # 获取主布局

        # 镜像
        box = QGroupBox('镜像')
        main_layout.addWidget(box, 0, 0, 2, 1)
        layout = QVBoxLayout(box)

        button = QPushButton('水平镜像')
        button.clicked.connect(lambda: main_window.commonDeal(mirror, 0))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('垂直镜像')
        button.clicked.connect(lambda: main_window.commonDeal(mirror, 1))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        # 转置
        box = QGroupBox('转置')
        main_layout.addWidget(box, 0, 1)
        layout = QVBoxLayout(box)

        button = QPushButton('图像转置')
        button.clicked.connect(lambda: main_window.commonDeal(transpose_image))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        # 旋转
        box = QGroupBox('旋转')
        main_layout.addWidget(box, 1, 1)
        layout = QVBoxLayout(box)

        rotate_spin = QDoubleSpinBox()
        rotate_spin.setRange(-360, 360)
        rotate_spin.setSingleStep(1)
        rotate_spin.setValue(0)
        rotate_spin.setPrefix('旋转角度(顺时针): ')

        button = QPushButton('旋转')
        button.clicked.connect(lambda: main_window.commonDeal(rotate_image1, -rotate_spin.value()))
        button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(rotate_spin)
        layout.addWidget(button)

        # 缩放
        box = QGroupBox('缩放')
        main_layout.addWidget(box, 0, 2, 2, 1)
        layout = QVBoxLayout(box)

        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.01, 5)
        width_spin.setValue(1)
        width_spin.setSingleStep(0.01)
        width_spin.setPrefix('宽缩放倍数: ')

        height_spin = QDoubleSpinBox()
        height_spin.setRange(0.01, 5)
        height_spin.setValue(1)
        height_spin.setSingleStep(0.01)
        height_spin.setPrefix('高缩放倍数: ')

        button = QPushButton('确定缩放')
        button.clicked.connect(lambda: main_window.commonDeal(zoom, height_spin.value(), width_spin.value()))
        button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(width_spin)
        layout.addWidget(height_spin)
        layout.addWidget(button)

        # 对数变换
        box = QGroupBox('对数变换')
        main_layout.addWidget(box, 0, 3, 2, 1)
        layout = QVBoxLayout(box)

        log_spin = QDoubleSpinBox()
        log_spin.setRange(0, 100)
        log_spin.setValue(1)
        log_spin.setSingleStep(0.1)
        log_spin.setPrefix('系数: ')

        button = QPushButton('变换')
        button.clicked.connect(lambda: main_window.commonDeal(LOGadj, log_spin.value()))
        button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(log_spin)
        layout.addWidget(button)

        # 幂次变换
        box = QGroupBox('幂次变换')
        main_layout.addWidget(box, 0, 4, 2, 1)
        layout = QVBoxLayout(box)

        pow_p_spin = QDoubleSpinBox()
        pow_p_spin.setRange(0, 100)
        pow_p_spin.setValue(1)
        pow_p_spin.setSingleStep(0.1)
        pow_p_spin.setPrefix('幂次: ')

        pow_c_spin = QDoubleSpinBox()
        pow_c_spin.setRange(0, 100)
        pow_c_spin.setValue(1)
        pow_c_spin.setSingleStep(0.1)
        pow_c_spin.setPrefix('系数: ')

        button = QPushButton('变换')
        button.clicked.connect(lambda: main_window.commonDeal(ChangePow, pow_c_spin.value(), pow_p_spin.value()))
        button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(pow_p_spin)
        layout.addWidget(pow_c_spin)
        layout.addWidget(button)

        # 灰度反转
        box = QGroupBox('灰度反转')
        main_layout.addWidget(box, 0, 5)
        layout = QVBoxLayout(box)

        button = QPushButton('反转')
        button.clicked.connect(lambda: main_window.commonDeal(GrayReveral))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        # 对比度调整
        box = QGroupBox('对比度调整')
        main_layout.addWidget(box, 1, 5)
        layout = QVBoxLayout(box)

        low_spin = QSpinBox()
        low_spin.setRange(0, 255)
        low_spin.setPrefix('调整后最低灰度: ')
        high_spin = QSpinBox()
        high_spin.setRange(0, 255)
        high_spin.setPrefix('调整后最高灰度: ')
        button = QPushButton('调整')
        button.clicked.connect(lambda: main_window.commonDeal(ContrastAdj, low_spin.value(), high_spin.value()))
        button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(low_spin)
        layout.addWidget(high_spin)
        layout.addWidget(button)


class GrayStretchPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        main_layout = QHBoxLayout(self)  # 获取主布局

        # 原始区
        x_box = QGroupBox()
        main_layout.addWidget(x_box)
        x_layout = QGridLayout(x_box)
        # 参数-start
        self.start_spin = QSpinBox()
        self.start_spin.setPrefix('分段区间起点:')
        self.start_spin.setRange(0, 254)  # 设置范围
        self.start_spin.setValue(0)
        self.last_start = 0

        self.start_slider = QSlider()
        self.start_slider.setOrientation(Qt.Horizontal)
        self.start_slider.setRange(0, 254)
        self.start_slider.setValue(0)

        x_layout.addWidget(self.start_spin, 0, 0)
        x_layout.addWidget(self.start_slider, 0, 1)
        # 参数-end
        self.end_spin = QSpinBox()
        self.end_spin.setPrefix('分段区间终点:')
        self.end_spin.setRange(1, 255)
        self.end_spin.setValue(255)
        self.last_end = 255

        self.end_slider = QSlider()
        self.end_slider.setOrientation(Qt.Horizontal)
        self.end_slider.setRange(1, 255)
        self.end_slider.setValue(255)

        x_layout.addWidget(self.end_spin, 1, 0)
        x_layout.addWidget(self.end_slider, 1, 1)
        # 设置触发器
        self.start_spin.valueChanged.connect(self.start_slider.setValue)
        self.start_slider.valueChanged.connect(self.start_spin.setValue)
        self.end_spin.valueChanged.connect(self.end_slider.setValue)
        self.end_slider.valueChanged.connect(self.end_spin.setValue)
        self.start_spin.valueChanged.connect(self.adjustXValue)
        self.start_slider.valueChanged.connect(self.adjustXValue)
        self.end_spin.valueChanged.connect(self.adjustXValue)
        self.end_slider.valueChanged.connect(self.adjustXValue)

        # 映射区
        y_box = QGroupBox()
        main_layout.addWidget(y_box)
        y_layout = QGridLayout(y_box)
        # 参数-start
        self.sout_spin = QSpinBox()
        self.sout_spin.setPrefix('映射区间起点:')
        self.sout_spin.setRange(0, 255)  # 设置范围
        self.sout_spin.setValue(0)
        self.last_sout = 0

        self.sout_slider = QSlider()
        self.sout_slider.setOrientation(Qt.Horizontal)
        self.sout_slider.setRange(0, 255)
        self.sout_slider.setValue(0)

        y_layout.addWidget(self.sout_spin, 0, 0)
        y_layout.addWidget(self.sout_slider, 0, 1)
        # 参数-end
        self.eout_spin = QSpinBox()
        self.eout_spin.setPrefix('映射区间终点:')
        self.eout_spin.setRange(1, 255)
        self.eout_spin.setValue(255)
        self.last_eout = 255

        self.eout_slider = QSlider()
        self.eout_slider.setOrientation(Qt.Horizontal)
        self.eout_slider.setRange(1, 255)
        self.eout_slider.setValue(255)

        y_layout.addWidget(self.eout_spin, 1, 0)
        y_layout.addWidget(self.eout_slider, 1, 1)
        # 设置触发器
        self.sout_spin.valueChanged.connect(self.sout_slider.setValue)
        self.sout_slider.valueChanged.connect(self.sout_spin.setValue)
        self.eout_spin.valueChanged.connect(self.eout_slider.setValue)
        self.eout_slider.valueChanged.connect(self.eout_spin.setValue)
        self.sout_spin.valueChanged.connect(self.adjustYValue)
        self.sout_slider.valueChanged.connect(self.adjustYValue)
        self.eout_spin.valueChanged.connect(self.adjustYValue)
        self.eout_slider.valueChanged.connect(self.adjustYValue)

        # 处理按钮
        deal_button = QPushButton('进行拉伸')
        main_layout.addWidget(deal_button)
        deal_button.setCursor(Qt.PointingHandCursor)
        deal_button.clicked.connect(lambda: main_window.commonDeal(grayStretch, self.start_spin.value(), self.end_spin.value(), self.sout_spin.value(), self.eout_spin.value()))

    def adjustXValue(self):
        current_start = self.start_spin.value()
        current_end = self.end_spin.value()
        if current_start >= current_end and self.last_start < current_start:
            self.end_spin.setValue(current_start + 1)
            self.end_slider.setValue(current_start + 1)
        elif current_end <= current_start and self.last_end > current_end:
            self.start_spin.setValue(current_end - 1)
            self.start_slider.setValue(current_end - 1)
        self.last_start = self.start_spin.value()
        self.last_end = self.end_spin.value()

    def adjustYValue(self):
        current_sout = self.sout_spin.value()
        current_eout = self.eout_spin.value()
        if current_sout > current_eout and self.last_sout < current_sout:
            self.eout_spin.setValue(current_sout)
            self.eout_slider.setValue(current_sout)
        elif current_eout < current_sout and self.last_eout > current_eout:
            self.sout_spin.setValue(current_eout)
            self.sout_slider.setValue(current_eout)
        self.last_sout = self.sout_spin.value()
        self.last_eout = self.eout_spin.value()


class NoisePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        main_layout = QHBoxLayout(self)  # 获取主布局

        # 椒盐噪声
        salt_pepper_box = QGroupBox('椒盐噪声')
        main_layout.addWidget(salt_pepper_box)
        layout = QGridLayout(salt_pepper_box)

        # 两个参数框
        self.salt_spin = QDoubleSpinBox()  # 盐
        self.salt_spin.setRange(0, 1)
        self.salt_spin.setSingleStep(0.01)
        self.salt_spin.setPrefix('盐的概率: ')
        self.salt_spin.setToolTip('两概率相加无法大于1')
        self.salt_spin.valueChanged.connect(self.salt_update)

        self.pepper_spin = QDoubleSpinBox()  # 椒
        self.pepper_spin.setRange(0, 1)
        self.pepper_spin.setSingleStep(0.01)
        self.pepper_spin.setToolTip('两概率相加无法大于1')
        self.pepper_spin.setPrefix('椒的概率: ')
        self.pepper_spin.valueChanged.connect(self.pepper_update)
        # 触发按钮
        button = QPushButton('点击添加')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(add_salt_pepper, self.salt_spin.value(), self.pepper_spin.value()))

        layout.addWidget(self.salt_spin, 0, 0)
        layout.addWidget(self.pepper_spin, 0, 1)
        layout.addWidget(button, 1, 0, 1, 2)

        # 高斯噪声
        gaussian_box = QGroupBox('高斯噪声')
        main_layout.addWidget(gaussian_box)
        layout = QGridLayout(gaussian_box)

        # 两个参数框
        mean_spin = QDoubleSpinBox()  # 均值
        mean_spin.setRange(0, 10)
        mean_spin.setSingleStep(0.01)
        mean_spin.setPrefix('噪声均值: ')

        stddev_spin = QDoubleSpinBox()  # 标准差
        stddev_spin.setRange(0, 10)
        stddev_spin.setSingleStep(0.01)
        stddev_spin.setPrefix('噪声标准差: ')
        # 触发按钮
        button = QPushButton('点击添加')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(add_gaussian, mean_spin.value(), stddev_spin.value()))

        layout.addWidget(mean_spin, 0, 0)
        layout.addWidget(stddev_spin, 0, 1)
        layout.addWidget(button, 1, 0, 1, 2)

        # 均匀噪声
        even_box = QGroupBox('均匀噪声')
        main_layout.addWidget(even_box)
        layout = QGridLayout(even_box)

        # 两个参数框
        self.min_value_spin = QDoubleSpinBox()
        self.min_value_spin.setRange(0, 10)
        self.min_value_spin.setSingleStep(0.01)
        self.min_value_spin.setPrefix('噪声的最小值: ')
        self.min_value_spin.valueChanged.connect(self.min_value_update)

        self.max_value_spin = QDoubleSpinBox()
        self.max_value_spin.setRange(0, 10)
        self.max_value_spin.setSingleStep(0.01)
        self.max_value_spin.setPrefix('噪声的最大值: ')
        self.max_value_spin.valueChanged.connect(self.max_value_update)
        # 触发按钮
        button = QPushButton('点击添加')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(add_even, self.min_value_spin.value(), self.max_value_spin.value()))

        layout.addWidget(self.min_value_spin, 0, 0)
        layout.addWidget(self.max_value_spin, 0, 1)
        layout.addWidget(button, 1, 0, 1, 2)

    def salt_update(self):
        salt = self.salt_spin.value()
        pepper = self.pepper_spin.value()
        if salt + pepper > 1:
            self.pepper_spin.setValue(1 - salt)

    def pepper_update(self):
        salt = self.salt_spin.value()
        pepper = self.pepper_spin.value()
        if salt + pepper > 1:
            self.salt_spin.setValue(1 - pepper)

    def min_value_update(self):
        if self.min_value_spin.value() > self.max_value_spin.value():
            self.max_value_spin.setValue(self.min_value_spin.value())

    def max_value_update(self):
        if self.max_value_spin.value() < self.min_value_spin.value():
            self.min_value_spin.setValue(self.max_value_spin.value())


class SmoothPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        main_layout = QHBoxLayout(self)  # 获取主布局

        # 基本滤波
        basic_box = QGroupBox()
        main_layout.addWidget(basic_box)
        layout = QHBoxLayout(basic_box)

        # 模板尺寸
        self.size_spin = QSpinBox()
        layout.addWidget(self.size_spin)
        self.size_spin.setPrefix('滤波模板尺寸: ')
        self.size_spin.setRange(3, 5555)
        self.size_spin.setValue(3)
        self.size_spin.setSingleStep(2)
        self.size_spin.valueChanged.connect(self.size_update)

        # 功能
        v_layout = QVBoxLayout()
        layout.addLayout(v_layout)

        button = QPushButton('邻域平均滤波')
        button.setCursor(Qt.PointingHandCursor)
        sigma_spin = QDoubleSpinBox()
        sigma_spin.setRange(0, 255)
        sigma_spin.setSingleStep(0.1)
        sigma_spin.setPrefix('误差门限: ')

        button.clicked.connect(lambda: main_window.commonDeal(neighbourAverage, self.size_spin.value(), sigma_spin.value()))
        tmp_layout = QHBoxLayout()
        tmp_layout.addWidget(button)
        tmp_layout.addWidget(sigma_spin)
        v_layout.addLayout(tmp_layout)

        button = QPushButton('均值滤波')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(averageFilter, self.size_spin.value()))
        v_layout.addWidget(button)

        button = QPushButton('中值滤波')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(midFilter, self.size_spin.value()))
        v_layout.addWidget(button)

        # 自适应滤波
        adapt_box = QGroupBox()
        main_layout.addWidget(adapt_box)
        layout = QVBoxLayout(adapt_box)

        self.Smin_spin = QSpinBox()
        self.Smin_spin.setRange(3, 5555)
        self.Smin_spin.setValue(3)
        self.Smin_spin.setSingleStep(2)
        self.Smin_spin.setPrefix('初始模板尺寸: ')

        self.Smax_spin = QSpinBox()
        self.Smax_spin.setRange(3, 5555)
        self.Smax_spin.setValue(5)
        self.Smax_spin.setSingleStep(2)
        self.Smax_spin.setPrefix('最大模板尺寸: ')

        button = QPushButton('自适应中值滤波')
        button.setCursor(Qt.PointingHandCursor)

        self.Smin_spin.valueChanged.connect(self.Smin_update)
        self.Smax_spin.valueChanged.connect(self.Smax_update)
        button.clicked.connect(lambda: main_window.commonDeal(AdpMedianFilt, self.Smin_spin.value(), self.Smax_spin.value()))
        layout.addWidget(self.Smin_spin)
        layout.addWidget(self.Smax_spin)
        layout.addWidget(button)

    def size_update(self):
        size = self.size_spin.value()
        if size % 2 == 0:
            self.size_spin.setValue(size + 1)

    def Smin_update(self):
        Smin = self.Smin_spin.value()
        Smax = self.Smax_spin.value()
        if Smin % 2 == 0:
            Smin += 1
            self.Smin_spin.setValue(Smin)
        if Smin > Smax:
            self.Smax_spin.setValue(Smin)

    def Smax_update(self):
        Smin = self.Smin_spin.value()
        Smax = self.Smax_spin.value()
        if Smax % 2 == 0:
            Smax += 1
            self.Smax_spin.setValue(Smax)
        if Smax < Smin:
            self.Smin_spin.setValue(Smax)


class SharpenPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        main_layout = QHBoxLayout(self)  # 获取主布局

        # 一阶
        first_box = QGroupBox()
        main_layout.addWidget(first_box)
        layout = QGridLayout(first_box)

        button = QPushButton('Roberts算子锐化-边缘图像')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(robertsFilter, 0))
        layout.addWidget(button, 0, 0)

        button = QPushButton('Roberts算子锐化-叠加图像')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(robertsFilter, 1))
        layout.addWidget(button, 0, 1)

        button = QPushButton('Sobel算子锐化-边缘图像')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(sobelFilter, 0))
        layout.addWidget(button, 1, 0)

        button = QPushButton('Sobel算子锐化-叠加图像')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(sobelFilter, 1))
        layout.addWidget(button, 1, 1)

        button = QPushButton('Prewitt算子锐化-边缘图像')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(prewittFilter, 0))
        layout.addWidget(button, 2, 0)

        button = QPushButton('Prewitt算子锐化-叠加图像')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(prewittFilter, 1))
        layout.addWidget(button, 2, 1)

        # 拉普拉斯
        second_box = QGroupBox()
        main_layout.addWidget(second_box)
        layout = QVBoxLayout(second_box)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(3, 5555)
        self.size_spin.setValue(3)
        self.size_spin.setSingleStep(2)
        self.size_spin.setPrefix('模板尺寸: ')
        self.size_spin.valueChanged.connect(self.size_update)
        button1 = QPushButton('拉普拉斯算子锐化-边缘图像')
        button1.setCursor(Qt.PointingHandCursor)
        button1.clicked.connect(lambda: main_window.commonDeal(laplacianFilter, self.size_spin.value(), 0))
        button2 = QPushButton('拉普拉斯算子锐化-叠加图像')
        button2.setCursor(Qt.PointingHandCursor)
        button2.clicked.connect(lambda: main_window.commonDeal(laplacianFilter, self.size_spin.value(), 1))
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(self.size_spin)

    def size_update(self):
        size = self.size_spin.value()
        if size % 2 == 0:
            self.size_spin.setValue(size + 1)


class SegmentationPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        main_layout = QHBoxLayout(self)

        # 阈值分割
        box = QGroupBox('阈值分割')
        main_layout.addWidget(box)
        layout = QVBoxLayout(box)

        spin1 = QSpinBox()
        spin1.setRange(0, 255)
        spin1.setPrefix('阈值: ')
        button = QPushButton('分割')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(thresholdcut, spin1.value()))
        layout.addWidget(spin1)
        layout.addWidget(button)

        # 基本全局阈值法
        box = QGroupBox('基本全局阈值分割')
        main_layout.addWidget(box)
        layout = QVBoxLayout(box)

        spin2 = QDoubleSpinBox()
        spin2.setRange(0, 255)
        spin2.setSingleStep(0.1)
        spin2.setPrefix('ΔT: ')
        button = QPushButton('分割')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(basicGlobalThreshold, spin2.value()))
        layout.addWidget(spin2)
        layout.addWidget(button)

        # Otsu
        box = QGroupBox('Otsu方法')
        main_layout.addWidget(box)
        layout = QVBoxLayout(box)

        button = QPushButton('分割')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(otsuThreshold))
        layout.addWidget(button)


class HSIPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.split_window = SplitWindow(main_window)
        main_layout = QHBoxLayout(self)  # 获取主布局

        # 转换
        tran_box = QGroupBox('转换')
        main_layout.addWidget(tran_box)
        layout = QVBoxLayout(tran_box)

        button = QPushButton('由RGB转换为HSI')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(R2H))
        layout.addWidget(button)

        button = QPushButton('由HSI转换为RGB')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: main_window.commonDeal(H2R))
        layout.addWidget(button)

        # 分量
        split_box = QGroupBox('分量')
        main_layout.addWidget(split_box)
        layout = QVBoxLayout(split_box)

        button = QPushButton('显示当前图像的R、G、B分量')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(self.show_RGB)
        layout.addWidget(button)

        button = QPushButton('显示当前图像的H、S、I分量')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(self.show_HSI)
        layout.addWidget(button)

    def show_RGB(self):
        try:
            self.split_window.loadRGB()
            self.split_window.show()
        except:
            QMessageBox.critical(self.main_window, '错误', '加载失败')

    def show_HSI(self):
        try:
            self.split_window.loadHSI()
            self.split_window.show()
        except:
            QMessageBox.critical(self.main_window, '错误', ' 加载失败')


class SplitWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.setMinimumSize(1, 1)
        self.main_window = main_window
        main_layout = QHBoxLayout(self)

        # 加载组件
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)

        self.label1 = QLabel()
        self.label1.setSizePolicy(sizePolicy)
        self.image1 = QImage()
        self.button1 = QPushButton()
        self.button1.setCursor(Qt.PointingHandCursor)
        self.button1.clicked.connect(lambda: self.saveFig(self.image1))
        self.box1 = QGroupBox()
        layout = QVBoxLayout(self.box1)
        layout.addWidget(self.label1)
        layout.addWidget(self.button1)

        self.label2 = QLabel()
        self.label2.setSizePolicy(sizePolicy)
        self.image2 = QImage()
        self.button2 = QPushButton()
        self.button2.setCursor(Qt.PointingHandCursor)
        self.button2.clicked.connect(lambda: self.saveFig(self.image2))
        self.box2 = QGroupBox()
        layout = QVBoxLayout(self.box2)
        layout.addWidget(self.label2)
        layout.addWidget(self.button2)

        self.label3 = QLabel()
        self.label3.setSizePolicy(sizePolicy)
        self.image3 = QImage()
        self.button3 = QPushButton()
        self.button3.setCursor(Qt.PointingHandCursor)
        self.button3.clicked.connect(lambda: self.saveFig(self.image3))
        self.box3 = QGroupBox()
        layout = QVBoxLayout(self.box3)
        layout.addWidget(self.label3)
        layout.addWidget(self.button3)

        main_layout.addWidget(self.box1)
        main_layout.addWidget(self.box2)
        main_layout.addWidget(self.box3)

    def loadRGB(self):
        total = getRGB(self.main_window.current_img)
        self.image1 = arrayToImage(total[0])
        self.image2 = arrayToImage(total[1])
        self.image3 = arrayToImage(total[2])
        self.updateImg()
        self.box1.setTitle('R分量')
        self.box2.setTitle('G分量')
        self.box3.setTitle('B分量')
        self.button1.setText('保存R分量图')
        self.button2.setText('保存G分量图')
        self.button3.setText('保存B分量图')
        self.setWindowTitle('RGB分量图')

    def loadHSI(self):
        total = getHsI(R2H(self.main_window.current_img))
        self.image1 = arrayToImage(total[0])
        self.image2 = arrayToImage(total[1])
        self.image3 = arrayToImage(total[2])
        self.updateImg()
        self.box1.setTitle('H分量')
        self.box2.setTitle('S分量')
        self.box3.setTitle('I分量')
        self.button1.setText('保存H分量图')
        self.button2.setText('保存S分量图')
        self.button3.setText('保存I分量图')
        self.setWindowTitle('HSI分量图')

    def saveFig(self, image):
        path, _ = QFileDialog.getSaveFileName(self, '保存为', './untitled', 'JPEG(*.jpg;*.jpeg);;PNG(*.png);;BMP(*.bmp);;TIFF(*.tif;*.tiff)')
        if len(path) > 0:
            try:
                image.save(path)
            except:
                QMessageBox.critical(self, '错误', '保存失败')

    def updateImg(self):
        pixmap1 = QPixmap.fromImage(self.image1).scaled(self.label1.size(), aspectRatioMode=Qt.KeepAspectRatio)
        pixmap2 = QPixmap.fromImage(self.image2).scaled(self.label2.size(), aspectRatioMode=Qt.KeepAspectRatio)
        pixmap3 = QPixmap.fromImage(self.image3).scaled(self.label3.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.label1.setPixmap(pixmap1)
        self.label2.setPixmap(pixmap2)
        self.label3.setPixmap(pixmap3)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.updateImg()

