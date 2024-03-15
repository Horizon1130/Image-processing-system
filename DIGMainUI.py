# coding=utf-8

import cv2
import numpy as np
from DIChildGUI import *
from DICore import *
from PyQt5.QtPrintSupport import *


class MainGUI(QWidget):
    # 用于标记的常量
    COLOR_IMG = 1  # 彩色图像
    GRAY_IMG = 2  # 灰度图像

    def __init__(self):
        super().__init__()
        # 初始化UI
        self.setUI()
        # 初始化类成员
        self.path = '.'  # 当前文件路径，便于保存等操作
        self.isOpen = False  # 当前是否有文件打开
        self.unsaved = False  # 是否未保存
        self.isPhoto = False  # 是否为照片
        self.type = 0  # 记录当前图像的种类
        self.undo_list = []  # 撤销队列
        self.redo_list = []  # 重做队列
        self.hist_window = HistWindow(self)  # 直方图展示窗口
        # 进入时直接进行一次关闭操作
        self.myClose()

    ''' 所有UI创建 '''

    def setUI(self):
        # 窗口标题
        self.setWindowTitle('数字图像工具')
        # 窗口Icon
        self.setWindowIcon(QIcon('boxIcon.ico'))
        # 窗口尺寸
        self.resize(1200, 850)
        # 设置窗口最小尺寸
        self.setMinimumSize(1, 1)
        # 设置主布局
        self.mainLayout = QGridLayout(self)
        # 设置菜单栏
        self.setMenuUI()
        # 功能模块
        self.setFunctionModule()
        # 图像显示模块
        self.setImageModule()
        # 参数设置模块
        self.setArgumentModule()
        # 设置状态栏
        self.setStatusUI()

    def setMenuUI(self):
        # 创建菜单栏
        menubar = QMenuBar(self)
        self.mainLayout.setMenuBar(menubar)

        # 菜单：文件
        file_menu = QMenu('文件(&F)', self)  # 文件Menu
        menubar.addMenu(file_menu)
        # 菜单项
        open_action = QAction('打开', self)  # 打开Action
        open_action.triggered.connect(self.myOpen)  # 触发
        open_action.setShortcut('Ctrl+O')  # 设置快捷键
        file_menu.addAction(open_action)

        photo_action = QAction('拍照', self)  # 拍照Action
        photo_action.triggered.connect(self.myPhoto)  # 触发
        photo_action.setShortcut('Ctrl+C')  # 设置快捷键
        file_menu.addAction(photo_action)

        file_menu.addSeparator()

        self.save_action = QAction('保存', self)  # 保存Action
        self.save_action.triggered.connect(self.mySave)  # 触发
        self.save_action.setShortcut('Ctrl+S')  # 设置快捷键
        file_menu.addAction(self.save_action)

        self.saveAs_action = QAction('另存为', self)  # 另存为Action
        self.saveAs_action.triggered.connect(self.mySaveAs)  # 触发
        self.saveAs_action.setShortcut('Ctrl+Shift+S')  # 设置快捷键
        file_menu.addAction(self.saveAs_action)

        self.print_action = QAction('打印', self)  # 打印Action
        self.print_action.triggered.connect(self.myPrint)  # 触发
        self.print_action.setShortcut('Ctrl+P')  # 设置快捷键
        file_menu.addAction(self.print_action)

        file_menu.addSeparator()

        self.close_action = QAction('关闭', self)  # 关闭Action
        self.close_action.triggered.connect(self.myClose)  # 触发
        self.close_action.setShortcut('Ctrl+K')  # 设置快捷键
        file_menu.addAction(self.close_action)

        file_menu.addSeparator()

        exit_action = QAction('退出', self)  # 退出Action
        exit_action.triggered.connect(self.close)  # 触发
        exit_action.setShortcut('Alt+F4')  # 设置快捷键
        file_menu.addAction(exit_action)

        # 菜单：编辑
        edit_menu = QMenu('编辑(&E)', self)  # 编辑Menu
        menubar.addMenu(edit_menu)
        # 菜单项
        self.undo_action = QAction('撤销', self)  # 撤销Action
        self.undo_action.triggered.connect(self.myUndo)  # 触发
        self.undo_action.setShortcut('Ctrl+Z')  # 设置快捷键
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction('重做', self)  # 重做Action
        self.redo_action.triggered.connect(self.myRedo)  # 触发
        self.redo_action.setShortcut('Ctrl+Y')  # 设置快捷键
        edit_menu.addAction(self.redo_action)

    def setFunctionModule(self):
        func_box = QGroupBox('图像处理功能')  # 创建容器
        self.func_page = QStackedLayout(func_box)  # 获取布局
        self.mainLayout.addWidget(func_box, 0, 0, 2, 1)

        # 空白页
        self.no_func_page = QWidget()
        self.func_page.addWidget(self.no_func_page)
        layout = QVBoxLayout(self.no_func_page)
        tip = QLabel('未加载图像')
        tip.setAlignment(Qt.AlignCenter)
        layout.addWidget(tip)

        # 灰度图像页
        self.gray_func_page = QWidget()
        self.func_page.addWidget(self.gray_func_page)
        layout = QVBoxLayout(self.gray_func_page)

        # 创建按钮
        button = QPushButton('显示直方图')
        button.clicked.connect(self.showHist)
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('基本变换')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.base_gray_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('直方图均衡化')
        button.clicked.connect(lambda: self.commonDeal(equalizehist))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('灰度拉伸')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.gray_stretch_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('添加噪声')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.noise_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('平滑滤波')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.smooth_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('锐化滤波')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.sharpen_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('图像分割')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.segmentation_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        # 彩色图像页
        self.color_func_page = QWidget()
        self.func_page.addWidget(self.color_func_page)
        layout = QVBoxLayout(self.color_func_page)

        # 创建按钮
        button = QPushButton('基本变换')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.base_color_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('转灰度图像')
        button.clicked.connect(lambda: self.commonDeal(RGB2GRAY))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('直方图均衡化-I通道')
        button.clicked.connect(lambda: self.commonDeal(HSIhisteq))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

        button = QPushButton('RGB与HSI')
        button.clicked.connect(lambda: self.page_widget.setCurrentWidget(self.HSI_page))
        button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(button)

    def setImageModule(self):
        image_widget = QWidget()  # 创建容器
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(9)
        image_widget.setSizePolicy(sizePolicy)  # 设置容器尺寸策略
        layout = QGridLayout(image_widget)  # 获取布局
        self.mainLayout.addWidget(image_widget, 0, 1)

        # 提示字符
        self.original_text = QLabel('原图像')
        self.original_text.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout.addWidget(self.original_text, 0, 0)
        self.processed_text = QLabel('处理后图像')
        self.processed_text.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        layout.addWidget(self.processed_text, 0, 1)

        # 两个图像Label
        label_sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label_sizePolicy.setVerticalStretch(1)
        self.original_label = QLabel()  # 原始图像窗
        self.original_label.setSizePolicy(label_sizePolicy)  # 设置尺寸策略
        self.original_label.setScaledContents(True)  # 图像自适应
        self.original_image = QImage()  # 原始图像
        self.processed_label = QLabel()  # 处理后图像窗
        self.processed_label.setSizePolicy(label_sizePolicy)  # 设置尺寸策略
        self.processed_label.setScaledContents(True)  # 图像自适应
        self.processed_image = QImage()  # 处理后图像
        layout.addWidget(self.original_label, 1, 0)
        layout.addWidget(self.processed_label, 1, 1)

    def setArgumentModule(self):
        arg_box = QGroupBox('参数选项')  # 参数选项组
        self.mainLayout.addWidget(arg_box, 1, 1)

        self.page_widget = QStackedWidget()  # 创建容器
        QVBoxLayout(arg_box).addWidget(self.page_widget)

        # 空页
        self.null_page = QWidget()
        self.page_widget.addWidget(self.null_page)

        # 灰度图基本变换
        self.base_gray_page = BaseGrayPage(self)
        self.page_widget.addWidget(self.base_gray_page)

        # 彩图基本变换
        self.base_color_page = BaseColorPage(self)
        self.page_widget.addWidget(self.base_color_page)

        # 灰度拉伸
        self.gray_stretch_page = GrayStretchPage(self)
        self.page_widget.addWidget(self.gray_stretch_page)

        # 噪声
        self.noise_page = NoisePage(self)
        self.page_widget.addWidget(self.noise_page)

        # 平滑
        self.smooth_page = SmoothPage(self)
        self.page_widget.addWidget(self.smooth_page)

        # 锐化
        self.sharpen_page = SharpenPage(self)
        self.page_widget.addWidget(self.sharpen_page)

        # 分割
        self.segmentation_page = SegmentationPage(self)
        self.page_widget.addWidget(self.segmentation_page)

        # HSI
        self.HSI_page = HSIPage(self)
        self.page_widget.addWidget(self.HSI_page)

    def setStatusUI(self):
        self.statusBar = QStatusBar(self)
        self.mainLayout.addWidget(self.statusBar, 2, 0, 1, 3)
        self.statusBar.showMessage('状态栏test')

    ''' 所有重写函数 '''

    def resizeEvent(self, a0):
        self.updateImg()

    def closeEvent(self, a0):
        if self.myClose():
            a0.accept()
        else:
            a0.ignore()

    ''' 所有触发性函数 '''

    # 打开
    def myOpen(self) -> bool:
        try:
            path, _ = QFileDialog.getOpenFileName(self, '打开', self.path, '图像文件(*.bmp;*.jpg;*.jpeg;*.png;*.tif;*.tiff)')
            if len(path) > 0 and self.myClose():
                self.basic_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)  # 载入图像
                self.current_img = self.basic_img.copy()  # 拷贝
                self.original_image = arrayToImage(self.basic_img)  # 转换成QImage
                self.processed_image = self.original_image.copy()  # 拷贝
                self.isOpen = True
                self.isPhoto = False
                self.path = path  # 更新文件路径
                self.updateType()  # 更新图像类型
                self.updateAble()  # 更新功能可用状态
                self.updateImg()  # 更新图像
                self.updateStatus()  # 更新状态栏
                return True
            else:
                return False
        except:
            QMessageBox.critical(self, '错误', '图像打开失败')
            return False
    # 拍照
    def myPhoto(self) -> bool:
        try:
            if self.myClose():
                QMessageBox.information(self, '提示', '按Enter拍照，按q取消')
                flag, frame = photo()
                if flag:
                    self.basic_img = frame
                    self.current_img = self.basic_img.copy()
                    self.original_image = arrayToImage(self.basic_img)
                    self.processed_image = self.original_image.copy()
                    self.isOpen = True
                    self.isPhoto = True
                    self.unsaved = True  # 照片属于未保存
                    self.updateType()  # 更新图像类型
                    self.updateAble()  # 更新功能可用状态
                    self.updateImg()  # 更新图像
                    self.updateStatus()  # 更新状态栏
                    self.updateTitle()  # 更新标题
                    return True
                else:
                    QMessageBox.information(self, '提示', '已取消')
                    return False
            else:
                return False
        except:
            QMessageBox.critical(self, '错误', '拍照失败')
            return False
    # 保存
    def mySave(self) -> bool:
        if self.isPhoto:  # 如果是照片直接调用另存为
            if self.mySaveAs():
                self.isPhoto = False  # 另存成功
                self.updateStatus()  # 更新状态栏
                return True  # 保存成功
            else:
                return False
        else:
            try:
                self.processed_image.save(self.path)  # 保存
                self.unsaved = False  # 已保存
                self.updateTitle()  # 更新标题
                return True  # 保存成功
            except:
                QMessageBox.critical(self, '错误', '保存失败')
                return False
    # 另存
    def mySaveAs(self) -> bool:
        try:
            path, _ = QFileDialog.getSaveFileName(self, '保存为', './untitled', 'JPEG(*.jpg;*.jpeg);;PNG(*.png);;BMP(*.bmp);;TIFF(*.tif;*.tiff)')
            if len(path) > 0:
                self.processed_image.save(path)  # 保存
                self.path = path  # 修改当前路径
                self.unsaved = False  # 已保存
                self.updateTitle()  # 更新标题
                self.updateStatus()  # 更新状态栏
                return True  # 保存成功
            else:
                return False
        except:
            QMessageBox.critical(self, '错误', '保存失败')
            return False
    # 关闭
    def myClose(self) -> bool:
        doClose = False  # 是否关闭
        if self.unsaved:
            reply = QMessageBox.question(self, '提示', '有未保存的更改，是否保存？', QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:  # 选择保存
                if self.mySave():  # 如果保存成功
                    doClose = True
            elif reply == QMessageBox.No:  # 选择不保存
                self.unsaved = False  # 取消未保存状态
                self.updateTitle()  # 更新标题
                doClose = True
        else:  # 不需要保存
            doClose = True

        if doClose:
            self.original_image = self.processed_image = QImage()  # 清空图像
            self.isOpen = False  # 未打开状态
            self.undo_list.clear()  # 清空队列
            self.redo_list.clear()  # 清空队列
            self.hist_window.close()  # 关闭直方图窗口
            self.updateImg()  # 更新图像
            self.updateAble()  # 更新功能可用状态
            self.updateStatus()  # 更新状态栏
            return True
        else:
            return False
    # 打印
    def myPrint(self) -> bool:
        try:
            printer = QPrinter()  # 创建打印
            print_dialog = QPrintDialog(printer, self)
            if print_dialog.exec():
                painter = QPainter(printer)  # 创建绘图
                rect = painter.viewport()  # 获取绘图窗口大小
                size = self.processed_image.size()  # 获取图像大小
                size.scale(rect.size(), Qt.KeepAspectRatio)  # 保持纵横比
                painter.setViewport(rect.x(), rect.y(), size.width(), size.height())  # 调整打印视图大小
                painter.setWindow(self.processed_image.rect())
                painter.drawImage(0, 0, self.processed_image)
                painter.end()
                return True
            else:
                return False
        except:
            QMessageBox.critical(self, '错误', '打印失败')
            return False
    # 撤销
    def myUndo(self):
        self.redo_list.append(self.current_img)
        self.current_img = self.undo_list.pop(-1)
        self.unsaved = True
        self.processed_image = arrayToImage(self.current_img)
        self.hist_window.close()  # 关闭直方图窗口
        self.updateImg()
        self.updateType()
        self.updateTitle()
        self.updateStatus()
        self.updateAble()
    # 重做
    def myRedo(self):
        self.undo_list.append(self.current_img)
        self.current_img = self.redo_list.pop(-1)
        self.unsaved = True
        self.processed_image = arrayToImage(self.current_img)
        self.hist_window.close()  # 关闭直方图窗口
        self.updateImg()
        self.updateType()
        self.updateTitle()
        self.updateStatus()
        self.updateAble()

    def showHist(self):
        if self.hist_window.isHidden():
            try:
                self.hist_window.loadHist()
                self.hist_window.show()
            except:
                QMessageBox.critical(self, '错误', '直方图加载失败')

    def commonDeal(self, func, *args):
        try:
            self.addUndo()  # 添加撤销
            self.current_img = func(self.current_img.copy(), *args)  # 进行处理
            self.unsaved = True
            self.processed_image = arrayToImage(self.current_img)  # 转换格式
            self.hist_window.close()  # 关闭直方图窗口
            self.updateTitle()  # 更新标题
            self.updateType()  # 更新类型
            self.updateImg()  # 更新图像
            self.updateAble()  # 更新功能可用状态
            self.updateStatus()  # 更新状态栏
        except:
            QMessageBox.critical(self, '错误', '图像处理失败')

    ''' 所有辅助性函数 '''
    def addUndo(self):
        self.undo_list.append(self.current_img)
        self.redo_list.clear()

    def updateAble(self):
        if self.isOpen:
            if self.type == self.GRAY_IMG:
                self.func_page.setCurrentWidget(self.gray_func_page)
            elif self.type == self.COLOR_IMG:
                self.func_page.setCurrentWidget(self.color_func_page)
            self.original_text.setVisible(True)
            self.processed_text.setVisible(True)
            self.save_action.setEnabled(True)
            self.saveAs_action.setEnabled(True)
            self.print_action.setEnabled(True)
            self.close_action.setEnabled(True)
        else:
            self.func_page.setCurrentWidget(self.no_func_page)
            self.original_text.setVisible(False)
            self.processed_text.setVisible(False)
            self.page_widget.setCurrentWidget(self.null_page)
            self.save_action.setDisabled(True)
            self.saveAs_action.setDisabled(True)
            self.print_action.setDisabled(True)
            self.close_action.setDisabled(True)

        if len(self.redo_list) == 0:
            self.redo_action.setDisabled(True)
        else:
            self.redo_action.setEnabled(True)
        if len(self.undo_list) == 0:
            self.undo_action.setDisabled(True)
        else:
            self.undo_action.setEnabled(True)

    def updateImg(self):
        # 图像
        origin = QPixmap.fromImage(self.original_image).scaled(self.original_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.original_label.setPixmap(origin)
        processed = QPixmap.fromImage(self.processed_image).scaled(self.processed_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.processed_label.setPixmap(processed)

    def updateTitle(self):
        if self.unsaved:
            if not self.windowTitle().startswith('*'):
                self.setWindowTitle('*' + self.windowTitle())
        else:
            if self.windowTitle().startswith('*'):
                self.setWindowTitle(self.windowTitle()[1:])

    def updateStatus(self):
        if self.isOpen:
            if self.isPhoto:
                self.statusBar.showMessage('原始图像分辨率: %dx%d  处理后图像分辨率: %dx%d  文件: 来自相机'%(self.original_image.width(), self.original_image.height(), self.processed_image.width(), self.processed_image.height()))
            else:
                self.statusBar.showMessage('原始图像分辨率: %dx%d  处理后图像分辨率: %dx%d  文件: %s'%(self.original_image.width(), self.original_image.height(), self.processed_image.width(), self.processed_image.height(), self.path))
        else:
            self.statusBar.showMessage('当前未打开任何图像')

    def updateType(self):
        if self.isOpen:
            if len(self.current_img.shape) == 3:
                if self.type != self.COLOR_IMG:
                    self.page_widget.setCurrentWidget(self.null_page)
                self.type = self.COLOR_IMG
            else:
                if self.type != self.GRAY_IMG:
                    self.page_widget.setCurrentWidget(self.null_page)
                self.type = self.GRAY_IMG


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    win = MainGUI()
    win.show()
    sys.exit(app.exec_())
