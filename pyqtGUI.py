# -*- coding: utf-8 -*-

"""This code implements box size recognition on a conveyor with UI."""

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton,QMainWindow, QVBoxLayout, QSlider,QHBoxLayout
from PyQt5.QtCore import QRect, Qt, QThread, pyqtSignal, pyqtSlot,QObject
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
from time import sleep
import argparse
import imutils
import os
import serial
import serial.tools.list_ports



class Signals(QObject):
    """Class to change Canny filter threshold value
    
    Attributes:
        None
    """
    change_canny_value = pyqtSignal(int)
    

class VideoThread(QThread):
    """Worker class for box detection. Called in thread.
    
    Attributes:
        change_pixmap_signal: Signal passing image array from VideoThread to main thread
        change_pixmap_signal1: Signals passing image array from VideoThread to main thread
        change_size_value: Signal passing box size values to main thread
        change_box_len_value: Signal passing length-meter values
        error_flag: Interrupt flag
        sigs: Signals() class instance 
        _run_flag (bool): flag that window is running
        cannyVal (int): canny filter thresh value
        px_cm_x (int): Camera's parameters:horizontal resolution / matrix size in CM
        px_cm_y (int): Camera's parameters:vertical resolution / matrix size in CM
        focal_length_cm (int): Camera lens focal length
    Args:
            rect: np.ndarray object to crop frame
            sensPort: str sensor port name
    """

    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pixmap_signal1 = pyqtSignal(np.ndarray)
    change_size_value = pyqtSignal(str)
    change_box_len_value = pyqtSignal(str)
    error_flag = pyqtSignal(bool)

    def __init__(self, rect, sensPort):
        super().__init__()
        self.rect = rect
        self.sensPort = sensPort
        self.sigs = Signals()
        self.sigs.change_canny_value.connect(self.update_canny)
        self._run_flag = True
        self.cannyVal = 0
        self.px_cm_x = 640/0.24 
        self.px_cm_y = 480/0.18 
        self.focal_length_cm = 0.41 
        
    
    def detect(self, frame, table_dist, dist_to_box,canny_thresh):
        """ Method to detect a box in frame.
        Uses canny filter, contour detection, polygon generalization and min bounding rectangle to recieve decent sizes.
        Converts box sizes from pixels to cm using thin lens equation.
        args: 
            frame (np.ndarray): image as 3-dimensional array 
            table_dist (int): static initial distance from camera to table
            dist_to_box (int): distance from camera to box, updates dynamically
            canny_thresh (int): canny filter threshold value in range(0,255) recieved from UI slider object

        returns:
            box1 (np.ndarray) - array with bounding rectangle coordinates
        """
        kernel = np.ones((5,5),np.uint8)
        kernelDilation = np.ones((3,3),np.uint8)
        #Camera to floor distance
        table = table_dist
        #Camera to box distance
        dist_box_cm = dist_to_box
        box_h = table - dist_box_cm
        image = frame
        #Resize image (if necessary)
        scale_percent = 100 #Percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim)
        ratio = resized.shape[0] / float(resized.shape[0])
        #Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        #Blur
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0) - alternative variant
        blurred = cv2.bilateralFilter(gray,6,75,75)
        #Canny filter to detect contours on image
        edges = cv2.Canny(blurred,0,canny_thresh)
        #Threshold image 
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(closing,kernelDilation,iterations = 2)
        box1 = None
        try:
            #Finding contours
            cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
            rectAreas = list()
            rectShapes = list()
            for c in cnts:
                M = cv2.moments(c)
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                rectAreas.append(cv2.contourArea(c))
                rectShapes.append(c)
            #Selecting the biggest contour 
            maxIndex = rectAreas.index(max(rectAreas))
            #Convex hull
            hull = cv2.convexHull(rectShapes[maxIndex], returnPoints=True)
            
            #Approximating Hull to a polygon to reduce unnecesary edges
            box = cv2.approxPolyDP(hull, 0.07 * cv2.arcLength(hull,True), True)
            if len(box) == 4:
                #Creating a min bounding rectangle for aprroximated polygon
                rect = cv2.minAreaRect(box)
                box1 = cv2.boxPoints(rect) 
                box1 = np.int0(box1)
                cv2.drawContours(resized,[box1],0,(0,255,0),2)
                #Calculating box sizes in pixels
                px_len1 = abs(np.sqrt((box1[0][0]- box1[1][0])**2+(box1[0][1]- box1[1][1])**2))
                px_len2 = abs(np.sqrt((box1[1][0]- box1[2][0])**2+(box1[1][1]- box1[2][1])**2))
                #Converting pixels to cm
                size_x_cm_mtx = px_len1/self.px_cm_x
                size_y_cm_mtx = px_len2/self.px_cm_y
                H1 = (((dist_box_cm*size_x_cm_mtx)-(self.focal_length_cm*size_x_cm_mtx))/self.focal_length_cm)/(scale_percent/100)
                H2 = ((dist_box_cm*size_y_cm_mtx-self.focal_length_cm*size_y_cm_mtx)/self.focal_length_cm)/(scale_percent/100)
                
                cv2.putText(resized, f'{H1}, {H2}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1,1) 
                cv2.putText(resized, f'{box_h} box height', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1,1) 
                cv2.putText(resized, f'{table} table dist', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1,1)
                cv2.putText(resized, f'{dist_box_cm} box dist', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1,1)
                
            else:
                pass
            new_dims =(image.shape[0], image.shape[1])
            #resized = cv2.resize(resized, new_dims)
            #Emitting image arrays to main thread
            self.change_pixmap_signal.emit(thresh)
            self.change_pixmap_signal1.emit(resized)
            return box1
        except:
            cv2.putText(resized, f'{table} table dist', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1,1)
            #Emitting image arrays to main thread
            self.change_pixmap_signal.emit(thresh)
            self.change_pixmap_signal1.emit(resized)
            return box1

    
    def pollSerial(self, port):
        """#Polling serial port length meter (??needs revision and rewriting??)"""
        data = port.readline()
        if len(data) != 0:
            string_n = data.decode()  # decode byte string into Unicode  
            string = string_n.rstrip() # remove \n and \r
            intLen = int(string)        # convert string to float
            return intLen
    
    
    @pyqtSlot(int)
    def update_canny(self, canny):
        """Slot to update Canny thresh value
        Args:
            canny (int): recieved via signal
        Returns:
            None
        """
        self.cannyVal = canny

    
    def run(self):
        """Main worker method that captures camera stream measures distance to box, calls detect() method
        Args: 
            None
        Returns:
            None
        """
        #Capture from usb web cam
        cap = cv2.VideoCapture(0)
        errFlag = False
        #Initializing port
        port_object_1 = serial.Serial(self.sensPort,115200, timeout =1, stopbits=1)
        #port_object_2 = serial.Serial(port_list[1],115200, timeout =1, stopbits=1)
        '''
        Asking serial devices 
        port_object_1.write('Who are you')
        data1 = port_object_1.read()
        port_object_2.write('Who are you')
        data2 = port_object_2.read()
        if data1 == 'i am sensor':
            port_name1 = port_object_1.name
            port_object_1.close()
            sensor_port = serial.Serial(port_name1,115200, timeout =1, stopbits=1)
            table_dist = pollSerial(port_object_1)
        elif data1 == 'i am screen':
            port_name2 = port_object_2.name
            port_object_1.close()
            sensor_port = serial.Serial(port_name2,115200, timeout =1, stopbits=1)

        if data2 == 'i am sensor':
            port_name1 = port_object_1.name
            port_object_2.close()
            sensor_port = serial.Serial(port_name1,115200, timeout =1, stopbits=1)
            table_dist = pollSerial(port_object_1)
        elif data2 == 'i am screen':
            port_name2 = port_object_2.name
            port_object_2.close()
            sensor_port = serial.Serial(port_name2,115200, timeout =1, stopbits=1)
        ''' 
        #Measuring initial camera-table distance
        table_dist = self.pollSerial(port_object_1)
        #Creating a blank image if error occurs
        blank_image = np.zeros((480,620,3), np.uint8)
        blank_image[:,0:620] = (255,0,0)
        contourList = list()
        
        #Run while window is up
        while self._run_flag:
            try:
                box_dist = self.pollSerial(port_object_1)
                self.change_box_len_value.emit(f"Table distance: {table_dist}   Box distance: {box_dist}  ")
                ret, cv_img = cap.read()
                #Cropping image based on ROI rectangle from Adjust window
                cv_img = cv_img[abs(int(self.rect[2])):abs(int(self.rect[3])), abs(int(self.rect[0])):abs(int(self.rect[1]))]
                
                if ret:
                    contour = self.detect(cv_img, table_dist, box_dist, self.cannyVal)
                    #Works only when box passes under the sensors
                    if (table_dist - box_dist )>1:
                        if contour is not None:
                            if len(contourList)<20000:
                                contourList.append(contour)
                            else:
                                contourList = list()
                    else:
                        rectAreas = list()
                        if len(contourList)!=0:
                            middleIndices = int(len(contourList)/2)
                            contourList = contourList[(middleIndices-10):(middleIndices+10)]
                            for c in contourList:
                                rectAreas.append(cv2.contourArea(c))
                            maxIndex = rectAreas.index(max(rectAreas))
                            px_len1 = abs(np.sqrt((contourList[maxIndex][0][0]- contourList[maxIndex][1][0])**2+(contourList[maxIndex][0][1]- contourList[maxIndex][1][1])**2))
                            px_len2 = abs(np.sqrt((contourList[maxIndex][1][0]- contourList[maxIndex][2][0])**2+(contourList[maxIndex][1][1]- contourList[maxIndex][2][1])**2))
                            #print('px_len:',px_len1,px_len2 )
                            size_x_cm_mtx = px_len1/self.px_cm_x
                            size_y_cm_mtx = px_len2/self.px_cm_y
                            H1 = (((box_dist*size_x_cm_mtx)-(self.focal_length_cm*size_x_cm_mtx))/self.focal_length_cm)/(100/100)
                            H2 = ((box_dist*size_y_cm_mtx-self.focal_length_cm*size_y_cm_mtx)/self.focal_length_cm)/(100/100)
                            """At this point we emit the value of boxes' sizes. May insert QR generator here"""
                            self.change_size_value.emit('S1 = %.2f , S2 = %.2f, H = %d      ' % (H1, H2, (table_dist-box_dist)))
                            contourList = list()
                        
                else: 
                    self.change_pixmap_signal.emit(blank_image)
                    self.change_pixmap_signal1.emit(blank_image)
                    errFlag = True
                    self.error_flag.emit(errFlag)
                    break
            except:
                errFlag = True
                self.error_flag.emit(errFlag)
                break
        #Shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class MyLabel(QLabel):
    """Class for ROI selection
    Attributes: 
        x0,y0,x1,y1: initial click coordinates
        flag (bool): mouse click/release event flag
        rect (QRect): rectangle object
    """
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    rect =QRect(x0, y0, (x1-x0), (y1-y0))
    
    def mousePressEvent(self,event):
        """Mouse click event"""
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
    
    def mouseReleaseEvent(self,event):
        """Mouse release event"""
        self.flag = False
    
    def mouseMoveEvent(self,event):
        """Mouse move event"""
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()
    
    def paintEvent(self, event):
        """Drawing rectangle"""
        super().paintEvent(event)
        self.rect =QRect(self.x0, self.y0, (self.x1-self.x0), (self.y1-self.y0))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red,2,Qt.SolidLine))
        painter.drawRect(self.rect)
        

class AnotherWindow(QWidget):
    """Worker UI window class"""
    def __init__(self, rect, sensorPort):
        super().__init__()
        self.setGeometry(0, 0, 1000, 600) 
        self.rect = rect
        self.sensorPort = sensorPort
        self.setWindowTitle("Qt live label demo")
        self.display_width = 440
        self.display_height = 360
        self.value = 80
        #Create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        self.image_label1 = QLabel(self)
        self.image_label1.resize(self.display_width, self.display_height)
        #Create a text label
        self.textLabel = QLabel('Webcam')
        self.adjustWin = None
        btn = QPushButton("Adjust", self)
        btn.move(300 , 450)
        self.label = QLabel(str('Return'), self)
        self.label.move(300, 400)
        #Create label with box sizes
        self.labelSize = QLabel(str('                   Size                   '), self)
        #self.labelSize.setStyleSheet("border: 1px solid black;")
        self.labelSize.adjustSize()
        self.labelSize.move(360, 400)
        #Create label  for length meter
        self.labelLen = QLabel(str('Len'), self)
        self.labelLen.adjustSize()
        self.labelLen.move(360, 380)
        #Create button returning to adjust screen
        btn.clicked.connect(self.buttonClicked)
        self.error_flag = False
        #Сreate a vertical box layout and add the two labels
        vbox = QHBoxLayout()
        vbox.addWidget(self.image_label)
        self.image_label.move(0,0)
        self.image_label1.move(450,0)
        vbox.addWidget(self.image_label1)
        vbox.addWidget(self.textLabel)
        #Create slider widget
        sld = QSlider(Qt.Horizontal, self)
        sld.setRange(0, 255)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setPageStep(5)
        sld.valueChanged.connect(self.updateLabel)
        self.label = QLabel('0', self)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label.setMinimumWidth(80)
        sld.move(400, 450)
        self.label.move(400,500)
        vbox.addWidget(sld)
        vbox.addSpacing(15)
        vbox.addWidget(self.label)
        #Сreate the video capture thread
        self.thread = VideoThread(self.rect, self.sensorPort)
        #Сonnect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_pixmap_signal1.connect(self.update_image1)
        self.thread.change_size_value.connect(self.update_size)
        self.thread.change_box_len_value.connect(self.update_len)
        self.thread.error_flag.connect(self.update_err_flag)
        try:
            self.thread.start()
        except:
            self.buttonClicked()

    
    def updateLabel(self, value):
        """Update current canny value method"""
        self.thread.sigs.change_canny_value.emit(value)
        self.label.setText(str(value))

    def closeEvent(self, event):
        """Close window event"""
        self.thread.stop()
        event.accept()

    def buttonClicked(self):
        """Return to adjust event"""
        self.close()
        if self.adjustWin is None:
            self.winErr = Adjust()
        self.winErr.show()
        
    
    @pyqtSlot(bool)
    def update_err_flag(self, err_flag):
        """Slot updatiung error flag"""
        self.error_flag = err_flag
        if self.error_flag == True:
            self.buttonClicked()

    @pyqtSlot(str)
    def update_len(self, size_str):
        """Slot updating length meter label"""
        self.labelLen.adjustSize()
        self.labelLen.setText(str(size_str))

    @pyqtSlot(str)
    def update_size(self, size_str):
        """Slot updating length meter label"""
        self.labelSize.adjustSize()
        self.labelSize.setText(str(size_str))

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Slot updating image """
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        
    @pyqtSlot(np.ndarray)
    def update_image1(self, cv_img):
        """Slot updating image """
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label1.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Converting np array image format to QPixmap format"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class WaitPort(QThread):
    """Monitoring port connection in thread
    Attributes:
        sensor_port: signal for port monitoring
        _run_flag (bool): flag that window is up
    """
    sensor_port = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        """Worker method monitoring port connection"""
        while self._run_flag:
            ports = serial.tools.list_ports.comports(include_links=False)
            port_list =[]
            for port in ports :
                port_list.append(port.device)
                #print('Find port '+ port.device)
            if len(port_list)!=0:
                portSensor = port_list[0]
                self.sensor_port.emit(portSensor)
            else: 
                self.sensor_port.emit('Connect sensor')

    def stop(self):
        """Stop thread"""
        self._run_flag = False
        self.wait()


class CamAdjust(QThread):
    """Frame crop thread class
    Attributes:
        change_pixmap_signal: Signal passing image array from VideoThread to main thread
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
    
    def run(self):
        """Worker method for camera stream"""
        #Creating a blank image if error occurs
        blank_image = np.zeros((480,620,3), np.uint8)
        blank_image[:,0:620] = (255,0,0)
        while self._run_flag:
            cap = cv2.VideoCapture(0)                
            while self._run_flag:
                ret, cv_img = cap.read()
                if ret:
                    self.change_pixmap_signal.emit(cv_img)
                else:
                    self.change_pixmap_signal.emit(blank_image)
                    break
            cap.release()

    def stop(self):
        """Stop thread"""
        #Sets run flag to False and waits for thread to finish
        self._run_flag = False
        self.wait()


class ErrorWindow(QWidget):
    """Dialogue window class"""
    def __init__(self, errCode):
        super().__init__()
        self.errCode = errCode
        self.setGeometry(0, 0, 200, 100) 
        self.setWindowTitle("Error")
        btn = QPushButton("Exit", self)
        btn.move(50 , 70)
        #Error code here
        self.label = QLabel(str(errCode), self)
        self.label.move(65, 25)
        btn.clicked.connect(self.buttonClicked)
        self.show()
    def buttonClicked(self):
        self.close()
      


class Adjust(QWidget):
    """Adjust window class"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        #Creating worker window object
        self.w = None
        #Creating dialogue window object
        self.winErr = None
        self.setWindowTitle ('draw rectangle in label ')
        self.label = QLabel('Connect sensor', self)
        self.label.setMinimumWidth(80)
        self.lb = MyLabel(self) 
        self.display_width = 620
        self.display_height = 700
        #Create the label that holds the image
        self.image_label = QLabel(self)
        self.resize(self.display_width, self.display_height)
        self.textLabel = QLabel('Webcam Adjust')
        self.label.move(250, 550)
        vbox = QHBoxLayout()
        vbox.addWidget(self.image_label)
        btn1 = QPushButton("Crop", self)
        btn1.move(250 , 600)
        #Creating Camera stream thread object
        self.thread = CamAdjust()
        #Creating port monitor thread object
        self.thread1 = WaitPort()
        #Connecting pixmap signal
        self.thread.change_pixmap_signal.connect(self.update_image)
        #Connecting sensor signal
        self.thread1.sensor_port.connect(self.updateLabel)
        self.thread.start()
        self.thread1.start()
        #Initial ROI rectangle
        self.imgShape = QRect(0, 0, 0, 0 )
        #Connecting button event
        btn1.clicked.connect(self.buttonClicked)
        self.show()
        
    
    @pyqtSlot(str)
    def updateLabel(self, value):
        """Slot for port label update"""
        self.label.setText(str(value))

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.lb.setGeometry(QRect(0, 0, cv_img.shape[1], cv_img.shape[0]))
        self.imgShape = QRect(0, 0, cv_img.shape[1], cv_img.shape[0])
        self.lb.setPixmap(qt_img)
        self.lb.setCursor(Qt.CrossCursor)

    def convert_cv_qt(self, cv_img):
        """Converting np array image format to QPixmap format"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        """Close window event"""
        self.thread.stop()
        event.accept()

    def buttonClicked(self):
        """Button click event"""
        if self.lb.rect == QRect(0, 0, 0, 0 ):
            self.lb.rect = self.imgShape
            rect = (self.lb.rect.x(), self.lb.rect.x()+self.lb.rect.width(),self.lb.rect.y(), self.lb.rect.y()+self.lb.rect.height())
        else:
            if self.lb.rect.width() > 0 and self.lb.rect.height() > 0:
                rect = (self.lb.rect.x(), self.lb.rect.x()+self.lb.rect.width(),self.lb.rect.y(), self.lb.rect.y()+self.lb.rect.height())
            elif self.lb.rect.width() < 0 and self.lb.rect.height() > 0:
                rect = (self.lb.rect.x()+self.lb.rect.width(), self.lb.rect.x(),self.lb.rect.y(), self.lb.rect.y()+self.lb.rect.height())
            elif self.lb.rect.width() > 0 and self.lb.rect.height() < 0:
                rect = (self.lb.rect.x(), self.lb.rect.x()+self.lb.rect.width(), self.lb.rect.y()+self.lb.rect.height(),self.lb.rect.y())
            elif self.lb.rect.width() < 0 and self.lb.rect.height() < 0:
                rect = (self.lb.rect.x()+self.lb.rect.width(), self.lb.rect.x(),self.lb.rect.y()+self.lb.rect.height(),self.lb.rect.y())
        if self.label.text() == 'Connect sensor':
            if self.winErr is None:
                self.winErr = ErrorWindow('Connect sensor')
            self.winErr.show()
        else:
            self.close()
            if self.w is None:
                self.w = AnotherWindow(rect,self.label.text())
            self.w.show()
       

if __name__ == '__main__':
    app = QApplication(sys.argv)
    x = Adjust()
    sys.exit(app.exec_())