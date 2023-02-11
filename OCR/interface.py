from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import numpy as np
from numpy.core.arrayprint import str_format
import tensorflow  as tf
from keras.preprocessing import image








class Ui_MainWindow(object):
    path_aug="/home/radia/Downloads/file2/content/saved_model/my_model2"
    path_no_aug="/home/radia/Downloads/file/content/saved_model/my_model"
        
    
    model = tf.keras.models.load_model(path_aug)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 606)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(40, 40, 581, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(670, 80, 93, 28))
        self.pushButton.setObjectName("pushButton")
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(670, 40, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 111, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 230, 301, 211))
        #self.label_2.setGeometry(QtCore.QRect(30, 220, 741, 321))
        
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 80, 191, 131))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 30, 160, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout.addWidget(self.radioButton_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(280, 110, 400, 150))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Project TAL"))
        self.pushButton.setText(_translate("MainWindow", "Start training"))
        self.label.setText(_translate("MainWindow", "Choose an image"))
        self.groupBox.setTitle(_translate("MainWindow", "Training parameters"))
        self.radioButton.setText(_translate("MainWindow", "With Augmentaion"))
        self.radioButton_2.setText(_translate("MainWindow", "No Augmentaion "))
        self.pushButton_2.setText(_translate("MainWindow", "Browse"))

        self.pushButton.clicked.connect(self.startTraining)
        self.pushButton_2.clicked.connect(self.browse)
        self.statusbar.setEnabled(True)
        self.lineEdit.setPlaceholderText("Path to image")

        # f retranslate à la fin ga3
        self.radioButton.clicked.connect(self.diplayAug)
        self.radioButton_2.clicked.connect(self.displayNoAug)
# nouvelles méthodes au mm niveau de retranslate
    def displayNoAug(self):
        self.label_3.setText("number of convolutionel layers 3 \n  learning rate  0.001 \n activation_func relu\n n° of neurones in the hiden  layer is 384\nloss func loss=categorical_crossentropy")
                
                
                

    def diplayAug(self):
        self.label_3.setText("number of convolutionel layers 3 \n  learning rate  0.001 \n activation_func relu\n n° of neurones in the hiden  layer is 288\nloss func loss=categorical_crossentropy\nthe optimal dropout value is 0.2.")
                
    def startTraining(self):
        img = self.lineEdit.text()
        result =""
        if (img):
            if (self.radioButton.isChecked()):
                global path_aug
                path_aug="/home/radia/Downloads/file2/content/saved_model/my_model2"
            
                res,c=self.prediction(path_aug,path_image)
                # Augmentation
                c=" \nName :"+c
                result = "\n---------------Found classes ----------------\n\n"+str(res) +c
            else:
                
                global path_no_aug
                path_no_aug="/home/radia/Downloads/file/content/saved_model/my_model"
                # No Augmentation
                res,c= self.prediction(path_no_aug,path_image)
                c=" \nName :"+c
                result = "\n---------------Found classes ----------------\n\n"+str(res) +c
            self.label_2.setText(result)
        else : 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Cannot find image")
            msg.setInformativeText('Please enter a valid path.')
            msg.setWindowTitle("Error")
            msg.exec_()

    def browse(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","Images (*.png *.jpg *.tif)", options=options)
        if fileName:
            self.lineEdit.setText(fileName)
            global path_image
            path_image=fileName
    
    def prediction(self,path, patimg):
        
        model = tf.keras.models.load_model(path)
        s=['AAN',"ABD",'ALA','ALADHI','ALAM','ALLAH','ALLATI','ALYAWM','AN','AW','FI'
        ,'HADHA','HADHIHI','HIA','HOUNAKA','HOWA','ILA','KABLA','KAD','KAMA'
        ,'KANA','KHILALA','MA','MAA','MAN','MOHAMED','TOMA','YAKON']
        img = image.load_img(patimg, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes =  model.predict(images, batch_size=10)
        print("---------------Found classes ----------------")
        print( classes[0])
        print( classes[0][0])
        a=""
        for i in range (len(classes[0])):
            if (1-classes[0][i])<0.1:
                print(s[i])
                a=s[i]

        return classes [0],a


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    # pyuic5 -x "tal.ui" -o interface.py

    sys.exit(app.exec_())
