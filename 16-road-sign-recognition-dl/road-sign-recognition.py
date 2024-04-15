from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data = []
labels = []

classes = 43

cur_path = os.getcwd()  # To get current directory

classs = {1: "Speed limit (20km/h)",
          2: "Speed limit (30km/h)",
          3: "Speed limit (50km/h)",
          4: "Speed limit (60km/h)",
          5: "Speed limit (70km/h)",
          6: "Speed limit (80km/h)",
          7: "End of speed limit (80km/h)",
          8: "Speed limit (100km/h)",
          9: "Speed limit (120km/h)",
          10: "No passing",
          11: "No passing veh over 3.5 tons",
          12: "Right-of-way at intersection",
          13: "Priority road",
          14: "Yield",
          15: "Stop",
          16: "No vehicles",
          17: "Veh > 3.5 tons prohibited",
          18: "No entry",
          19: "General caution",
          20: "Dangerous curve left",
          21: "Dangerous curve right",
          22: "Double curve",
          23: "Bumpy road",
          24: "Slippery road",
          25: "Road narrows on the right",
          26: "Road work",
          27: "Traffic signals",
          28: "Pedestrians",
          29: "Children crossing",
          30: "Bicycles crossing",
          31: "Beware of ice/snow",
          32: "Wild animals crossing",
          33: "End speed + passing limits",
          34: "Turn right ahead",
          35: "Turn left ahead",
          36: "Ahead only",
          37: "Go straight or right",
          38: "Go straight or left",
          39: "Keep right",
          40: "Keep left",
          41: "Roundabout mandatory",
          42: "End of no passing",
          43: "End no passing veh > 3.5 tons"}


# Retrieving the images and their labels
print("Obtaining Images & its Labels...")
for i in range(classes):
    path = os.path.join(cur_path, 'dataset/train/', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
            print("{0} Loaded".format(a))

        except Exception as e:
            print("Error loading image:", e)
print("Dataset Loaded")

# Converting list into arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

# splitting training and testing test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                    random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseButton = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseButton.setGeometry(QtCore.QRect(190, 370, 131, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.BrowseButton.setFont(font)
        self.BrowseButton.setObjectName("BrowseButton")
        self.TrainingButton = QtWidgets.QPushButton(self.centralwidget)
        self.TrainingButton.setGeometry(QtCore.QRect(470, 440, 131, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.TrainingButton.setFont(font)
        self.TrainingButton.setObjectName("TrainingButton")
        self.ClassifyButton = QtWidgets.QPushButton(self.centralwidget)
        self.ClassifyButton.setGeometry(QtCore.QRect(190, 440, 131, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ClassifyButton.setFont(font)
        self.ClassifyButton.setObjectName("ClassifyButton")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(470, 370, 131, 41))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(190, 100, 411, 221))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(290, 30, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 340, 55, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseButton.clicked.connect(self.loadImage)
        self.ClassifyButton.clicked.connect(self.classifyFunction)
        self.TrainingButton.clicked.connect(self.trainingFunction)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseButton.setText(_translate("MainWindow", "Browse"))
        self.TrainingButton.setText(_translate("MainWindow", "Training"))
        self.ClassifyButton.setText(_translate("MainWindow", "Clssify"))
        self.label_2.setText(_translate("MainWindow", "Character Recognition GUI"))
        self.label_3.setText(_translate("MainWindow", "output"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image",
                                                             "", "Image Files(*.png *.jpg *.jpeg *.bmp);;All Files(*)")
        if fileName:
            try:
                print(fileName)
                self.file = fileName
                pixmap = QtGui.QPixmap(fileName)
                pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
                self.label.setPixmap(pixmap)
                self.label.setAlignment(QtCore.Qt.AlignCenter)
            except Exception as e:
                print("Error loading image:", e)

    def classifyFunction(self):
        model = load_model("model.h5")
        print("Loaded model from disk")
        path2 = self.file
        print(path2)
        test_image = Image.open(path2)
        test_image = test_image.resize((30, 30))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.array(test_image)

        result = model.predict_classes(test_image)[0]
        sign = classs[result + 1]
        print(sign)
        self.textEdit.setText(sign)

    def trainingFunction(self):
        self.textEdit.setText("Training under process...")
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                         input_shape=x_train.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))
        print("Model Initialized")

        # model compilation
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
        model.save("model.h5")

        plt.figure(0)
        plt.plot(history.history['acc'], label='training accuracy')
        plt.plot(history.history['val_acc'], label='val accuracy')
        plt.title("Accuracy")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('Accuracy.png')

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title("Loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('Loss.png')
        self.textEdit.setText("Saved model & Graph to disk")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
