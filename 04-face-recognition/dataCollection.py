import cv2
import os #for directory based operations

haar_file = 'haarcascade_frontalface_default.xml'

datasets = 'dataset'
sub_data = 'Surya'

path = os.path.join(datasets, sub_data) #dataset/Nikithan
if not os.path.isdir(path):
    os.mkdir(path)
(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file) #loading face detection algorithm
webcam = cv2.VideoCapture(0)

count = 1

while count < 31:
    print(count)
    _,img = webcam.read() #read frame from camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # color frame to gray
    faces = face_cascade.detectMultiScale(gray, 1.3,4) # detect face
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #draw rectangle around face
        face = gray[y:y+h, x:x+w] #crop only face part
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(os.path.join(path, '{}.png'.format(count)), face_resize)

        count += 1

    cv2.imshow('OpenCv', img)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()
        
     


