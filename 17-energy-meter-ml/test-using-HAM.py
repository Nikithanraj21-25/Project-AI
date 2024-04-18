from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

fileName = "Energy Meter.csv"
names = ['Voltage','Current','Power','Class']
dataset = read_csv(fileName, names=names)

array = dataset.values
x = array[:,0:3]
y = array[:,3]
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=1,shuffle=True)

model = SVC(gamma = 'auto')

model.fit(x_train,y_train)

result = model.score(x_val, y_val)
print(result)

value=[[215.7979,0.170775,36.85288637]]
predictions = model.predict(value)
print(predictions[0])

