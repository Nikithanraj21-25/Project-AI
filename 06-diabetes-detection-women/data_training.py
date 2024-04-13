from numpy import loadtxt # load the dataset
from keras.models import Sequential # adding layers in sequential order
from keras.layers import Dense #perform mathematics of actiavtion Fn

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',') #for read / for load dataset

# segragate data into input and output
x = dataset[:,0:8] # input/feature/x
y = dataset[:,8] # output/classes/y
print(x)

# designing neural network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(48, activation = 'relu'))
model.add(Dense(96, activation = 'relu'))
model.add(Dense(124, activation = 'relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train
model.fit(x, y, epochs=20, batch_size=10)

_,accuracy = model.evaluate(x,y)
print('Accuracy: %.2f' % (accuracy*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
 json_file.write(model_json)
model.save_weights("model.h5")
print("Model saved to disk")






