import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,MaxPool2D,Dense,BatchNormalization,Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

X = np.load('spectogram_x3.npy')[:,:,:,None]
y = np.load('label_x3.npy')
y = to_categorical(y,10)

print(X.shape)
print(y.shape)
'''
print('--------------Data Processing Started=------------------')
for i in range(X.shape[0]):
	t = np.concatenate((X[i,:,:645][None,:,:],X[i,:,645:][None,:,:]),axis=0)[:,:,:,None]
	try:
		X1 = np.concatenate((X1,t),axis=0)
	except NameError:
		X1 = t

print(X1.shape)
'''

# for i in range(X1.shape[0]):
# 	try:
# 		y1 = np.concatenate((y1,[i//200]),axis=0)
# 	except NameError:
# 		y1 = np.array([0])

# print('--------------Data Processing Ended------------------')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

def model():
	model = Sequential()
	model.add(Conv2D(6,(3,3),padding='same',input_shape=X_train.shape[1:],activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(MaxPool2D((4,4),strides=4))
	model.add(Conv2D(12,(3,3),padding='same',activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(MaxPool2D((4,4),strides=4))
	model.add(Conv2D(24,(3,3),padding='same',activation='relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(MaxPool2D((2,2),strides=3))
	model.add(Flatten())
	# model.add(Dense(128,activation='relu'))
	# model.add(Dropout(0.3))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32,activation='relu'))
	model.add(Dense(10,activation='softmax'))
	return model
'''
def model_inception():
	X = Input(shape=X.shape[1:])
	X1 = 
'''
model = model()
model.compile(optimizer=Adam(lr=5e-5,decay=5e-7),loss='categorical_crossentropy',metrics=['accuracy','mse'])
print(model.summary())
flag = input("Start Training? - \t")
if flag=='y':
	history = model.fit(X_train,y_train,epochs=75,batch_size=8,validation_data=(X_test,y_test))


plt.plot(history.history['loss'],label='t loss')
plt.plot(history.history['val_loss'],label='v loss')
plt.legend()
plt.show()
