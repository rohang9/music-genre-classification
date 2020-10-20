import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


vals = pd.read_csv('GTZAN_Mean_STD.csv')
vals = vals.drop(columns='filename')

labels = vals.label
one_hot = pd.get_dummies(labels)

y = np.array(one_hot)

X = np.array(vals.drop(columns='label'))
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def make_model():
	model = Sequential()
	model.add(Dense(256,activation='relu',input_shape=(X.shape[1],)))
	model.add(Dropout(0.2))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(32,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(16,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10,activation='softmax'))
	return model

model = make_model()
opt = Adam(lr=1e-3,decay=1e-6)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=25,validation_data=(X_test,y_test))

#plotting of graphs
plt.plot(history.history['loss'],label="training loss")
plt.plot(history.history['val_loss'],label="validation loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],label="training accuracy")
plt.plot(history.history['val_accuracy'],label="validation accuracy")
plt.xlabel("Iterations")
plt.ylabel("accuracy")
plt.title("accuracy vs Iterations")
plt.legend()
plt.show()
