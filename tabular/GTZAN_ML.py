import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import seaborn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

vals = pd.read_csv('GTZAN_Mean_STD.csv')
vals = vals.drop(columns='filename')

genre_list = vals.iloc[:,-1]
encoder = sklearn.preprocessing.LabelEncoder()
y = encoder.fit_transform(genre_list)

vals = vals.drop(columns='label')
vals['label'] = y

y = np.array(vals['label'])
X = np.array(vals.drop(columns='label'))
X = sklearn.preprocessing.StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X = PCA(n_components=2).fit_transform(X)


DC = LogisticRegression(multi_class='multinomial',max_iter=3000,C=0.1)
#DC = RandomForestClassifier(n_estimators=10,max_features=2,max_depth=8)
#DC = SVC()
#lgcf = RFE(lgc,12)
model_dc = DC.fit(X_train,y_train)
predictions_dc = model_dc.predict(X_test)
accuracy_dc = accuracy_score(y_test,predictions_dc)
precision_dc = precision_score(y_test,predictions_dc,average='macro')
recall_dc = recall_score(y_test,predictions_dc,average='macro')
f1_dc = f1_score(y_test,predictions_dc,average='macro')
predictions_train_dc = model_dc.predict(X_train)
accuracy_train_dc = accuracy_score(y_train,predictions_train_dc)
confusion_dc = confusion_matrix(y_test, predictions_dc)
print('----------------------------------------------------LogisticRegression----------------------------------------------------')
print('Training set accuracy=\t',accuracy_train_dc)
print('Test set accuracy=\t',accuracy_dc)
print('Test set precision=\t',precision_dc)
print('Test set recall=\t',recall_dc)
print('Test set f1-score=\t',f1_dc)
print('Confusion Matrix=\t\n',confusion_dc)


#DC = LogisticRegression(multi_class='multinomial',max_iter=1000)
#DC = RandomForestClassifier(n_estimators=10,max_features=2,max_depth=8)
DC = SVC()
#lgcf = RFE(lgc,12)
model_dc = DC.fit(X_train,y_train)
predictions_dc = model_dc.predict(X_test)
accuracy_dc = accuracy_score(y_test,predictions_dc)
precision_dc = precision_score(y_test,predictions_dc,average='macro')
recall_dc = recall_score(y_test,predictions_dc,average='macro')
f1_dc = f1_score(y_test,predictions_dc,average='macro')
predictions_train_dc = model_dc.predict(X_train)
accuracy_train_dc = accuracy_score(y_train,predictions_train_dc)
confusion_dc = confusion_matrix(y_test, predictions_dc)
print('----------------------------------------------------------SVM------------------------------------------------------------')
print('Training set accuracy=\t',accuracy_train_dc)
print('Test set accuracy=\t',accuracy_dc)
print('Test set precision=\t',precision_dc)
print('Test set recall=\t',recall_dc)
print('Test set f1-score=\t',f1_dc)
print('Confusion Matrix=\t\n',confusion_dc)


#DC = LogisticRegression(multi_class='multinomial',max_iter=1000)
DC = RandomForestClassifier(n_estimators=10,max_features=3,max_depth=8)
#DC = SVC()
#lgcf = RFE(lgc,12)
model_dc = DC.fit(X_train,y_train)
predictions_dc = model_dc.predict(X_test)
accuracy_dc = accuracy_score(y_test,predictions_dc)
precision_dc = precision_score(y_test,predictions_dc,average='macro')
recall_dc = recall_score(y_test,predictions_dc,average='macro')
f1_dc = f1_score(y_test,predictions_dc,average='macro')
predictions_train_dc = model_dc.predict(X_train)
accuracy_train_dc = accuracy_score(y_train,predictions_train_dc)
confusion_dc = confusion_matrix(y_test, predictions_dc)
print('-------------------------------------------------RandomForest------------------------------------------------------')
print('Training set accuracy=\t',accuracy_train_dc)
print('Test set accuracy=\t',accuracy_dc)
print('Test set precision=\t',precision_dc)
print('Test set recall=\t',recall_dc)
print('Test set f1-score=\t',f1_dc)
print('Confusion Matrix=\t\n',confusion_dc)
