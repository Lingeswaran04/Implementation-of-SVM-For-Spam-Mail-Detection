# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output. 

## PROGRAM:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: LINGESWARAN K
RegisterNumber: 212222110022

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
```
```
import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
```
### DATA.HEAD():

```
data.head()
```
### DATA.TAIL():
```
data.tail()
```
### DATA.INFO():
```
data.info()
```
### DATA.ISNULL().SUM():
```
data.isnull().sum()
```
### X,Y DATA VALUES

```
x=data["v1"].values
y=data["v2"].values
```
### SKLEARN
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
### TRANSFORM 
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
### IMPORT 
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
### ACCURACY 
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/87d05b26-9111-4544-a830-b700bd9a5e3e)

### DATA.HEAD():

![image](https://github.com/user-attachments/assets/e2ba6705-663f-4947-8ff4-487543c5a8ff)

### DATA.TAIL():

![image](https://github.com/user-attachments/assets/010716b4-5d15-48cb-8a6c-b35b241dd530)


### DATA.INFO():

![image](https://github.com/user-attachments/assets/e54f0bc0-47f2-4b22-9deb-32098d8a93ad)

### DATA.ISNULL().SUM():
![image](https://github.com/user-attachments/assets/f35333ed-d112-472d-88bc-93145233f97b)

### Y_PREDICTION VALUE:
![image](https://github.com/user-attachments/assets/0eb284b3-06a6-4db3-b2f2-36d4b327955d)

### ACCURACY: 
![image](https://github.com/user-attachments/assets/b2afb625-ee80-48c4-b672-a4c31d1bfbf4)



## RESULT:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
