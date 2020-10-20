import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

images=[]
labels=[]

# train = 80% test=20%
TEST_SIZE = 0.2

'''
in my case i had some csv file with name : RESULT0, RESYLT1, etc
You can overwrite the loop to fit your filename
'''

for i in range(1,26):
   with open('RESULT%i.csv'%i,) as csvfile:
     data = csv.reader(csvfile)
     header = next(data)
     for j in data:
        if header != None :
          # chose the columns to be considered
          images.append(j[3:139])
          labels.append(j[139])
    
print(labels)         
print(len(labels))

images = np.array(images, dtype=np.float64)

knn_results=0
svm_results=0

x_train, x_test, y_train, y_test = tts(images, labels, test_size=TEST_SIZE)


# this loop now calculates the result of KNN and SVM 1000 times
for i in range(1000):    
    x_train, x_test, y_train, y_test = tts(images, labels, test_size=TEST_SIZE)
    
    clf = KNN() 
    clf2 = SVC()
    
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    knn_results += accuracy
    
    clf2.fit(x_train,y_train)
    pred2 = clf2.predict(x_test)
    accuracy2 = accuracy_score(y_test, pred2)
    svm_results += accuracy2

# average of the accuracy for KNN and SVM
print(knn_results/1000)
print(svm_results/1000)