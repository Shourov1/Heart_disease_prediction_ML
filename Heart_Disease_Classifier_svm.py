# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset(Train)
dataset = pd.read_csv('Training_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Importing the dataset(Test)
dataset = pd.read_csv('Test_data.csv')
X_test = dataset.iloc[:, :].values


# Missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(copy=True, fill_value=None, missing_values= np.nan, strategy='mean', verbose=0)

imputer = imputer.fit(X[:, 0:13])
X[:, 0:13] = imputer.transform(X[:, 0:13])

#TEST
imputer_test = imputer.fit(X_test[:,:])
X_test[:,:] = imputer_test.transform(X_test[:,:])

#Feature Scaling
from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X = sc.fit_transform(X)
X_test = sc.fit_transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)
#classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

#print (y_pred)
#print ("Accuracy: ")
print("Accuracy: ", int(classifier.score(X,y)*100), "%")
