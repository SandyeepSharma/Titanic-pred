import pandas as pd
import numpy as np


data = pd.read_csv(r"C:/Users/sandy/Dropbox/data/titanic/train.csv")
data = data.dropna(axis=0, subset=['Embarked'])
data = data.iloc[:,[1,2,4,5,6,7,9,11]]
X= data.iloc[:,1:]
y= data.iloc[:,0]

X.isnull().sum()
y.isnull().sum()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN" , strategy= "mean", axis=0)
X.iloc[:,2:-1] = imputer.fit_transform(X.iloc[:,2:-1])

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
LE = LabelEncoder()
X.iloc[:,[1]] = LE.fit_transform(X.iloc[:,[1]])
X.iloc[:,[-1]] = LE.fit_transform(X.iloc[:,[-1]])

OHE = OneHotEncoder(categorical_features =[1])
X = OHE.fit_transform(X).toarray()
X = X[:,1:]

OHE = OneHotEncoder(categorical_features =[-1])
X = OHE.fit_transform(X).toarray()
X = X[:,1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


from sklearn.preprocessing import Normalizer , StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#score = 77.5

#random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 350, criterion ='gini' , random_state= 42)
classifier.fit(X_train , y_train)
y_pred = classifier.predict(X_test)
#score = 76.2 - entropy
#score = 74.2 - gini


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)
y_pred =classifier.predict(X_test)
#score = 75.8


from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#score = 80.8(rbf)
#score = 79.8(linear)
#score = 75.8(sigmoid)
#score = 82.0(poly)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10 , metric = 'manhattan')
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
#score = 79.0(poly)

from sklearn.metrics import confusion_matrix ,accuracy_score ,classification_report
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
