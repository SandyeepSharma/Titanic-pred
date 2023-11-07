import pandas as pd
import numpy as np


data_train = pd.read_csv(r"C:\Users\sandy\Dropbox\Part 0 - Self_practice\titanic\dataset\train.csv")
data_test = pd.read_csv(r"C:\Users\sandy\Dropbox\Part 0 - Self_practice\titanic\dataset\test.csv")
#data_test_orig = pd.read_csv(r"/home/sandeep/disk_C/csv_file/titanic_dataset/test.csv")

data_train['family'] = data_train['SibSp'] +data_train['Parch']
data_test['family'] = data_test['SibSp'] +data_test['Parch']
#data = data.dropna(axis=0, subset=['Embarked'])
data_train = data_train.iloc[:,[1,2,4,5,9,11,12]]
data_test = data_test.iloc[:,[1,3,4,8,10,11]]
X= data_train.iloc[:,1:]
y= data_train.iloc[:,0]


X.isnull().sum()
y.isnull().sum()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN" , strategy= "median", axis=0)
X.iloc[:,[2]] = imputer.fit_transform(X.iloc[:,[2]])
X.Embarked.mode
X = X.fillna('S')


imputer = Imputer(missing_values = "NaN" , strategy= "median", axis=0)
data_test.iloc[:,[2]] = imputer.fit_transform(data_test.iloc[:,[2]])
imputer = Imputer(missing_values = "NaN" , strategy= "mean", axis=0)
data_test.iloc[:,[3]] = imputer.fit_transform(data_test.iloc[:,[3]])


data = [X, data_test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
    
    
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)    

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
LE = LabelEncoder()
X.iloc[:,[1]] = LE.fit_transform(X.iloc[:,[1]])
X.iloc[:,[4]] = LE.fit_transform(X.iloc[:,[4]])

data_test.iloc[:,[1]] = LE.fit_transform(data_test.iloc[:,[1]])
data_test.iloc[:,[4]] = LE.fit_transform(data_test.iloc[:,[4]])

OHE = OneHotEncoder(categorical_features =[1])
X = OHE.fit_transform(X).toarray()
X = X[:,1:]

OHE = OneHotEncoder(categorical_features =[-2])
X = OHE.fit_transform(X).toarray()
X = X[:,1:]

OHE = OneHotEncoder(categorical_features =[1])
data_test = OHE.fit_transform(data_test).toarray()
data_test = data_test[:,1:]

OHE = OneHotEncoder(categorical_features =[-2])
data_test = OHE.fit_transform(data_test).toarray()
data_test = data_test[:,1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


#from sklearn.preprocessing import Normalizer , StandardScaler
#sc = StandardScaler()
#X_train= sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#score = 77.5

#random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 350, criterion ='entropy' , random_state= 42)
classifier.fit(X_train , y_train)
y_pred = classifier.predict(X_test)
#score = 76.9 - entropy
#score = 76.4 - gini



from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator =classifier , X = X_train , y = y_train , cv = 10)
acMean = accuracies.mean()
acStd = accuracies.std()


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {"n_estimators": [350],
              "criterion":['gini','entropy'],
              "max_depth": [1, 3, 5,7],
              "min_samples_split": [3,5, 10, 15],
              "min_samples_leaf": [2, 5, 10, 15],
              "min_weight_fraction_leaf": [0.1, 0.05, 0.005, 0.001]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'f1',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_



# Fitting Final Model on training set
from sklearn.ensemble import RandomForestClassifier
tunedRF = RandomForestClassifier(n_estimators = best_parameters["n_estimators"], 
                                    criterion = best_parameters["criterion"], 
                                    max_depth = best_parameters["max_depth"], 
                                    min_samples_split = best_parameters["min_samples_split"],  
                                    min_samples_leaf = best_parameters["min_samples_leaf"], 
                                    min_weight_fraction_leaf = best_parameters["min_weight_fraction_leaf"])
tunedRF.fit(X_train, y_train)

# Predicting the Test set results
y_pred = tunedRF.predict(X_test)



from sklearn.metrics import confusion_matrix ,accuracy_score ,classification_report
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = tunedRF, X = X_train, y = y_train, cv = 10)
acMeanTuned = accuracies.mean()
acStdTuned = accuracies.std()


##predicting for test data
y_pred_test = classifier.predict(data_test)
result = pd.Series(y_pred_test)
PId = data_test_orig.iloc[:,0]


df = PId.to_frame().join(result)

Result= pd.concat([PId,result])
path = '/home/sandeep/Dropbox/coaching_practice/titanic/result.csv'
Result = result.to_csv(path)