import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train = pd.read_csv(r"C:\Users\sandy\Dropbox\coaching_practice\titanic\train.csv")
train.head()
X = train.iloc[:,2:]
y = train.iloc[:,[1]]
test = pd.read_csv(r"C:\Users\sandy\Dropbox\coaching_practice\titanic/test.csv")
X_totest = test.iloc[:,1:]
result_sheet =test.iloc[:,[0]]
train.info()
train.describe()
X.isnull().sum()

#%age of null  values in each column
train.columns.values
train.isnull().sum()/train.isnull().count()


sns.relplot(x = 'Age',y ='Survived',col = 'Sex', data = train , kind = 'line')
sns.barplot(x='Pclass', y='Survived', data=train)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)


data = [X, X_totest]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
X['not_alone'].value_counts()

axes = sns.factorplot('relatives','Survived', 
                      data=train, aspect = 2.5, )


#######################fill cabin###################################
for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna('U0')


df_name = X.iloc[:,1]
df_name = df_name.str.extract(r'(\w+)\.')
df = pd.concat([X,df_name ], axis = 1 )


df.isnull().sum()
df['Age'].fillna(df.groupby(0)['Age'].transform('median'),inplace=True)

#Imputation

#####Converting Age and Fare into category#################

    df['Age'] = df['Age'].astype(int)
    df.loc[ df['Age'] <= 11, 'Age'] = 0
    df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'Age'] = 1
    df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2
    df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3
    df.loc[(df['Age'] > 27) & (df['Age'] <= 33), 'Age'] = 4
    df.loc[(df['Age'] > 33) & (df['Age'] <= 40), 'Age'] = 5
    df.loc[(df['Age'] > 40) & (df['Age'] <= 66), 'Age'] = 6
    df.loc[ df['Age'] > 66, 'Age'] = 6
    
    

    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare']   = 3
    df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare']   = 4
    df.loc[ df['Fare'] > 250, 'Fare'] = 5
    df['Fare'] = df['Fare'].astype(int) 
    
df.groupby(0).size()
df[0].unique()
df[0]=df[0].replace(['Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
       'Jonkheer', 'Dona'], 'unknown')

###############################Label Encoding###########################



#############Again Splitting into Train and TesT###########################


#########################################################################

from sklearn.model_selection import train_test_split
X_train1 , X_test1 , y_train1 , y_test1 = train_test_split(X_train_mod , y_train , test_size =.25)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators= 150 , criterion ='entropy')
RF.fit(X_train1,y_train1)
y_pred = RF.predict(X_test1)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RF ,X = X_train1, y=y_train1 , cv =10)
acMean = accuracies.mean()
acStd = accuracies.std()

###############################Grid Search ##########################################
from sklearn.model_selection import GridSearchCV
parameters = { "n_estimators" : [150 ],
               "criterion"    : ['entropy'],
               "max_depth": [1, 3, 5,7],
              "min_samples_split": [3,5, 10, 15],
              "min_samples_leaf": [2, 5, 10, 15],
              "min_weight_fraction_leaf": [0.1, 0.05, 0.005, 0.001]
        }
grid_search = GridSearchCV(estimator = RF, param_grid= parameters,scoring = 'f1',
                           cv =10)
grid_search = grid_search.fit(X_train1, y_train1)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

####################fitting best params#####################################
tunedRF = RandomForestClassifier(n_estimators = 150, #best_parameters["n_estimators"], 
                                    criterion = 'entropy' ,#best_parameters["criterion"], 
                                    max_depth = 5 ,#best_parameters["max_depth"], 
                                    min_samples_split = 3 ,#best_parameters["min_samples_split"],  
                                    min_samples_leaf = 2 ,#best_parameters["min_samples_leaf"], 
                                    min_weight_fraction_leaf = .005) #best_parameters["min_weight_fraction_leaf"])
tunedRF.fit(X_train1, y_train1)
# Predicting the Test set results
y_pred1 = tunedRF.predict(X_test1)

############check accuracy################################
from sklearn.model_selection import cross_val_score
accuracies1 = cross_val_score(estimator = tunedRF ,X = X_train1, y=y_train1 , cv =10)
acMean1 = accuracies.mean()
acStd1 = accuracies.std()


from sklearn.metrics import confusion_matrix ,accuracy_score ,classification_report
cm1 = confusion_matrix(y_test1, y_pred1)
ac1 = accuracy_score(y_test1, y_pred1)
print(classification_report(y_test1, y_pred1))

#########################Making file#########################################################
y_pred_kaggle = RF.predict(X_test_mod)
y_pred_kaggle =pd.Series(y_pred_kaggle)
df1 = df_test.iloc[:,0]
df1 = pd.concat([df1,y_pred_kaggle], axis = 1)
df1.to_csv('titanic.csv')

#######################Baggng and Boosting#########################################
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
rf_b = RandomForestClassifier()
svc_b = SVC(kernel ='rbf')
evc = VotingClassifier(estimators = [('rf_b',rf_b),('svc_b',svc_b)],voting ='hard')
evc.fit(X_train1,y_train1)
y_pred1 = evc.predict(X_test1)

