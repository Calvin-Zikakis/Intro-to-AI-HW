import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pandas import DataFrame

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
    df = pd.read_csv(file)
    return df

def trainOnAllData(df, clf):
    #Use this function for part 4, once you have selected the best model
    Label = "Credit"
    Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

    X_train, y_train = df[Features], df[Label]
    
    clf.fit(X_train, y_train)

    saveBestModel(clf)

df = readData("credit_train.csv")

print(df)


classifiers = {"Logistic Regression": LogisticRegression(),
            "Naive Bayes": GaussianNB(),
            "SVM": svm.SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(),
            "Ada Boost": AdaBoostClassifier()}

Scores = {}
#dic for scores


for name, clf in zip(classifiers.keys(), classifiers.values()):
    #loop through the different classifiers

    inner_dict = {}
    #used for storing ROC scores

    X, y_1 = df[Features], df[Label]
    X = np.array(X)
    y_1 = np.array(y_1)
    #grab dataset features and labels

    y = np.array([])
    
    skf = StratifiedKFold(n_splits=10)
    #split the data with k = 10


    total = 0
    interation = 0
    average = 0
    #average varible to use across folds

    total_AUC = 0
    average_AUC = 0
    #average varible to use across folds

    print(name)

    for i in y_1:
    #switching "good"  and "bad" tags to ones and zeros for AUC scores
        if i == "good":
            #good = 1
            y = np.append(y, 1)
        else:
            y = np.append(y, 0)
            # bad = 0
    
    std_array = np.array([])

    for train_index, test_index in skf.split(X,y):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        #score the fitment of the classifier

        y_score = clf.predict(X_test)

        #fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

        roc_score = metrics.roc_auc_score(y_test, y_score)

        std_array = np.append(std_array, roc_score)

        total_AUC = total_AUC + roc_score

        total = total + score
        interation += 1

    average_AUC = total_AUC / interation
    average = total / interation

    inner_dict["Average Score"] = average
    inner_dict["Average AUROC"] = average_AUC
    inner_dict["STD"] = np.std(std_array)

    Scores[name] = inner_dict


new_df = pd.DataFrame(Scores)
print(new_df)

#Part B

classifiers = {
            "SVM": svm.SVC(),
            "Random Forest": RandomForestClassifier()}


X, y_1 = df[Features], df[Label]
X = np.array(X)
y_1 = np.array(y_1)
#grab dataset features and labels

y = np.array([])


print("SVM")

for i in y_1:
#switching "good"  and "bad" tags to ones and zeros for AUC scores
    if i == "good":
        #good = 1
        y = np.append(y, 1)
    else:
        y = np.append(y, 0)
        # bad = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)


Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
#found online as best values to use. 

param_grid_SVM = {'C': Cs, 'gamma' : gammas}

grid_search = GridSearchCV(svm.SVC(),param_grid_SVM, cv=10, scoring = 'roc_auc')

grid_search.fit(X,y)
print(grid_search.best_params_)



SVC_clf = svm.SVC( C=0.001, gamma=0.001)
SVC_clf.fit(X_train, y_train)
score_SVC = SVC_clf.score(X_test, y_test)
 #score the fitment of the classifier

print(score_SVC)



X, y_1 = df[Features], df[Label]
X = np.array(X)
y_1 = np.array(y_1)
#grab dataset features and labels

y = np.array([])


print("Random Forest")

for i in y_1:
#switching "good"  and "bad" tags to ones and zeros for AUC scores
    if i == "good":
        #good = 1
        y = np.append(y, 1)
    else:
        y = np.append(y, 0)
        # bad = 0


param_grid_Random_Forest = {
    'max_depth': [5, 10, 20, 30],
    'n_estimators': [1, 50, 100, 200]}


#grid_search_RF = GridSearchCV(RandomForestClassifier(),param_grid_Random_Forest, cv=10, scoring = 'roc_auc')
#grid_search_RF.fit(X,y)
#print(grid_search_RF.best_params_)
# dont want these on when i run this everytime. they take foreverrrerrrr



X, y_1 = df[Features], df[Label]
X = np.array(X)
y_1 = np.array(y_1)
#grab dataset features and labels
y = np.array([])
skf = StratifiedKFold(n_splits=10)
#split the data with k = 10
for i in y_1:
#switching "good"  and "bad" tags to ones and zeros for AUC scores
    if i == "good":
        #good = 1
        y = np.append(y, 1)
    else:
        y = np.append(y, 0)
        # bad = 0

total = 0
interation = 0
average = 0
#average varible to use across folds

total_AUC = 0
average_AUC = 0
#average varible to use across folds

std_array = np.array([])

y_predicted = np.array([])

for train_index, test_index in skf.split(X,y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    
    RandomForest = RandomForestClassifier(max_depth=10, n_estimators=100)
    RandomForest.fit(X_train, y_train)
    score_RF = RandomForest.score(X_test, y_test)
    y_score = RandomForest.predict(X_test)

    #fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

    roc_score = metrics.roc_auc_score(y_test, y_score)

    std_array = np.append(std_array, roc_score)

    total_AUC = total_AUC + roc_score

    total = total + score

    print(confusion_matrix(y_test, y_score)," <-- Confusion Matrix")
    print(metrics.accuracy_score(y_test, y_score), " <-- Accuracy Score")
    print(metrics.precision_score(y_test, y_score), " <-- Percision Score")
    print(metrics.recall_score(y_test, y_score), " <-- Recall Score")

    y_predicted = np.append(y_predicted, y_score)

    interation += 1






average_AUC = total_AUC / interation
average = total / interation

print(average_AUC, " <-- Average AUC")


new_df_C = pd.DataFrame(df)

new_y = np.array([])
for i in y_predicted:
    if i == 1:
        #good = 1
        new_y = np.append(new_y, "good")
    else:
        new_y = np.append(new_y, "bad")
        # bad = 0


new_df_C['Predicted'] = new_y


export_csv = new_df_C.to_csv("bestModel.output")



## Part D - Best Model



trainOnAllData(df, RandomForestClassifier(max_depth=10, n_estimators=100))

