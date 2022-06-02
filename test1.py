import sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import f1_score,average_precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold, GridSearchCV


import warnings
warnings.filterwarnings("ignore")

import math
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from statistics import mean
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
raw_data = pd.read_csv('Phishing.csv')
lb_enc = LabelEncoder()
raw_data["NEW_RESULT"] = lb_enc.fit_transform(raw_data["Result"])
raw_data[["Result", "NEW_RESULT"]]
df = raw_data.drop(['Result'], axis = 1)
df.head(10)
coloum = df.shape[1] - 1
data_X = df.drop(['NEW_RESULT'], axis=1)
data_y = pd.DataFrame(df['NEW_RESULT'])
y = data_y.loc[:,:].values
X = data_X.iloc[:,:].values
np.set_printoptions(threshold=sys.maxsize)
#print(y)
y = np.where(y >= 1, 1, 0)
#print(y)

import random
import pyswarms as ps


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from sklearn import linear_model

# Create an instance of the classifier
classifier = linear_model.LogisticRegression()

# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = coloum
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j


def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


# Initialize swarm, arbitrary
options = {'c1': 0.6, 'c2': 0.4, 'w':0.8, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = coloum # dimensions should be the number of features
optimizer = ps.discrete.BinaryPSO(n_particles=50, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)




# Create two instances of LogisticRegression
classfier = linear_model.LogisticRegression()

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Perform classification and store performance in P
classifier.fit(X_selected_features, y)

# Compute performance
subset_performance = (classifier.predict(X_selected_features) == y).mean()


print('Subset performance: %.3f' % (subset_performance))



def evaluation(clf, X, Y):
    print(f'Accuracy')
    acc = cross_val_score(clf, X, Y, scoring="accuracy", cv = 5)
    print(acc)
    print("Accuracy Score (Mean): ", acc.mean())
    print("Standard Error: ", acc.std())
    

    print(f'\nF1 Score')
    f1_score = cross_val_score(clf, X, Y, scoring="f1", cv = 5)
    print(f1_score)
    print("F1 Score (Mean): ", f1_score.mean())
    print("Standard Error: ", f1_score.std())
    
    print(f'\nPrecision')
    pre = cross_val_score(clf, X, Y, scoring="precision", cv = 5)
    print(pre)
    print("Precision (Mean): ", pre.mean())
    print("Standard Error: ", pre.std())
    
    print(f'\nSensitivity')
    rec = cross_val_score(clf, X, Y, scoring="recall", cv = 5)
    print(rec)
    print("Recall (Mean): ", rec.mean())
    print("Standard Error: ", rec.std())



X_train, X_test, y_train, y_test = train_test_split(X_selected_features, y, test_size=0.2, random_state=0, stratify=y)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)




def hyperParameterTuning_DecisionTree(features, labels):
    params = {
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_leaf": [3, 4, 5],
        "min_samples_split": [8, 10, 12],
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, 40, 50],
        "random_state": [10, 20, 30, 40, 50]
    }
    
    rf_model = DecisionTreeClassifier()
    
    gsearch = GridSearchCV(estimator = rf_model, param_grid = params, cv = 5, n_jobs = -1, verbose = 1)
    
    gsearch.fit(features,labels)
    
    return gsearch.best_params_




hyperParameterTuning_DecisionTree(X_train, y_train)
clf_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 30, max_features = 'auto',
                                      min_samples_leaf = 4, min_samples_split = 8, random_state = 30)
evaluation(clf_tree, X_test, y_test)



def hyperParameterTuning_KNN(features, labels):
    params = {
        "n_neighbors": [3, 5, 8, 10, 13],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"],
        "metric": ["minkowski", "euclidean", "manhattan"],
        "p": [1, 2, 3, 4, 5]
    }
    
    rf_model = KNeighborsClassifier()
    
    gsearch = GridSearchCV(estimator = rf_model, param_grid = params, cv = 5, n_jobs = -1, verbose = 1)
    
    gsearch.fit(features,labels)
    
    return gsearch.best_params_


hyperParameterTuning_KNN(X_train, y_train)
clf_KNN = KNeighborsClassifier(algorithm = "kd_tree", metric = "minkowski", n_neighbors = 3, p = 1, weights = "distance")
evaluation(clf_KNN, X_test, y_test)


def hyperParameterTuning_RF(features, labels):
    params = {
        "n_estimators": [30, 50, 70, 100],
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, 40, 50],
        "random_state": [10, 20, 30, 40, 50]
    }
    
    rf_model = RandomForestClassifier()
    
    gsearch = GridSearchCV(estimator = rf_model, param_grid = params, cv = 5, n_jobs = -1, verbose = 1)
    
    gsearch.fit(features,labels)
    
    return gsearch.best_params_

hyperParameterTuning_RF(X_train, y_train)
clf_rf = RandomForestClassifier(criterion = 'entropy', max_depth = 20, n_estimators = 70, random_state = 40)
evaluation(clf_rf, X_test, y_test)



def hyperParameterTuning_SVC(features, labels):
    params = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [1, 3, 5],
        "random_state": [10, 20, 30, 40, 50]
    }
    
    rf_model = SVC()
    
    gsearch = GridSearchCV(estimator = rf_model, param_grid = params, cv = 5, n_jobs = -1, verbose = 1)
    
    gsearch.fit(features,labels)
    
    return gsearch.best_params_


hyperParameterTuning_SVC(X_train, y_train)
clf_svm = SVC(random_state = 10, kernel='poly', C = 5)
evaluation(clf_svm, X_test, y_test)



def hyperParameterTuning_MLP(features, labels):
    params = {
        "hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
        "activation": ['tanh', 'relu'],
        "max_iter": [500, 600, 800, 1000],
        "learning_rate": ['constant','adaptive', "invscaling"],
        "random_state": [10, 20, 30, 40, 50]
    }
    
    rf_model = MLPClassifier()
    
    gsearch = GridSearchCV(estimator = rf_model, param_grid = params, cv = 5, n_jobs = -1, verbose = 1)
    
    gsearch.fit(features,labels)
    
    return gsearch.best_params_


hyperParameterTuning_MLP(X_train, y_train)
clf_mlp = MLPClassifier(activation = "relu", hidden_layer_sizes = (50, 100, 50), learning_rate = "constant", 
                        max_iter = 500, random_state = 10) 
evaluation(clf_mlp, X_test, y_test)




