#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors:
    * Salvador Corts Sánchez
    * Álvaro Fernández García
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

#Constant
SEED = 8957
TRAIN_TEST_SPLIT = 0.2 
CV_SPLITS = 5           # Cross Validation splits
DATA_SUBSET = 5000      # For SVM hyperparameters estimation

MAX_TREE = 150          # Random forest
MIN_TREE = 10

MAX_ESTIMATORS = 150    # AdaBoost
MIN_ESTIMATORS = 10

##################################################################################################################

def HotEncoder(feature):
    le = LabelEncoder()
    enc = OneHotEncoder(sparse=False)
    localFeature = feature.copy()
    localFeature = le.fit_transform(localFeature)
    localFeature = localFeature.reshape(len(localFeature), 1)
    localFeature = enc.fit_transform(localFeature)

    return localFeature

##################################################################################################################

def Preprocessing(dataset):
    # Encode categorical features and cast integer to float 
    age_feature = dataset['age'].astype(np.float64).reshape([-1,1])
    job_feature = HotEncoder(dataset['job'])
    marital_feature = HotEncoder(dataset['marital'])
    education_feature = HotEncoder(dataset['education'])
    default_feature = HotEncoder(dataset['default'])
    housing_feature = HotEncoder(dataset['housing'])
    loan_feature = HotEncoder(dataset['loan'])
    contact_feature = HotEncoder(dataset['contact'])
    day_feature = HotEncoder(dataset['day_of_week'])
    month_feature = HotEncoder(dataset['month'])
    #duration_feature = dataset['duration'].astype(np.float64).reshape([-1,1])
    campaign_feature = dataset['campaign'].astype(np.float64).reshape([-1,1])
    pdays_feature = dataset['pdays'].astype(np.float64).reshape([-1,1])
    previous_feature = dataset['previous'].astype(np.float64).reshape([-1,1])
    poutcome_feature = HotEncoder(dataset['poutcome'])
    empvarrate_feature = dataset['empvarrate'].astype(np.float64).reshape([-1,1])
    conspriceidx_feature = dataset['conspriceidx'].astype(np.float64).reshape([-1,1])
    consconfidx_feature = dataset['consconfidx'].astype(np.float64).reshape([-1,1])
    euribor3m_feature = dataset['euribor3m'].astype(np.float64).reshape([-1,1])
    nremployed_feature = dataset['nremployed'].astype(np.float64).reshape([-1,1])

    # Build X dataset
    X = np.concatenate([
        # Scalar features
        age_feature,
        #duration_feature,
        campaign_feature,
        pdays_feature,
        previous_feature,
        empvarrate_feature,
        conspriceidx_feature, 
        euribor3m_feature, 
        nremployed_feature,
        consconfidx_feature,

        # Catergorical features
        job_feature,
        marital_feature,
        education_feature,
        default_feature,
        housing_feature,
        loan_feature,
        contact_feature,
        month_feature,
        day_feature,
        poutcome_feature
    ], axis=1)

    scalarLimitIndex = 8

    # Build y dataset:
    y = np.where(dataset['y'] == '"yes"', 1, -1)

    return X, y, scalarLimitIndex

##################################################################################################################

def GetBestNumberEstimators(model, x, y, min_estimators, max_estimators):
    kf = model_selection.KFold(n_splits=CV_SPLITS, random_state=SEED)
    train_score_means = []
    test_score_means = []
    
    for n in range(min_estimators, max_estimators):
        model.set_params(n_estimators=n)
        train_scores = []
        test_scores = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(x_train, y_train)

            predictions_train = model.predict(x_train)
            predictions_test = model.predict(x_test)
            train_scores.append(metrics.roc_auc_score(y_train, predictions_train))
            test_scores.append(metrics.roc_auc_score(y_test, predictions_test))

        if len(model.estimators_) < n:
            break

        train_scores_mean = np.sum(train_scores)/len(train_scores)
        test_scores_mean = np.sum(test_scores)/len(test_scores)

        train_score_means.append(train_scores_mean)
        test_score_means.append(test_scores_mean)

    return train_score_means, test_score_means
    
##################################################################################################################

def GetHyperparametersSVM(model, X, y, C, kernel, gamma, size):
    index = np.random.choice(X.shape[0], size, replace=False)
    X_reduced = X[index]
    y_reduced = y[index]
    best_comb = []

    print("Initial estimation")
    i = 1

    for k in kernel:
        for g in gamma:
            for c in C:
                model.set_params(C=c, gamma=g, kernel=k)
                score = model_selection.cross_val_score(model, X_reduced, y_reduced, scoring='roc_auc', cv=CV_SPLITS, n_jobs=-1)    
                score = score.mean()
                print("\t[{}/{}][{}][{}][{}] Score: {}".format(i, len(kernel)*len(gamma)*len(C), k, g, c, score))

                if len(best_comb) in range(0, 3):
                    best_comb.append([score, k, g, c])
                    best_comb.sort(key=lambda item: item[0], reverse=True)
                elif best_comb[-1][0] < score:
                    best_comb[-1] = [score, k, g, c]
                    best_comb.sort(key=lambda item: item[0], reverse=True)
                    best_comb = best_comb[:3]
                i+=1

    print("Three best combinations: ", best_comb)

    print("Second estimation")

    best_score = 0
    final_comb = []
    i = 1

    for hyperparams in best_comb:
        print("\t" + str(i) + " from " + str(len(best_comb)))
        model.set_params(C=hyperparams[3], gamma=hyperparams[2], kernel=hyperparams[1])
        score = model_selection.cross_val_score(model, X, y, scoring='roc_auc', cv=CV_SPLITS, n_jobs=-1)    
        score = score.mean()
        if score > best_score:
            best_score = score
            final_comb = hyperparams
        i+=1
            
    
    return final_comb

##################################################################################################################

def scoreModel(model, X, y, Type):
    predictions = model.predict(X)
    print("\n[+] " + Type + " score accuracy: ", metrics.accuracy_score(y, predictions))
    print("[+] " + Type +  " score roc_auc: ", metrics.roc_auc_score(y, predictions))
    print(metrics.confusion_matrix(y, predictions))


##################################################################################################################
# Preprocessing Data
##################################################################################################################

dataset = np.genfromtxt('datos/bank-additional-full.csv', delimiter=';', dtype=None, names=True, encoding='utf-8')
X, y, scalarLimitIndex = Preprocessing(dataset)

print("N: ", X.shape[0])
print("Number of features: ", X.shape[1])
print("Number of 'yes': ", len(y[y==1]))
print("Number of 'no': ", len(y[y==-1]))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=SEED, stratify=y)

#Scaling features
X_train_scaled = X_train.copy()
X_train_scaled[:, :scalarLimitIndex] = StandardScaler().fit_transform(X_train[:,:scalarLimitIndex])

X_test_scaled = X_test.copy()
X_test_scaled[:, :scalarLimitIndex] = StandardScaler().fit_transform(X_test[:,:scalarLimitIndex])


# Regularization hyperparameter:
C = np.logspace(-4, 4, 10)
# kflod for CV
kfold = model_selection.KFold(n_splits=CV_SPLITS)


##################################################################################################################
# Logistic Regresión
##################################################################################################################
print("\nFindig C hyperparameter for Logistic Regression...")

penalty = ['l2']
hyperparam = dict(C=C, penalty=penalty)

lr = linear_model.LogisticRegression(class_weight='balanced')
# Discomment for search hyperparameter again:
"""clf = GridSearchCV(lr, hyperparam, cv=kfold, scoring='roc_auc', n_jobs=-1)
model = clf.fit(X_train_scaled, y_train)
lr_C = model.best_estimator_.get_params()['C']"""
lr_C = 0.046415888336127774
 
print("\tBest C for Logistic Regression: ",  lr_C)

lr.set_params(C=lr_C, penalty='l2')
lr.fit(X_train_scaled, y_train)

scoreModel(lr, X_train_scaled, y_train, 'Train')

##################################################################################################################
# AdaBoost
##################################################################################################################
print("\nFinding optimal number of estimators for AdaBoost...")

ada = AdaBoostClassifier(random_state=SEED)

# Visualice error meaured by number of estimators:
# Discomment for search hyperparameter again:
"""Ein, Eval = GetBestNumberEstimators(ada, X_train, y_train, MIN_ESTIMATORS, MAX_ESTIMATORS)
plt.plot(np.arange(1, len(Ein)+1), Ein, label="Training")
plt.plot(np.arange(1, len(Ein)+1), Eval, label="Validation")
plt.title('AdaBoost estimators behaviour')
plt.xlabel('Number of estimators')
plt.ylabel('Score')
plt.legend()
plt.show()

estimators = np.arange(MIN_ESTIMATORS, MAX_ESTIMATORS)
clf = RandomizedSearchCV(estimator=ada, scoring='roc_auc', param_distributions=dict(n_estimators=estimators), n_jobs=-1, random_state=SEED, cv=kfold)
model = clf.fit(X_train, y_train)
best_number_estimators = model.best_estimator_.get_params()['n_estimators']"""
best_number_estimators = 130

print("Best number of estimators for AdaBoost: ", best_number_estimators)

ada.set_params(n_estimators=best_number_estimators)
ada.fit(X_train, y_train)

scoreModel(ada, X_train, y_train, 'Train')

##################################################################################################################
# Random Forest (RF)
##################################################################################################################
print("\nFinding optimal number of trees for Random Forest...")

rf = RandomForestClassifier(n_jobs=-1, random_state=SEED, criterion='entropy', bootstrap=True, max_features='auto', 
    class_weight='balanced', max_depth=1000, max_leaf_nodes=1000)

# Visualice error meaured by number of estimators:
# Discomment for search hyperparameter again:
"""Ein, Eval = GetBestNumberEstimators(rf, X_train, y_train, MIN_ESTIMATORS, MAX_ESTIMATORS)
plt.plot(np.arange(1, len(Ein)+1), Ein, label="Training")
plt.plot(np.arange(1, len(Ein)+1), Eval, label="Validation")
plt.title('Random Forest estimators behaviour')
plt.xlabel('Number of estimators')
plt.ylabel('Score')
plt.legend()
plt.show()

estimators = np.arange(MIN_TREE, MAX_TREE)
clf = RandomizedSearchCV(estimator=rf, scoring='roc_auc', param_distributions=dict(n_estimators=estimators), n_jobs=-1, random_state=SEED, cv=kfold)
model = clf.fit(X_train, y_train)
best_number_estimators = model.best_estimator_.get_params()['n_estimators']"""
best_number_estimators = 130

print("Best number of trees for RF: ", best_number_estimators)

rf.set_params(n_estimators=best_number_estimators)
rf.fit(X_train, y_train)

scoreModel(rf, X_train, y_train, 'Train')

##################################################################################################################
# Neural Networks (NN)
##################################################################################################################
print("\nFinding optimal topology and alpha for Neural Network...")

hiddenLayer = [(30), (30,30), (30,30,30), (30,20,10)]
activators = ["tanh", "logistic", "relu"]
hyperparam = dict(alpha=C, hidden_layer_sizes=hiddenLayer, activation=activators)

nn = MLPClassifier(random_state=SEED)

# Discomment for search hyperparameter again:
"""clf = GridSearchCV(nn, hyperparam, cv=kfold, scoring='roc_auc', n_jobs=-1)
model = clf.fit(X_train_scaled, y_train)
bestAlpha = model.best_estimator_.get_params()['alpha']
bestTopology = model.best_estimator_.get_params()['hidden_layer_sizes']
bestActivator = model.best_estimator_.get_params()['activation']"""
bestAlpha = 0.005994842503189409
bestTopology = (30)
bestActivator = 'tanh'

print("Best Alpha: ", bestAlpha, "\tBest topology: ", bestTopology, "\tBest activator: ", bestActivator)

nn.set_params(hidden_layer_sizes=bestTopology, alpha=bestAlpha, activation=bestActivator)
nn.fit(X_train_scaled, y_train)

scoreModel(nn, X_train_scaled, y_train, 'Train')


##################################################################################################################
# Support Vector Machine (SVM)
##################################################################################################################
print("\nFinding hyperparameters for SVM...")

kernels= ['rbf']
gamma = np.logspace(-2, 1, 5)

svc = SVC(cache_size=500, class_weight='balanced')

# Discomment for search hyperparameter again:
"""SVC_PARAMS = GetHyperparametersSVM(svc, X_train_scaled, y_train, C, kernels, gamma, DATA_SUBSET)
opt_kernel = SVC_PARAMS[1]
opt_gamma = SVC_PARAMS[2]
opt_c = SVC_PARAMS[3]"""
 
opt_kernel = 'rbf'
opt_gamma = 0.01
opt_c = 0.3593813663804626

print("Kernel = " + opt_kernel + " Gamma = " + str(opt_gamma) + " C = " + str(opt_c))

svc.set_params(kernel=opt_kernel, gamma=opt_gamma, C=opt_c)
svc.fit(X_train_scaled, y_train)

scoreModel(svc, X_train_scaled, y_train, 'Train')

##################################################################################################################
# BEST MODEL:
# Random Forest: Test
##################################################################################################################
print("\nBest model: RANDOM FOREST. \nTesting model...")

# Score:
predictions = rf.predict(X_test)
print("\n[+] Test score accuracy: ", metrics.accuracy_score(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))

# Plot ROC curve:
probs = rf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
print("[+] Area under ROC curve: ", roc_auc)
 
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()