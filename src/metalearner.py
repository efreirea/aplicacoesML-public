from lib.utils import ErrorMetricCalculator
import argparse
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from skfeature.function.statistical_based import CFS
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold

from sklearn.svm import  SVC

random_state=42

def preprocess(data,igCol):
    data_preproc = data.copy(deep=True)
    data_preproc['len'] = data_preproc['len'].astype(float)
    data_preproc['max_fft_imag'] = data_preproc['max_fft_imag'].astype(float)
    # data_preproc.dropna(subset=['Winner','Community_0','Community_1','Community_2'],inplace=True)
    data_preproc.dropna(subset=['Winner'],inplace=True)

    if igCol is not None and len(igCol)>0:
        data_preproc.drop(columns=igCol,inplace=True)

    le = LabelEncoder()
    data_preproc['Winner'] = le.fit_transform(data_preproc['Winner'])
    cat_columns = ['sector'] if igCol is None or 'sector' not in igCol else []
    for i in range(7): ## GAMBIARRA. DEIXAR MAIs ORG DEPOIS
        col_name='Community_{}'.format(i)
        if col_name in data_preproc.columns:
            print("================COMMUNITY COLUMN DETECTED=================")
            data_preproc[col_name] = data_preproc[col_name].astype(str)
            data_preproc.fillna(value={col_name: 'NA'},inplace=True)
            cat_columns.append(col_name)
    data_preproc = pd.get_dummies(data_preproc, prefix_sep="__", columns=cat_columns,dtype=float)

    var_thresh = VarianceThreshold()
    var_thresh.fit(data_preproc)
    data_preproc = data_preproc.iloc[:,var_thresh.get_support()]

    return data_preproc

def save_single_fig(performance_metrics,metric_name,model_names,baseline=0.90,filename=""):
    list_of_tuples = list(zip(*performance_metrics))
    df = pd.DataFrame(list_of_tuples, columns = model_names)

    plt.figure()

    ax = sns.boxplot(data=df, order=model_names)
    ax.axhline(baseline, ls='--',label="Baseline: Stratified Class",color='r')
    ax.set(xlabel='Models', ylabel=metric_name)
    ax.figure.savefig("{}_{}.eps".format(filename,metric_name))
    plt.legend()


def save_figs(model_results,metric_names,metric_baselines,filename):
    sns.set(style="whitegrid")

    models_names = model_results.keys()
    # List of tuples of the form (metric_array_1,metric_array_2,...)
    performance_results = model_results.values()

    for metric_name, baseline, *mx in zip(metric_names,metric_baselines,*performance_results):
        print(metric_name)
        print(mx)
        save_single_fig(mx,metric_name,models_names,baseline=baseline,filename=filename)
        print("================================================")

def calculate_baselines(X,Y):
    baselines = []
    #Using majority class as baseline just for accuracy
    dummy_model = DummyClassifier(strategy='most_frequent').fit(X,Y)
    y_pred = dummy_model.predict(X)
    baselines.append(metrics.accuracy_score(Y, y_pred))

    #Using stratfied random rediction as baseline just for the other metrics
    dummy_model = DummyClassifier(strategy='stratified').fit(X,Y)
    y_pred = dummy_model.predict(X)
    baselines.append(metrics.cohen_kappa_score(Y, y_pred))
    baselines.append(metrics.f1_score(Y, y_pred))
    baselines.append(metrics.recall_score(Y, y_pred))
    return baselines


def hyperparam_opt_execution_inner(X,Y,learner_instance,parameter_space,workers=-1,cv=5):
    clf = GridSearchCV(learner_instance, parameter_space, n_jobs=workers, cv=cv,iid=False)
    clf.fit(X,Y)
    return clf

def train(x_train,y_train,filename,fs):
    selected_idx = feature_selection(x_train,y_train,fs)
    print("Selected Features #: {}".format(len(selected_idx)))
    print("Selected Features : {}".format(x_train.columns[selected_idx]))

    x_train = x_train.iloc[:, selected_idx].copy(deep=True)

    scaling_model = MinMaxScaler()
    scaling_model = scaling_model.fit(x_train)
    x_train = scaling_model.transform(x_train)

    parameter_space_dt = {
        'min_samples_split': [2,4,8, 16, 32, 64],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'random_state': [random_state]
    }
    parameter_space_knn = {
        'n_neighbors': [1,3,5,7,9,11,15],
        'p': [1, 2],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree','kd_tree','brute'],
    }

    parameter_space_svc = {
        'C': [0.125,0.5,2,8],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    }

    clf_dt = hyperparam_opt_execution_inner(x_train,y_train,DecisionTreeClassifier(),parameter_space_dt,workers=-1,cv=LeaveOneOut())
    # clf_dt = hyperparam_opt_execution_inner(x_train,y_train,DecisionTreeClassifier(random_state=random_state),parameter_space_dt,workers=-1,cv=StratifiedKFold(n_splits=10,random_state=random_state))
    print(' DT - Best parameters found:\n', clf_dt.best_params_)

    clf_knn = hyperparam_opt_execution_inner(x_train,y_train,KNeighborsClassifier(),parameter_space_knn,workers=-1,cv=LeaveOneOut())
    # clf_knn = hyperparam_opt_execution_inner(x_train,y_train,KNeighborsClassifier(),parameter_space_knn,workers=-1,cv=StratifiedKFold(n_splits=10,random_state=random_state))
    print(' KNN - Best parameters found:\n', clf_knn.best_params_)

    clf_svc = hyperparam_opt_execution_inner(x_train,y_train,SVC(gamma='scale'),parameter_space_svc,workers=-1,cv=LeaveOneOut())
    # clf_svc = hyperparam_opt_execution_inner(x_train,y_train,SVC(gamma='scale'),parameter_space_svc,workers=-1,cv=StratifiedKFold(n_splits=10,random_state=random_state))
    print(' SVC - Best parameters found:\n', clf_svc.best_params_)

    return selected_idx,{'DT':clf_dt,'KNN':clf_knn,'svc':clf_svc},scaling_model

def test(x_test,y_test,learning_models,scaling_model,selected_features):

    x_test = x_test.iloc[:, selected_features].copy(deep=True)
    x_test = scaling_model.transform(x_test)

    print("====Baselines=======")
    baselines = calculate_baselines(x_test,y_test)
    print(baselines)
    for learning_model_name,learning_model in learning_models.items():
        print("=========={}=============".format(learning_model_name))
        y_true, y_pred = y_test, learning_model.predict(x_test)
        print('Results on the test set:')
        print(metrics.classification_report(y_true, y_pred))
        print("Accuracy score", metrics.accuracy_score(y_test, y_pred))
def feature_selection(X,Y,fs):
    X = X.values if type(X) == pd.DataFrame else X
    Y = Y.values if type(Y) == pd.Series else Y
    # return [13, 16, 17, 18, 19, 21]
    # return [3,  4,  5,  6,  8, 10, 11, 13, 14, 15, 16, 17, 21, 23, 25, 27]

    if fs=='cfs':
        criteria = CFS.cfs(X,Y)
    elif fs=='rfe':
        model = LogisticRegression(solver='lbfgs',random_state=random_state,max_iter=4000)
        rfecv = RFECV(estimator=model, cv=StratifiedKFold(n_splits=5,random_state=random_state))
        rfecv.fit(X, Y)
        criteria = np.argwhere(rfecv.support_).ravel()
    elif fs=='nofs':
        criteria = range(X.shape[1])
    else:
        raise ValueError("Invalid feature selection method: {}".format(fs))
    # print(rfecv.support_)
    print(criteria)
    return criteria
def run(full_path,igCol,fs):
    filename = os.path.basename(full_path)
    data = pd.read_csv(full_path,index_col=0)
    data_preproc = preprocess(data,igCol)
    # X = data_preproc.drop(columns=['CH','GCH','Winner'])
    # X = data_preproc.drop(columns=['RW','AR','HL','Winner'])
    X = data_preproc.drop(columns=['RW','GCH','Winner'])
    Y = data_preproc.Winner
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,stratify=Y,random_state=42)
    print("Train Shape")
    print(x_train.shape)
    print("Test Shape")
    print(x_test.shape)
    selected_features, learning_models, scaling_model = train(x_train,y_train,filename,fs)
    test(x_test,y_test,learning_models,scaling_model,selected_features)

if __name__=='__main__':
    np.random.seed(1)

    parser = argparse.ArgumentParser(description='Execute meta learning step')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f',help='Metabase')
    parser.add_argument('-igCol',action='append',help='Drop column before processing')
    parser.add_argument('-fs',help='Feature selection method',required=True,choices=['cfs','rfe','nofs'])
    args = parser.parse_args()
    print(args)
    if args.f is not None:
        run(args.f,args.igCol,args.fs)
