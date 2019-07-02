#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pickle
import os
import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split as splt
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import RepeatedStratifiedKFold as rskf
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc
from scipy.sparse import csr_matrix 
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# In[68]:


def prep_coltype(df, headers):
    boolval = []
    headers = headers + ["subject_id", "ScoreClass", "visit"]
    for header in headers:
        boolval.append(df.columns.str.startswith(header))
    return df[df.columns[np.any(boolval, axis=0)]]
def prepare_dataset(X, y, x_ind, y_ind):
    X_, y_ = [], []
    for i in x_ind:
        X_ = X_ + X[i]
    for i in y_ind:
        y_ = y_ + y[i]
    return (X_,y_)
def bootstrap_ci(x, B=1000, alpha=0.05, seed=42):
    """Computes the (1-alpha) Bootstrap confidence interval
    from empirical bootstrap distribution of sample mean.

    The lower and upper confidence bounds are the (B*alpha/2)-th
    and B * (1-alpha/2)-th ordered means, respectively.
    For B = 1000 and alpha = 0.05 these are the 25th and 975th
    ordered means.
    """

    x_arr = np.ravel(x)

    if B < 2:
        raise ValueError("B must be >= 2")

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")

    np.random.seed(seed)

    bmean = np.empty(B, dtype=np.float)
    for b in range(B):
        idx = np.random.random_integers(0, x_arr.shape[0]-1, x_arr.shape[0])
        bmean[b] = np.mean(x_arr[idx])

    bmean.sort()
    lower = int(B * (alpha * 0.5))
    upper = int(B * (1 - (alpha * 0.5)))

    return (bmean[lower], bmean[upper])

def get_data_for_trial(df, headers, x_ind, y_ind):
    df_specific = prep_coltype(df, headers)
    d = {i: df.loc[df.subject_id == i, df.columns] for i in range(df.subject_id.iat[-1]+1)}
    d = {dx: d[dx] for dx in d if d[dx].shape[0] != 0}
    X = []
    y = []
    X1, y1, X2, y2, X3, y3, X4, y4 = [], [], [], [], [], [], [], []
    for i in range(4):
        tempX = []
        tempy = []
        for _, value in d.items():
            row = value.iloc[[i]]
            tempX.append(row.drop(['subject_id', 'visit', 'ScoreClass'], axis=1).values[0].astype(int))
            tempy.append(int(row.ScoreClass))
        X.append(tempX)
        y.append(tempy)
    X_new, y_new = prepare_dataset(X, y, x_ind, y_ind)
    X_new_np = np.array(X_new)
    y_new_np = np.array(y_new)
    X_new_np += 1
    print(X_new_np.shape)
    print(y_new_np.shape)
    X_csr = csr_matrix(X_new_np)
    return X_csr, y_new_np

def random_forest_training(X, y, S, filename):
    os.mkdir(filename)
    for i in range(5):
        logFile = open(filename + "/" + str(i) + ".txt", "w")
        X_tr, X_ts, y_tr, y_ts, S_tr, S_ts = splt(X, y, S, test_size=0.2, random_state=i, stratify = S)
        rskf_ = rskf(n_splits=5, n_repeats=10, random_state=42)
        counter = 0
        for train_index, val_index in rskf_.split(X_tr, S_tr):
            X_train, X_val = X_tr[train_index], X_tr[val_index]
            y_train, y_val = y_tr[train_index], y_tr[val_index]
            print(X_train.shape)
            print(y_train.shape)
            forest = rfc(n_estimators = 1000, max_depth = 100, n_jobs=-1)
            forest.fit(X_train, y_train)
            y_val_our = forest.predict(X_val)
            mc = mcc(y_val, y_val_our)
            ac = acc(y_val, y_val_our)
            logFile.write('{} Split {} Iteration: MCC: {}, ACC: {}'.format(i, counter, mc, ac))
            pickle.dump(forest, open(filename + "/" + str(i) + "-" + str(counter) + "forest.pkl", "wb"))
            counter = counter + 1
        forest = rfc(n_estimators = 1000, max_depth = 100, -1)
        forest.fit(X_tr, y_tr)
        y_ts_our = forest.predict(X_ts)
        mc = mcc(y_val, y_val_our)
        ac = acc(y_val, y_val_our) 
        logFile.write('Final Iteration: MCC: {}, ACC: {}'.format(mc, ac))
        pickle.dump(forest, open(filename + "/final-forest.pkl", "wb"))
        mccCI = bootstrap_ci(np.array(dataMCC[i]))
        accCI = bootstrap_ci(np.array(dataACC[i]))
        logFile.write('MCC Interval: {} - {}'.format(mccCI[0], mccCI[1]))
        logFile.write('ACC Interval: {} - {}'.format(accCI[0], accCI[1]))
        logFile.close()

def report_everything(csvFile, headers, x_ind, y_ind, filename):
    df = pd.read_csv(csvFile)
    df.sort_values("visit")
    df = df.select_dtypes(exclude=['object', 'datetime64'])
    df = df.drop(labels = ['SCORE','ana_fis:smoking_recod', 'lab:glucose', 'lab:calculated_ldl', 'lab:total_cholesterol', 'ana:age'], axis=1)
    df = df[df.columns[df.max() > 0]]
    df.head()
    groups = df.groupby(["ScoreClass", "ana:gender"])
    X = df.values
    print(X.shape)
    S = np.zeros(X.shape[0])
    for i, (_, dfGroup) in enumerate(groups):
        indicies = dfGroup.index.values
        S[indicies] = i
    
    X_, y_ = get_data_for_trial(df, headers, x_ind, y_ind)
    random_forest_training(X_, y_, S, filename)


# In[69]:


#X_1, y_1 = get_data_for_trial(df, ["lab:", "ult_tsa:"], [0, 1, 2, 3], [0, 1, 2, 3])
#X_2, y_2 = get_data_for_trial(df, ["lab:"], [0, 1, 2, 3], [0, 1, 2, 3])
#X_3, y_3 = get_data_for_trial(df, ["ana_pat:", "ult_tsa:"], [0, 1, 2, 3], [0, 1, 2, 3])
#X_4, y_4 = get_data_for_trial(df, ["ana_pat:"], [0, 1, 2, 3], [0, 1, 2, 3])
#X_5, y_5 = get_data_for_trial(df, ["lab:", "ult_tsa:", "ana_pat"], [0, 1, 2, 3], [0, 1, 2, 3])
#random_forest_training(X_1, y_1, S, "total_lab-ult_tsa")
#random_forest_training(X_2, y_2, "total_lab")
#random_forest_training(X_3, y_3, "total_ana_pat-ult_tsa")
#random_forest_training(X_4, y_4, "total_ana_pat")
#random_forest_training(X_5, y_5, "total_lab-ult_tsa-ana_pat")
report_everything("Data/new_wScore.csv", ["lab:", "ult_tsa:"], [0, 1, 2, 3], [0, 1, 2, 3], "total_lab-ult_tsa")


# In[ ]:




