#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import pickle
import numpy as np
import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as splt
import matplotlib.pyplot as plt
import mlpy
from scipy.stats import rankdata
from mlpy.bordacount import borda_count
from bokeh.plotting import figure
from bokeh.io import curdoc, show, output_notebook
from bokeh.resources import CDN
from scipy.sparse import csr_matrix


csv_path = "/home/webvalley/score-machine-learning/Data/new_wScore.csv"
data_path = "/datadrive/random_forests_clinical_data/"
COLUMNS_BLACK_LIST = ['SCORE', 'esa_obi:sbp', 'esa_obi:dbp',
                      'ana_fis:smoking_recod', 'lab:glucose',
                      'lab:calculated_ldl',
                      'lab:total_cholesterol',
                      'ana:age']
headDic = {"lab":"lab:", "anapat": "ana_pat:", "esaobi": "esa_obi:", "ulttsa": "ult_tsa:"}

def read_data(csv_data_file):
    df = pd.read_csv(csv_data_file)
    df.sort_values("visit")
    df = df.select_dtypes(exclude=['object', 'datetime64'])
    df = df.drop(labels=COLUMNS_BLACK_LIST, axis=1)
    # Get rid of all columns with all -1 (NaN) and/or Zeros
    df = df[df.columns[df.max() > 0]]
    return df

def load_final_forests(folder_path):
    f = IntProgress(min=0, max=5, description='Loading... ', bar_style='success')
    display(f)

    final_forests = []
    for i in range(5):
        forest_path = os.path.join(folder_path, str(i), "forest_training.pkl")
        clf = pickle.load(open(forest_path, "rb"))
        final_forests.append(clf)
        f.value += 1
    return final_forests

def select_features_set(df, features_set):
    features = list(features_set) + ['subject_id', 'ScoreClass', 'visit']
    return df[df.columns[df.columns.str.startswith(tuple(features))]]

def get_data_for_visits(df, features_set, visits_map):

    target_features = select_features_set(df, features_set)

    X, y = None, None
    for X_feat_visit, y_lab_visit in visits_map.items():
        X_v = target_features[target_features['visit'] == X_feat_visit].drop(['subject_id',
                                                                              'ScoreClass', 'visit'], axis=1).values

        y_v = target_features[target_features['visit'] == y_lab_visit]['ScoreClass'].values

        if X is None:
            X = X_v
            y = y_v
        else:
            X = np.vstack((X, X_v))
            y = np.hstack((y, y_v))

    X += 1
    X_csr = csr_matrix(X)
    return X_csr, y

def random_forest_confusion_matrix(forests, features_set, visit_map={0:0,1:1,2:2,3:3}, train_test_splits=5):
    df = read_data(csv_path)
    X, y = get_data_for_visits(df, features_set, visit_map)
    S = get_stratification_array(df, visit_map)
    cmatrix = np.zeros((5, 4, 4))
    for train_test_split_run in range(train_test_splits):
        X_tr, X_ts, y_tr, y_ts, S_tr, S_ts = splt(X, y, S, test_size=0.2,
                                                  random_state=train_test_split_run, stratify=S)

        # Re-train everything from scratch on the entire training set
        y_ts_our = forests[train_test_split_run].predict(X_ts)
        for i,y_our in enumerate(y_ts_our):
            cmatrix[train_test_split_run][y_our-1][y_ts[i]-1] += 1
    
    return cmatrix

def get_relevant_columns(foldername):
    heads = []
    for key in headDic:
        if key in foldername:
            heads.append(headDic[key])
    return heads

def select_columns_set(df, features_set):
    features = list(features_set)
    return df.columns[df.columns.str.startswith(tuple(features))].values

def get_stratification_array(df, visit_map):
    # Stratification map of samples for each of the visits
    visit_strat_map = {}
    for visit_nb, group in df.groupby(["visit"]):
        visit_strat_map[visit_nb] = group.index.values

    # Stratify samples based on Score and Sex
    S = np.zeros(df.shape[0])  # all the numbers of samples
    for i, (_, group) in enumerate(df.groupby(["ScoreClass", "ana:gender"])):
        indices = group.index.values
        S[indices] = i

    return np.asarray([S[visit_strat_map[visit]] for visit in visit_map]).ravel()

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


def generate_data(exp_dir):
    f = IntProgress(min=0, max=250, description='Loading... ', bar_style='success')
    display(f)
    feature_list = get_relevant_columns(exp_dir)
    cols = select_columns_set(read_data(csv_path), feature_list)
    folder_path = os.path.join(data_path, exp_dir)
    feature_importance = np.zeros((5, 50, len(cols)))
    final_forests = []
    for i in range(5):
        forest_path = os.path.join(folder_path, str(i))
        counter = 0
        for forest_file in os.listdir(forest_path):
            if forest_file.endswith('pkl'):
                if 'training' not in forest_file:
                    forest_file_path = os.path.join(forest_path, forest_file)
                    clf = pickle.load(open(forest_file_path, "rb"))
                    feature_importance[i,counter] = clf.feature_importances_
                    counter += 1
                    f.value += 1
                else:
                    forest_file_path = os.path.join(forest_path, forest_file)
                    clf = pickle.load(open(forest_file_path, "rb"))
                    final_forests.append(clf)
    print("DONE 1")
    #print(feature_importance)
    borda_rankings = []
    for importance in feature_importance:
        borda = borda_count((np.argsort(importance)))[0]
        borda_rankings.append(borda)
    borda_rankings_np = np.array(borda_rankings)
    feature_confidence = np.zeros((5,len(cols),2))
    print("DONE 2")
    for i in range(5):
        for j in range(len(cols)):
            f_imp = feature_importance[i, :, j]
            f_min_max = bootstrap_ci(f_imp)
            f_mean = np.mean(f_min_max)
            f_error = f_min_max[1] - f_mean
            feature_confidence[i,j] = (f_mean, f_error)
    print("DONE 3")
    feature_map = None
    if "all" in exp_dir:
        feature_map = {0:0, 1:1, 2:2, 3:3}
    elif "firstthird" in exp_dir:
        feature_map = {0: 2}
    else:
        feature_map = {0: 3}
    confmat = random_forest_confusion_matrix(final_forests, feature_list, feature_map)
    np.savez("/datadrive/random_forest_final_data/" + exp_dir + "data.npz", conf = feature_confidence, cols = cols, br = borda_rankings_np, confmat = confmat)
    print("DONE 4")
    return feature_confidence, borda_rankings, cols, confmat

def generate_graphs(feature_confidence, borda_rankings, cols, exp_dir):
    for i, confidence in enumerate(feature_confidence):
        fig = figure(y_range = cols[borda_rankings[i]][-10:], plot_height=250, title= exp_dir + " Categorical Importance Run: " + str(i+1))
        fig.hbar(y = cols[borda_rankings[i]][-10:], right = confidence[:,0][borda_rankings[i]][-10:], height=0.2)
        show(fig)
