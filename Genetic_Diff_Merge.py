#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pickle
import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as splt
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import RepeatedStratifiedKFold as rskf
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc
from scipy.sparse import csr_matrix
from typing import Tuple, Dict


# In[32]:


# =========
# CONSTANTS
# =========
gen_path = "/home/webvalley/score-machine-learning/DBPLIC1_wScoreCLass.xlsx"
csv_path = "/home/webvalley/score-machine-learning/Data/Milano_with_score.csv"
DEFAULT_FEATURES_SET = ['lab:', 'ana_pat:', 'esa_obi:']
COMPLETE_FEATURES_SET = ['ana_fis:', 'ana_pat:', 'ana_far:', 'esa_obi:',
                         'lab:', 'end:', 'lun_bod_sca:', 'eco_art:']
CV_REPEATS = 10
CV_NSPLITS = 5
SCORE_COLS_BLACKLIST = ['esa_obi:sbp', 'esa_obi:dbp',
                      'ana_fis:smoking_recod', 'lab:glucose',
                      'lab:calculated_ldl',
                      'lab:total_cholesterol',
                      'ana:age']
TRAIN_TEST_SPLIT_RUN = 5


# In[58]:


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
        idx = np.random.random_integers(0, x_arr.shape[0] - 1, x_arr.shape[0])
        bmean[b] = np.mean(x_arr[idx])

    bmean.sort()
    lower = int(B * (alpha * 0.5))
    upper = int(B * (1 - (alpha * 0.5)))

    return (bmean[lower], bmean[upper])


def read_data(csv_data_file):
    df = pd.read_csv(csv_data_file)
    df.set_index('patient_id', inplace=True)
    df.sort_values("visit:visit")
    return df


def filter_data(df, ):
    """
    """
    df = df.select_dtypes(exclude=['object', 'datetime64'])
    df = df.drop(labels=SCORE_COLS_BLACKLIST, axis=1)
    # Get rid of all columns with all -1 (NaN) and/or Zeros
    df = df[df.columns[df.max() > 0]]
    df = df[df.columns[df.var() > 0.1]]
    df = df[df.columns[df.median() != -1]]
    return df

def select_features_set(df, features_set):
    features = list(features_set) + ['patient_id', 'ScoreClass', 'visit:visit']
    return df[df.columns[df.columns.str.startswith(tuple(features))]]


# In[64]:


def get_data_for_visits(df, features_set, visits_map):
    """
    """

    target_features = df #select_features_set(df, features_set)
    #***Uncomment if using genetic features***
    #target_features = fuse_genetic(target_features, load_genetic_data(gen_path))
    
    
    X, y = [], None
    y = target_features[target_features['visit:visit'] == 3]['ScoreClass'].values
    X.append(filter_data(target_features[target_features['visit:visit'] == 0].drop(['ScoreClass', 'visit:visit'], axis=1)))
    X.append(filter_data(target_features[target_features['visit:visit'] == 1].drop(['ScoreClass', 'visit:visit'], axis=1)))

    # Convert Dense Matrix to a Sparse one
    # CSR_Sparse considers "0" as the empty value - not -1 as in the dataset
    # so we sum +1 to all the values
    X[0] = X[0].select_dtypes(exclude=['int'])
    X[1] = X[1].select_dtypes(exclude=['int'])
    
    X_feature_names = list(X[0].columns.values)
    X_features = list(X[0].values.transpose())
    print("DEBUG: ", X[0].values.shape)
    X_common_features = set(X[0].columns.values).intersection(X[1].columns.values)
    for feature in X_common_features:
            if "score" in feature:
                continue
            X_feature_names.append(feature)
            X_features.append(X[1][feature].values - X[0][feature].values)
    X_np = np.array(X_features)
    X_np += 1
    X_np = X_np.transpose()
    print(X_np)
    print(type(X_np))
    print(X_np.shape)
    X_csr = csr_matrix(X_np)

    # CROSSCHECK
    # ----------
    print('Dataset Shapes CrossCheck: ')
    print('X shape: ', X_csr.shape)
    print('y shape: ', y.shape)

    return X_csr, y


def get_stratification_array(df, visit_map):
    # Stratification map of samples for each of the visits
    visit_strat_map = {}
    for visit_nb, group in df.groupby(["visit:visit"]):
        visit_strat_map[visit_nb] = group.index.values

    # Stratify samples based on Score and Sex
    S = np.zeros(df.shape[0])  # all the numbers of samples
    for i, (_, group) in enumerate(df.groupby(["ScoreClass", "ana:gender"])):
        indices = group.index.values
        S[indices] = i
    return S
    #return np.asarray([S[visit_strat_map[visit]] for visit in visit_map]).ravel()


def random_forest_training(X, y, stratify_array, experiment_folder_path,
                           train_test_splits=TRAIN_TEST_SPLIT_RUN,
                           cv_nsplits=CV_NSPLITS, cv_repeats=CV_REPEATS):
    """"""

    for train_test_split_run in range(train_test_splits):
        mcc_scores = []
        acc_scores = []

        # Create the folder for the current experiment
        train_test_run_folder_path = os.path.join(experiment_folder_path, '{}'.format(train_test_split_run))
        os.makedirs(train_test_run_folder_path, exist_ok=True)

        feat_rankings_folder = os.path.join(train_test_run_folder_path, 'features_importance')
        os.makedirs(feat_rankings_folder, exist_ok=True)

        X_tr, X_ts, y_tr, y_ts, S_tr, S_ts = splt(X, y, stratify_array, test_size=0.2,
                                                  random_state=train_test_split_run, stratify=stratify_array)

        print('Experiment {} out of {} ...'.format(train_test_split_run + 1, train_test_splits), end=' ')

        rskf_ = rskf(n_splits=cv_nsplits, n_repeats=cv_repeats, random_state=42)
        cv_exp_number = 1
        for train_index, val_index in rskf_.split(X_tr, S_tr):
            X_train, X_val = X_tr[train_index], X_tr[val_index]
            y_train, y_val = y_tr[train_index], y_tr[val_index]
            forest = rfc(n_estimators=1000, n_jobs=-1)
            forest.fit(X_train, y_train)
            y_pred_val = forest.predict(X_val)
            mc = mcc(y_val, y_pred_val)
            ac = acc(y_val, y_pred_val)
            mcc_scores.append(mc)
            acc_scores.append(ac)

            # Save Feature ranking
            np.savez(os.path.join(feat_rankings_folder, 'feat_ranking_{:02d}.npz'.format(cv_exp_number)),
                     ranking=forest.feature_importances_)


            rf_pickle_filepath = os.path.join(train_test_run_folder_path, 'forest_{:02d}.pkl'.format(cv_exp_number))
            with open(rf_pickle_filepath, 'wb') as pickle_file:
                pickle.dump(forest, pickle_file)
            cv_exp_number += 1

        # Re-train everything from scratch on the entire training set
        forest = rfc(n_estimators=1000, n_jobs=-1)
        forest.fit(X_tr, y_tr)
        y_ts_our = forest.predict(X_ts)

        mc = mcc(y_ts, y_ts_our)
        ac = acc(y_ts, y_ts_our)

        rf_pickle_filepath = os.path.join(train_test_run_folder_path, 'forest_training.pkl'.format(cv_exp_number))
        with open(rf_pickle_filepath, 'wb') as pickle_file:
            pickle.dump(forest, pickle_file)

        # Store the logs for this experiment

        log_file_path = os.path.join(train_test_run_folder_path, 'log.csv')
        mcc_ci_min, mcc_ci_max = bootstrap_ci(np.asarray(mcc_scores))
        acc_ci_min, acc_ci_max = bootstrap_ci(np.asarray(acc_scores))

        scores = pd.DataFrame({'ACC': np.mean(acc_scores),
                               'ACC_CI_MIN': acc_ci_min, 'ACC_CI_MAX': acc_ci_max,
                               'MCC': np.mean(mcc_scores),
                               'MCC_CI_MIN': mcc_ci_min, 'MCC_CI_MAX': mcc_ci_max,
                               'ACC_TEST': ac, 'MCC_TEST': mc}, index=[0])
        scores.to_csv(log_file_path, sep=',')
        print('Done')


# In[65]:


def run_experiment(csv_data_file: str, features_set: Tuple,
                   visit_map: Dict, exp_log_folder_path: str):
    # visit_map: {0: 0, 1:1...} OR {0: 3}...

    # Create the folder in which logs will be saved
    os.makedirs(exp_log_folder_path, exist_ok=True)

    df = read_data(csv_data_file)
    X, y = get_data_for_visits(df, features_set, visit_map)

    # Stratify based on Score class and Gender
    S = get_stratification_array(df, visit_map)
    random_forest_training(X, y, S[0:1445], exp_log_folder_path)


# In[ ]:


def load_genetic_data(DATA_DIR):
    genetic_data = pd.read_excel(DATA_DIR)
    genetic_data = genetic_data.dropna(subset=["ScoreClass"], axis=0)
    #genetic_data = genetic_data.rename(columns={"ScoreClass":"GeneticScoreClass"})
    genetic_data = genetic_data.drop(labels=['Unnamed: 0','Score','ScoreClass'],axis=1)
    #Clean out binned data
    genetic_columns = list(genetic_data.columns)
    todrop_genetic_columns = []
    counter = 0
    for index, value in enumerate(genetic_columns):
        if ('VS') in value:
            todrop_genetic_columns.append(value)
            counter += 1
        elif('vs' in value):
            todrop_genetic_columns.append(value)
            counter += 1
        elif('1_2e3' in value):
            todrop_genetic_columns.append(value)
            counter += 1

    final_data = genetic_data.drop(labels=todrop_genetic_columns,axis=1)
    final_data.apply(pd.to_numeric)
    final_data = final_data.fillna(value=-1)
    return final_data

def fuse_genetic(df_reg, df_gen):
    d = {i: df_reg.loc[df_reg.patient_id == i, df_reg.columns] for i in range(df_reg.patient_id.iat[-1]+1)}
    d = {dx: d[dx] for dx in d if d[dx].shape[0] != 0}
    dd = {}
    newDfDf = pd.DataFrame()
    for row in df_gen.iterrows():
        row = row[1]
        subid = row['cod_pz']
        if subid in d:
            newDf = pd.DataFrame()
            for index, series in d[subid].iterrows():
                tempSeries = series.append(row)
                score = tempSeries.score
                if score < 1:
                    scoreClass = 0
                elif score < 2:
                    scoreClass = 1
                elif score < 5:
                    scoreClass = 2
                else:
                    scoreClass = 3
                tempSeries['ScoreClass'] = scoreClass
                newDf = newDf.append(tempSeries, ignore_index=True)
            newDfDf = newDfDf.append(newDf, ignore_index=True)
    return newDfDf

