#!/usr/bin/env python
# coding: utf-8

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

# =========
# CONSTANTS
# =========
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
    df.set_index('subject_id', inplace=True)
    df.sort_values("visit")
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
    features = list(features_set) + ['subject_id', 'ScoreClass', 'visit']
    return df[df.columns[df.columns.str.startswith(tuple(features))]]


def get_data_for_visits(df, features_set, visits_map):
    """
    """

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

    # Convert Dense Matrix to a Sparse one
    # CSR_Sparse considers "0" as the empty value - not -1 as in the dataset
    # so we sum +1 to all the values
    X += 1
    X_csr = csr_matrix(X)

    # CROSSCHECK
    # ----------
    print('Dataset Shapes CrossCheck: ')
    print('X shape: ', X_csr.shape)
    print('y shape: ', y.shape)

    return X_csr, y


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


def run_experiment(csv_data_file: str, features_set: Tuple,
                   visit_map: Dict, exp_log_folder_path: str):
    # visit_map: {0: 0, 1:1...} OR {0: 3}...

    # Create the folder in which logs will be saved
    os.makedirs(exp_log_folder_path, exist_ok=True)

    df = read_data(csv_data_file)
    X, y = get_data_for_visits(df, features_set, visit_map)

    # Stratify based on Score class and Gender
    S = get_stratification_array(df, visit_map)
    random_forest_training(X, y, S, exp_log_folder_path)


if __name__ == '__main__':

    import json
    from datetime import datetime

    parser = ArgumentParser("Random Forest Training with CV (DAP) using selected Clinical Features")

    parser.add_argument('--data', '-d', dest='datafile', default='Data/new_wScore.csv',
                        help='Path to the Dataset in CSV format')

    parser.add_argument('--features', '-f', nargs='+', dest='features_set', default=None,
                        help='List of features sets to include in the analysis')

    parser.add_argument('--all-features', action='store_true', dest='all_features',
                        help='Include all features (but ultrasound) in the feature set')

    parser.add_argument('--with-genetics', action='store_true', dest='with_genetics',
                        help='Include Genetics Features (NOT IMPLEMENTED YET)')

    parser.add_argument('--with-ultrasound', action='store_true', dest='with_us',
                        help="Include or Not UltraSound Features")

    parser.add_argument('--visits-map', '-v', default='all', dest='visits_map', help='Map of the Visit data')

    parser.add_argument('--output', '-o', dest='output_folder', default='',
                        help="Path to the destination folder where logs of experiments will be saved.")

    args = parser.parse_args()

    if args.features_set and any(f not in DEFAULT_FEATURES_SET for f in args.features_set):
        raise ValueError('Invalid Feature set provided')

    if args.with_genetics:
        raise NotImplementedError('Inclusion of Genetics Features not yet available')

    if args.visits_map == 'all':
        visits_map_mnemonic = 'all'  # used for output folder name, if None is specified
        args.visits_map = {0: 0, 1: 1, 2: 2, 3: 3}
    else:
        args.visits_map = json.loads(args.visits_map, object_hook=lambda d: {int(k): int(v) for k, v in d.items()})
        visits_labels = {0: 'first', 1: 'second', 2: 'third', 3: 'fourth'}
        visits_map_mnemonic = '_'.join('{}_{}'.format(visits_labels[k], visits_labels[v])
                                       for k, v in args.visits_map.items())

    if not args.features_set:
        if args.all_features:
            args.features_set = COMPLETE_FEATURES_SET
        else:
            args.features_set = DEFAULT_FEATURES_SET

    if args.with_us:
        args.features_set += ['ult_tsa:']

    if not args.output_folder:
        args.output_folder = '{}_{}'.format(visits_map_mnemonic,
                                            '_'.join(f.replace(':', '').replace('_', '') for f in args.features_set))

    print('==' * 40)
    start_dt = datetime.now()
    print('Experiment [{}, {}] - OUTPUT: {}'.format(args.features_set, args.visits_map, args.output_folder))
    print('Start: {}'.format(start_dt.strftime('%d-%m-%Y %H:%M:%S')))

    run_experiment(csv_data_file=args.datafile, features_set=tuple(args.features_set),
                   visit_map=args.visits_map, exp_log_folder_path=args.output_folder)

    end_dt = datetime.now()
    print('End: {}'.format(end_dt.strftime('%d-%m-%Y %H:%M:%S')))
    print('Exec Time: {}'.format(end_dt - start_dt))
    print('==' * 40)
