import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from consts import *
import os
import json
from tqdm import tqdm
import re

def train_random_forest(train_x, train_y, max_depth=3, num_trees=10, seed=666, sample_weight=None):
    clf = RandomForestClassifier(random_state=seed, max_depth=max_depth, n_estimators=num_trees)
    clf = clf.fit(train_x, train_y, sample_weight=sample_weight)
    return clf

def predict_random_forest(clf, x_test, y_test):
    preds = clf.predict(x_test)
    # get some basic stats
    pos_num = np.sum(y_test)
    neg_num = y_test.shape[0] - pos_num
    # get masks
    correct_ind_mask = preds == y_test
    # calc TPR
    tpr = np.sum(correct_ind_mask & (y_test == 1)) / pos_num
    # calc TNR
    tnr = np.sum(correct_ind_mask & (y_test == 0)) / neg_num
    # calc ACC
    acc = np.sum(correct_ind_mask) / float(y_test.shape[0])
    return tpr, tnr, acc

def predict_random_forest_ulti(clf, x_test, y_test):
    preds = clf.predict(x_test)
    # get some basic stats
    pos_num = np.sum(y_test)
    neg_num = y_test.shape[0] - pos_num
    # get masks
    correct_ind_mask = preds == y_test
    # calc TPR
    tpr = np.sum(correct_ind_mask & (y_test == 1)) / pos_num
    # calc TNR
    tnr = np.sum(correct_ind_mask & (y_test == 0)) / neg_num
    # calc ACC
    acc = np.sum(correct_ind_mask) / float(y_test.shape[0])
    return tpr, tnr, acc

def read_data_from_tsv(path):
    df = pd.read_csv(path, delim_whitespace=True, index_col=0)
    return df

def read_data_from_bins(dir):
    # read from files
    with open("%s/info.json" % dir) as f:
        info_dict = json.load(f)['data']
    features_names = np.loadtxt("%s/featuresNames.tsv" % dir, dtype=np.str, delimiter='\t')
    x = np.fromfile("%s/X.bin"  % dir, dtype=info_dict['X']['dtype']).reshape(info_dict['X']['N'], info_dict['X']['C'])
    y = np.fromfile("%s/Y.bin" % dir, dtype=info_dict['Y']['dtype'])
    sample_weight = np.fromfile("%s/sample_weight.bin" % dir, dtype=info_dict['sample_weight']['dtype'])
    return x, y, sample_weight, features_names

def read_and_save_data(outpath, file_paths_array, label_array, is_multi=False, dump_to_file=True):
    x = pd.DataFrame()
    y = []
    patient_nums = []
    # read each file and get the patient nums
    for path, label in zip(file_paths_array, label_array.keys()):
        print("Reading %s" % path)
        curr_x = read_data_from_tsv(path)
        patient_num = len(curr_x.columns)
        curr_y = np.full(patient_num, int(label_array[label][1]))
        x = pd.concat([x, curr_x], axis=1)
        y.extend(curr_y)
        patient_nums.append(patient_num)
        del curr_x
        del curr_y

    y = np.array(y)
    patient_nums = np.array(patient_nums)
    x = x.dropna(axis=0, how='any')
    features_names = x.index

    # calculate the sample weight for each sample
    sample_weight = np.zeros(patient_nums.sum())
    if is_multi:
        count = 0
        for i, patient_num  in enumerate(patient_nums):
            sample_weight[count : count+patient_num] = patient_nums.max() / patient_num
            count += patient_num
    else:
        count = 0
        for label, patient_num in zip(label_array.keys(), patient_nums):
            is_sick = label_array[label][1]
            if is_sick:
                sample_weight[count : count + patient_num] = 1
            else:
                sample_weight[count: count + patient_num] = patient_num
            count += patient_num
        num_sick = len(np.where(sample_weight == 1)[0])
        not_sick = np.where(sample_weight != 1)[0]

        sample_weight[not_sick] = num_sick / sample_weight[not_sick]

    if dump_to_file:
        # dump to files
        x = x.values.transpose()
        x = x.astype(np.uint16)
        x.tofile("%s/X.bin" % outpath)
        y = y.astype(np.uint8)
        sample_weight = sample_weight.astype(np.float32)
        features_names = features_names.values.astype(np.str)
        np.savetxt("%s/featuresNames.tsv" % outpath, features_names, delimiter='\t', fmt='%s')
        y.tofile("%s/Y.bin" % outpath)
        sample_weight.tofile("%s/sample_weight.bin" % outpath )

        shape_type_dicts = {"data":{"X": {"N": x.shape[0], "C": x.shape[1], "dtype": x.dtype.name, "description": "Columns are the sites and rows are the patients"},
                                     "Y": {"N": y.shape[0], "C": 1, "dtype": y.dtype.name, "description": "Label of each patient for classification"},
                                     "sample_weight": {"N": sample_weight.shape[0], "C": 1, "dtype": sample_weight.dtype.name, "description": "Weight each patient gets in this dataset"},
                                     "featuresNames": {"N": features_names.shape[0], "C": 1, "dtype": 'np.str', "description": "Name of each column (each site)"}}}

        with open("%s/info.json" % outpath, 'w') as f:
            json.dump(shape_type_dicts, f, indent=4)
    return x, y, sample_weight, features_names

def create_one_healthy_healthy_pair(outpath, pair_paths, type1, type2, verbose=0, dump_to_file=True):
    curr_outpath = "%s/sickHealthy/%s_%s"%(outpath, type1, type2)
    os.mkdir(curr_outpath)
    if verbose:
        print("Creating paired data for %s in %s" % (type, curr_outpath))
    curr_labels = OrderedDict({'Healthy1': tuple((type1, 1)), 'Healthy2': tuple((type2, 0))})
    x, y, sample_weight, features_names = read_and_save_data(curr_outpath, pair_paths, curr_labels, is_multi=True, dump_to_file=dump_to_file)
    return x, y, sample_weight, features_names

def create_one_sick_healthy_pair(outpath, paired_dict, type, verbose=0, dump_to_file=True):
    curr_outpath = "%s/sickHealthy/%s"%(outpath, type)
    os.mkdir(curr_outpath)
    if verbose:
        print("Creating paired data for %s in %s" % (type, curr_outpath))
    curr_labels = OrderedDict({'Sick': tuple((type, True)), 'Healthy': tuple((type, False))})
    curr_paths = [paired_dict[type][0], paired_dict[type][1]]
    x, y, sample_weight, features_names = read_and_save_data(curr_outpath, curr_paths, curr_labels, dump_to_file=dump_to_file)
    return x, y, sample_weight, features_names

def create_sick_healthy_pairs_bin(outpath, paired_dict=PAIRED_SICK_HEALTHY, verbose=0):
    for type, val in paired_dict.items():
        create_one_sick_healthy_pair(outpath, paired_dict, type, verbose)

def create_healthy_healthy_pairs_bin(outpath, paired_array, paired_types, verbose=0):
    for i in tqdm(np.arange(paired_array.shape[0])):
        type1 = paired_types[i][0]
        type2 = paired_types[i][1]
        create_one_healthy_healthy_pair(outpath, paired_array[i], type1, type2, verbose)

def shuffle_data(x, y, sample_weight, tt_split=0.3):
    train_idx = []
    test_idx = []
    # get unique labels and their counts
    uniques, counts= np.unique(Y, return_counts=True)
    # split each label into train and test indices
    for unq,cnt in zip(uniques, counts):
        curr_train_indices = np.random.rand(cnt) > tt_split
        curr_test_indices = np.logical_not(curr_train_indices)
        train_idx.extend(np.where(Y == unq)[0][curr_train_indices])
        test_idx.extend(np.where(Y == unq)[0][curr_test_indices])
    # shuffle in train and test after the split
    train_idx = np.random.permutation(np.array(train_idx))
    test_idx = np.random.permutation(np.array(test_idx))
    x_train = x[train_idx]
    y_train = y[train_idx]
    sample_weight_train = sample_weight[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    sample_weight_test = sample_weight[test_idx]
    return x_train, y_train, sample_weight_train, x_test, y_test, sample_weight_test


def dump_sites_to_tsv(outpath, sites):
    sites.to_csv(outpath, sep='\t', index=False)

def drop_col_feat_imp(model, x_train, y_train, cols_to_remove=None, random_state = 42):

    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(x_train, y_train)
    benchmark_score = model_clone.score(x_train, y_train)
    # list for storing feature importances
    importances = []

    if cols_to_remove is None:
        cols_to_remove = x_train.columns
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in tqdm(cols_to_remove):
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(x_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(x_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    imp_df = pd.DataFrame(data={'Feature':cols_to_remove, 'Importance':np.array(importances)})
    return imp_df


    # for type in TEST_PAIRED_SICK_HEALTHY.keys():
    #     path = "./%s"%type
    #     X, Y, sample_weight, features_names = read_data_from_bins(path)
    #     X_train, Y_train, sample_weight_train, X_test, Y_test, sample_weight_test = shuffle_data(X, Y, sample_weight)
    #     for i in num_total_trees
    #     forest = trainRandomForest(X_train, Y_train, sample_weight_train)
    #     importances =  forest.feature_importances_
    #     indices = np.argsort(importances)[::-1]
    #     importances_df = drop_col_feat_imp(forest, pd.DataFrame(X_train, columns=features_names), Y_train,
    #                                        cols_to_remove=features_names[indices][:2000])
    #     import pdb
    #     pdb.set_trace()
        # print(forest.score(X_test, Y_test))
        # importances =  forest.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
        #              axis=0)
        # indices = np.argsort(importances)[::-1]
        #
        # # Print the feature ranking
        # print("Feature ranking:")
        # N = 10
        # for f in range(N):
        #     print("%d. feature %s (%f)" % (f + 1, features_names[indices[f]], importances[indices[f]]))
        #
        # # Plot the feature importances of the forest
        # plt.figure()
        # plt.title("Feature importances")
        # plt.bar(range(N), importances[indices][:N],
        #        color="r", yerr=std[indices][:N], align="center")
        # plt.xticks(range(N), features_names[indices[:N]])
        # plt.xlim([-1, N])
        # plt.show()
        # # for tree_in_forest in clf.estimators_:
        # #     best_ftr = np.argmax(tree_in_forest.feature_importances_)
        # #     print(features_names[best_ftr], tree_in_forest.feature_importances_[best_ftr])
        #
