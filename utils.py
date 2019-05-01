import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from consts import *
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


def trainRandomForest(train_x, train_y, max_depth=5, num_trees=20, seed=666, sample_weight=None):
    clf = RandomForestClassifier(random_state=seed, n_estimators=num_trees)
    clf = clf.fit(train_x, train_y, sample_weight=sample_weight)
    return clf

def readDataFromTSV(path):
    df = pd.read_csv(path, delim_whitespace=True, index_col=0)
    return df

def readDataFromBins(dir):
    # read from files
    featuresNames = np.fromfile("%s/featuresNames.bin" % dir, dtype='<U10')
    num_cols = len(featuresNames)
    X = np.fromfile("%s/X.bin"  % dir, dtype=np.uint16).reshape(-1, num_cols)
    Y = np.fromfile("%s/Y.bin" % dir, dtype=np.uint8)
    sample_weight = np.fromfile("%s/sample_weight.bin" % dir, dtype=np.float32)
    return X, Y, sample_weight, featuresNames

def read_and_save_data(outpath, file_paths_array, label_array, suffix, is_multi=False):
    X = pd.DataFrame()
    Y = []
    patient_nums = []
    # read each file and get the patient nums
    for path, label in zip(file_paths_array, label_array.keys()):
        print("Reading %s" % path)
        curr_X = readDataFromTSV(path)
        patient_num = len(curr_X.columns)
        curr_Y = np.full(patient_num, label)
        X = pd.concat([X, curr_X], axis=1)
        Y.extend(curr_Y)
        patient_nums.append(patient_num)

    Y = np.array(Y)
    patient_nums = np.array(patient_nums)
    X = X.dropna(axis=0, how='any')
    featuresNames = X.index
    X = X.values.transpose()

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

    # dump to files
    X = X.astype(np.uint16)
    Y = Y.astype(np.uint8)
    sample_weight = sample_weight.astype(np.float32)
    featuresNames = featuresNames.values.astype(np.str)
    X.tofile("%s/X.bin"  %(outpath))
    Y.tofile("%s/Y.bin" %(outpath))
    sample_weight.tofile("%s/sample_weight.bin" %(outpath))
    with open("%s/featuresNames.bin"%(outpath), "wb") as f:
        f.write(featuresNames)

    shape_type_dicts = {"data":{"X": {"N": X.shape[0], "C": X.shape[1], "dtype": X.dtype.name, "description": "Columns are the sites and rows are the patients"},
                                 "Y": {"N": Y.shape[0], "C": 1, "dtype": Y.dtype.name, "description": "Label of each patient for classification"},
                                 "sample_weight": {"N": sample_weight.shape[0], "C": 1, "dtype": sample_weight.dtype.name, "description": "Weight each patient gets in this dataset"},
                                 "featuresNames": {"N": featuresNames.shape[0], "C": 1, "dtype": featuresNames.dtype.name, "description": "Name of each column (each site)"}}}

    with open("%s/info.json" % outpath, 'w') as f:
        json.dump(shape_type_dicts, f, indent=4)

def createSickHealthyPairsBin(paired_dict=PAIRED_SICK_HEALTHY):
    for type in paired_dict.keys():
        print("Creating paired data for %s" % type)
        outpath = "./%s"%type
        os.mkdir(outpath)
        temp_labels = OrderedDict({0: tuple((type, True)), 1: tuple((type, False))})
        temp_list = []
        temp_list.append(paired_dict[type][0])
        temp_list.append(paired_dict[type][1])
        read_and_save_data(outpath, temp_list, temp_labels, type)

def shuffle_data(X, Y, sample_weight, tt_split=0.3):
    train_idx = []
    test_idx = []
    uniques, counts= np.unique(Y, return_counts=True)
    for unq,cnt in zip(uniques, counts):
        curr_train_indices = np.random.rand(cnt) > tt_split
        curr_test_indices = np.logical_not(curr_train_indices)
        train_idx.extend(np.where(Y == unq)[0][curr_train_indices])
        test_idx.extend(np.where(Y == unq)[0][curr_test_indices])
    train_idx = np.random.permutation(np.array(train_idx))
    test_idx = np.random.permutation(np.array(test_idx))
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    sample_weight_train = sample_weight[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    sample_weight_test = sample_weight[test_idx]
    return X_train, Y_train, sample_weight_train, X_test, Y_test, sample_weight_test


def drop_col_feat_imp(model, X_train, y_train, cols_to_remove=None, random_state = 42):

    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []

    if cols_to_remove is None:
        cols_to_remove = X_train.columns
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in tqdm(cols_to_remove):
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    imp_df = pd.DataFrame(data={'Feature':cols_to_remove, 'Importance':np.array(importances)})
    return imp_df

if __name__ == '__main__':
    # createSickHealthyPairsBin(TEST_PAIRED_SICK_HEALTHY)
    for type in TEST_PAIRED_SICK_HEALTHY.keys():
        path = "./%s"%type
        X, Y, sample_weight, features_names = readDataFromBins(path)
        X_train, Y_train, sample_weight_train, X_test, Y_test, sample_weight_test = shuffle_data(X, Y, sample_weight)
        forest = trainRandomForest(X_train, Y_train, sample_weight_train)
        importances =  forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        importances_df = drop_col_feat_imp(forest, pd.DataFrame(X_train, columns=features_names), Y_train,
                                           cols_to_remove=features_names[indices][:2000])
        import pdb
        pdb.set_trace()
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
