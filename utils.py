import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from consts import *
import os
import json


def runRandomForest(train_x, train_y, test_x, test_y, max_depth = 5, num_trees=20, seed=None, class_weights=None):
    clf = RandomForestClassifier(random_state=seed, max_depth=max_depth, n_estimators=num_trees, class_weights=class_weights)
    clf = clf.fit(train_x, train_y)

def readDataFromTSV(path):
    df = pd.read_csv(path, delim_whitespace=True, compression="gzip", index_col=0)
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

def createSickHealthyPairsBin():
    for type in PAIRED_SICK_HEALTHY.keys():
        print("Creating paired data for %s" % type)
        outpath = "./%s"%type
        os.mkdir(outpath)
        temp_labels = OrderedDict({0: tuple((type, True)), 1: tuple((type, False))})
        temp_list = []
        temp_list.append(PAIRED_SICK_HEALTHY[type][0])
        temp_list.append(PAIRED_SICK_HEALTHY[type][1])
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



if __name__ == '__main__':
    for type in PAIRED_SICK_HEALTHY.keys():
        path = "./%s"%type
        X, Y, sample_weight, features_names = readDataFromBins(path)
        X_train, Y_train, sample_weight_train, X_test, Y_test, sample_weight_test = shuffle_data(X, Y, sample_weight)

