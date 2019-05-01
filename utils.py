import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from consts import *
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse as argparse


def trainRandomForest(train_x, train_y, max_depth=3, num_trees=10, seed=666, sample_weight=None):
    clf = RandomForestClassifier(random_state=seed, n_estimators=num_trees)
    clf = clf.fit(train_x, train_y, sample_weight=sample_weight)
    return clf

def predictRandomForest(clf, x_test, y_test):
    preds = clf.predict(x_test)
    # get some basic stats
    pos_num = np.sum(y_test)
    neg_num = y_test.shape[0] - pos_num
    # get masks
    true_ind_mask = preds == y_test
    false_ind_mask = np.logical_not(true_ind_mask)

    # calc TPR
    TPR = np.sum(true_ind_mask & (y_test == 1)) / pos_num
    # calc TNR
    TNR = np.sum(true_ind_mask & (y_test == 0)) / neg_num
    # calc ACC
    ACC = np.sum(true_ind_mask) / float(y_test.shape[0])
    return TPR, TNR, ACC

def readDataFromTSV(path):
    df = pd.read_csv(path, delim_whitespace=True, index_col=0)
    return df

def readDataFromBins(dir):
    # read from files
    with open("%s/info.json" % dir) as f:
        info_dict = json.load(f)['data']
    featuresNames = np.loadtxt("%s/featuresNames.tsv" % dir, dtype=np.str, delimiter='\t')
    num_cols = len(featuresNames)
    X = np.fromfile("%s/X.bin"  % dir, dtype=info_dict['X']['dtype']).reshape(info_dict['X']['N'], info_dict['X']['C'])
    Y = np.fromfile("%s/Y.bin" % dir, dtype=info_dict['Y']['dtype'])
    sample_weight = np.fromfile("%s/sample_weight.bin" % dir, dtype=info_dict['sample_weight']['dtype'])
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
        curr_Y = np.full(patient_num, int(label_array[label][1]))
        X = pd.concat([X, curr_X], axis=1)
        Y.extend(curr_Y)
        patient_nums.append(patient_num)
        del curr_X
        del curr_Y

    Y = np.array(Y)
    patient_nums = np.array(patient_nums)
    X = X.dropna(axis=0, how='any')
    featuresNames = X.index

    # dump X to file then delete
    X = X.values.transpose()
    X = X.astype(np.uint16)
    X.tofile("%s/X.bin" %(outpath))
    X_shape = X.shape
    X_dtype = X.dtype
    del X
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
    Y = Y.astype(np.uint8)
    sample_weight = sample_weight.astype(np.float32)
    featuresNames = featuresNames.values.astype(np.str)
    np.savetxt("%s/featuresNames.tsv" % outpath, featuresNames, delimiter='\t', fmt='%s')
    Y.tofile("%s/Y.bin" %(outpath))
    sample_weight.tofile("%s/sample_weight.bin" %(outpath))

    shape_type_dicts = {"data":{"X": {"N": X_shape[0], "C": X_shape[1], "dtype": X_dtype.name, "description": "Columns are the sites and rows are the patients"},
                                 "Y": {"N": Y.shape[0], "C": 1, "dtype": Y.dtype.name, "description": "Label of each patient for classification"},
                                 "sample_weight": {"N": sample_weight.shape[0], "C": 1, "dtype": sample_weight.dtype.name, "description": "Weight each patient gets in this dataset"},
                                 "featuresNames": {"N": featuresNames.shape[0], "C": 1, "dtype": 'np.str', "description": "Name of each column (each site)"}}}

    with open("%s/info.json" % outpath, 'w') as f:
        json.dump(shape_type_dicts, f, indent=4)

def createSickHealthyPairsBin(outpath, paired_dict=PAIRED_SICK_HEALTHY):
    for type, val in paired_dict.items():
        curr_outpath = "%s/sickHealthy/%s"%(outpath, type)
        os.mkdir(curr_outpath)
        print("Creating paired data for %s in %s" % (type, curr_outpath))
        currlabels = OrderedDict({'Sick': tuple((type, True)), 'Healthy': tuple((type, False))})
        curr_paths = []
        curr_paths.append(paired_dict[type][0])
        curr_paths.append(paired_dict[type][1])
        read_and_save_data(curr_outpath, curr_paths, currlabels, type)

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


def dumpSitesToTSV(outpath, sites):
    sites.to_csv(outpath, sep='\t', index=False)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath', type=str,
                        help='Outpath of our data and sites')
    parser.add_argument('--createBins', type=int, default=0,
                        help='Whether to create binary files of pairs. 0=False,1=SickHealthy,2=HealthyHealthy,3=Both')
    parser.add_argument('--dumpSites', type=int, default=0,
                        help='Whether to dump sites of pairs. 0=False,1=SickHealthy,2=HealthyHealthy,3=Both')
    parser.add_argument('--n', type=int, default=200,
                        help='Number of sites to dump per pair')
    parser.add_argument('--num_trees', type=int, default=10,
                        help='Number of trees for random forest')
    parser.add_argument('--depth', type=int, default=3,
                        help='Depth of tree')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.createBins == 1:
        createSickHealthyPairsBin(args.outpath, PAIRED_SICK_HEALTHY)
    elif args.createBins == 2:
        # createSickHealthyPairsBin(args.outpath, PAIRED_SICK_HEALTHY)
        pass
    elif args.createBins == 3:
        createSickHealthyPairsBin(args.outpath, PAIRED_SICK_HEALTHY)
        pass
    if args.dumpSites == 1:
        df_illumina_sorted = pd.read_csv('/cs/cbio/dank/project/indices/Illumina_450k_sorted.txt', sep='\t', names=['chr', 'locus', 'illumina_id'])
        df_illumina_sorted.set_index('illumina_id', inplace=True)
        # df_illumina_sorted.locus = df_illumina_sorted.locus.apply(np.int64)
        df_illumina_sorted.locus = df_illumina_sorted.locus.astype(np.str)
        df_bed = pd.read_csv('/cs/cbio/dank/project/indices/CpG.bed.gz', sep='\t',names=['chr', 'locus', 'cpg_index'])
        df_bed.locus = df_bed.locus.astype(np.str)
        df_bed.set_index(['locus', 'chr'], inplace=True)

        for type in tqdm(PAIRED_SICK_HEALTHY.keys()):
            tqdm.write("Calculating important sites for data type %s" % type)
            path = "%s/sickHealthy/%s" % (args.outpath, type)
            # read the binary date
            X, Y, sample_weight, features_names = readDataFromBins(path)
            # shuffle this data
            X_train, Y_train, sample_weight_train, X_test, Y_test, sample_weight_test = shuffle_data(X, Y, sample_weight)
            # we want to dump n sites, so train the random forest, get the most important feature and dump it
            important_features = []
            for i in tqdm(range(args.n)):
                forest = trainRandomForest(X_train, Y_train, sample_weight= sample_weight_train, max_depth=args.depth, num_trees=args.num_trees)
                TPR, TNR, ACC = predictRandomForest(forest, X_test, Y_test)
                tqdm.write("TPR %f, TNR %f, ACC %f"%(TPR, TNR, ACC))
                importances = forest.feature_importances_
                most_important_ind = np.argmax(importances)
                # add best feature to np array
                important_features.append(features_names[most_important_ind].astype(np.str))
                # remove the feature before next iteration
                X_train = np.delete(X_train, most_important_ind, axis=1)
                X_test = np.delete(X_test, most_important_ind, axis=1)
                features_names = np.delete(features_names, most_important_ind)

            df_illumina_sites = pd.DataFrame(np.array(important_features), columns=['illumina_id'])
            # join sites with illumina_id
            final_df = df_illumina_sites.join(df_illumina_sorted, on='illumina_id', how='left')
            final_df = final_df.join(df_bed, on=['locus', 'chr'], how='left')
            dumpSitesToTSV("%s/sickHealthy/%s/site_predictions_%s.tsv" % (args.outpath, type, type), final_df)

    # for type in TEST_PAIRED_SICK_HEALTHY.keys():
    #     path = "./%s"%type
    #     X, Y, sample_weight, features_names = readDataFromBins(path)
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
