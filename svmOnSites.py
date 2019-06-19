import argparse
import os
import numpy as np
import pandas as pd
from utils import *
from findTCGASites import find_important_features
from sklearn.svm import SVC

def get_random_dfs(df_x, df_wgbs, ftr_names, num_sites, num_to_choose_from):
    # Choose randomly instead
    rand_inds = np.random.choice(np.arange(num_to_choose_from), num_sites, replace=False)

    random_features = ftr_names[rand_inds]
    # create a reduced df of just these sites
    important_features = np.array(random_features)
    x_reduced_sites = df_x[random_features].values
    # get reduced x_WGBS
    x_WGBS_reduced = df_wgbs[random_features].values
    return x_reduced_sites, x_WGBS_reduced, important_features

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('outpath', type=str,
    #                     help='Outpath of our data and sites')
    parser.add_argument('wgbs_input_path', type=str,
                        help='Input path of wgbs data')
    parser.add_argument('data_input_path', type=str,
                        help='Input path of data, expect info jsons,bins', default=None)
    parser.add_argument('site_input_path', type=str,
                        help='Input path of site predictions', default=None)
    parser.add_argument('--cancer_type', type=str, default='all',
                        help='cancer type to test on')
    parser.add_argument('--pos_vals', type=str, nargs='+',
                        help='What is the positive string in data')
    parser.add_argument('--random', action="store_true", default=False,
                        help='Whether to use random sites')
    parser.add_argument('--num_sites', type=int, default=200,
                        help='How many random sites to use')
    parser.add_argument('--num_iters', type=int, default=1,
                        help='How many attempts to make')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    choose_random_sites = args.random
    # read the binary data
    print("Reading TCGA data")
    x, y, sample_weight, features_names = read_data_from_bins(args.data_input_path)
    x_df = pd.DataFrame(x, columns=[features_names])

    # read the wgbs data of all cancers
    print("Reading WGBS data")
    wgbs_df = pd.read_csv(args.wgbs_input_path, index_col=0).T

    print("Processing data")
    # create labels and get specific cancer data
    if args.cancer_type == 'all':
        data_keys = wgbs_df.index
    else:
        data_keys = [i for i in wgbs_df.index if args.cancer_type in i]
    if len(data_keys) == 0:
        print("No cancer of this type in WGBS data")
        exit(0)
    y_WGBS = np.zeros_like(data_keys, dtype=np.bool)
    for j in args.pos_vals:
        y_WGBS = np.logical_or(y_WGBS, [j in i for i in data_keys])
    wgbs_df_data = wgbs_df.loc[data_keys] * 1000

    # remove nan sites from WGBS
    nan_inds = np.where(np.any(np.isnan(wgbs_df_data.values), axis=0))
    nan_ftr_names = wgbs_df_data.keys()[nan_inds]

    wgbs_df_no_nan = wgbs_df_data.drop(nan_ftr_names, axis=1)
    wgbs_df_data_no_nan = wgbs_df_no_nan.sort_index(axis=1)
    # remove WGBS nan sites from x_df
    x_df_no_nan = x_df.drop(nan_ftr_names, axis=1)
    # sort x same as WGBS
    x_df_no_nan = x_df_no_nan.sort_index(axis=1)
    features_names_no_nan = np.array(x_df_no_nan.columns.get_level_values(0))
    num_no_nan_sites =  wgbs_df_no_nan.shape[1]

    print("Running svm predictions")
    # if not choose_random_sites:
    #     # read the sites
    #     sites = pd.read_csv(args.site_input_path, delim_whitespace=True)
    #     features_names_reduced_sites = sites.illumina_id.values
    #     # remove nan wgbs sites
    #     non_nan_ftrs = np.intersect1d(features_names_reduced_sites, wgbs_df_no_nan.columns)
    #
    #     # create a reduced df of just these sites
    #     x_reduced_sites = x_df[non_nan_ftrs].values
    #
    #     # get reduced WGBS
    #     x_WGBS_reduced = wgbs_df_no_nan[non_nan_ftrs].values

    TCGA_accs = np.zeros(args.num_iters)
    WGBS_accs = np.zeros(args.num_iters)
    for i in tqdm(range(args.num_iters)):
        # train svm on sites (either random or chosen)
        if choose_random_sites:
            x_reduced_sites, x_WGBS_reduced, important_features = get_random_dfs(x_df_no_nan, wgbs_df_no_nan, features_names_no_nan, args.num_sites, num_no_nan_sites)
            # train svm on sites (either random or chosen)
            x_train, y_train, sample_weight_train, x_test, y_test, sample_weight_test = shuffle_data(x_reduced_sites, y, sample_weight)
        else:
            args.min_TPR_for_dump = 0
            args.n = args.num_sites
            args.depth = 3
            args.num_trees = 10
            args.verbose = 1
            important_features, stats, x_train, y_train, sample_weight_train, x_test, y_test, sample_weight_test = find_important_features(x_df_no_nan, y, sample_weight, features_names_no_nan,
                                                                                                                                           False, args, most_important=True, return_all=True)

            # create a reduced df of just these sites
            x_train = x_train[important_features].values
            x_test = x_test[important_features].values

            # get reduced WGBS
            x_WGBS_reduced = wgbs_df_no_nan[important_features].values

        #TODO try linear kernel
        clf = SVC(class_weight='balanced', kernel='poly', degree=3)
        clf.fit(x_train, y_train)

        TCGA_accs[i] = np.sum(clf.predict(x_test) == y_test) / float(y_test.shape[0])
        WGBS_accs[i] = np.sum(clf.predict(x_WGBS_reduced) == y_WGBS) / float(len(y_WGBS))

    print("TCGA data accuracy: average %f, std %f, for %f true samples out of %f" %(TCGA_accs.mean(), TCGA_accs.std(), y_test.sum(), y_test.shape[0]))
    print("WGBS data accuracy: average %f, std %f, for %f true samples out of %f" % (WGBS_accs.mean(), WGBS_accs.std(), y_WGBS.sum(), y_WGBS.shape[0]))