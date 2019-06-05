import argparse
import os
import numpy as np
import pandas as pd
from utils import *
from findTCGASites import find_important_features
from sklearn.svm import SVC

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath', type=str,
                        help='Outpath of our data and sites')
    parser.add_argument('data_input_path', type=str,
                        help='Input path of data, expect info jsons,bins')
    parser.add_argument('site_input_path', type=str,
                        help='Input path of site predictions')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity of script')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    choose_random_sites = True

    # read the binary data
    x, y, sample_weight, features_names = read_data_from_bins(args.data_input_path)
    x_df = pd.DataFrame(x, columns=[features_names])

    # read the wgbs data
    wgbs_df = pd.read_csv('/cs/cbio/dank/project/TCGA_WGBS_c5.tsv', index_col=0).T
    BRCA_keys = [i for i in wgbs_df.index if "LIHC" in i]
    y_WGBS_BRCA = ["Primary" not in i for i in BRCA_keys]
    wgbs_df_BRCA = wgbs_df.loc[BRCA_keys] * 1000
    import pdb
    pdb.set_trace()
    # remove nan sites from WGBS
    nan_inds = np.where(np.any(np.isnan(wgbs_df_BRCA.values), axis=0))
    nan_ftr_names = wgbs_df_BRCA.keys()[nan_inds]

    wgbs_df_BRCA_no_nan = wgbs_df_BRCA.drop(nan_ftr_names, axis=1)

    # remove WGBS nan sites from x_df
    x_df_no_nan = x_df.drop(nan_ftr_names, axis=1)
    # sort x same as WGBS
    x_df_no_nan = x_df_no_nan.sort_index(axis=1)
    features_names_no_nan = np.array(x_df_no_nan.columns.get_level_values(0))
    n = 200
    num_no_nan_sites =  wgbs_df_BRCA_no_nan.shape[1]
    if choose_random_sites:
        # Choose randomly instead
        rand_inds = np.random.choice(np.arange(num_no_nan_sites), n)

        important_features = features_names_no_nan[rand_inds]
        # create a reduced df of just these sites
        important_features = np.array(important_features)
        x_reduced_sites = x_df_no_nan[important_features].values
        features_names_reduced_sites = important_features
    else:
        # read the sites
        sites = pd.read_csv(args.site_input_path, delim_whitespace=True)
        # create a reduced df of just these sites
        x_reduced_sites = x_df[sites.illumina_id].values
        features_names_reduced_sites = sites.illumina_id.values.astype('|S10')

        # TODO remove WGBS nan sites from here
        # TODO get indices in WGBS df


    # train svm on sites (either random or chosen)
    x_train, y_train, sample_weight_train, x_test, y_test, sample_weight_test = shuffle_data(x_reduced_sites, y, sample_weight)
    clf = SVC(class_weight='balanced', kernel='poly', degree=3)
    clf.fit(x_train, y_train)

    # predict on test data (TCGA)
    print("TCGA data accuracy: %f" %(np.sum(clf.predict(x_test) == y_test) / float(y_test.shape[0])))

    x_WGBS_reduced = wgbs_df_BRCA_no_nan.values.T[rand_inds].T
    print("WGBS data accuracy: %f" %(np.sum(clf.predict(x_WGBS_reduced) == y_WGBS_BRCA) / float(len(y_WGBS_BRCA))))