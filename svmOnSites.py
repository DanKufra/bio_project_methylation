import argparse
import os
import numpy as np
from utils import *
from findTCGASites import find_important_features
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

    # read the binary data
    x, y, sample_weight, features_names = read_data_from_bins(args.data_input_path)
    sites = pd.read_csv(args.site_input_path, delim_whitespace=True)
    x_df = pd.DataFrame(x, columns=[features_names])
    x_reduced_sites = x_df[sites.illumina_id].values
    features_names_reduced_sites = sites.illumina_id.values.astype('|S10')
    args.n = 5
    args.min_TPR_for_dump = 0
    args.depth = 3
    args.num_trees = 10
    important_features, stats = find_important_features(x_reduced_sites, y, sample_weight, features_names_reduced_sites, False, args)
