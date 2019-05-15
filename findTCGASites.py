from utils import *
import numpy as np
import pandas as pd
from consts import *
from tqdm import tqdm
import argparse as argparse

def find_important_features(x, y, sample_weight, features_names, is_multi, args):
    # shuffle this data
    x_train, y_train, sample_weight_train, x_test, y_test, sample_weight_test = shuffle_data(x, y, sample_weight)
    important_features = []
    stats = np.zeros((args.n, 3))
    pbar = tqdm(total=args.n)
    count = 0
    # we want to dump n sites, so train the random forest, get the most important feature and dump it
    while count <  args.n:
        forest = train_random_forest(x_train, y_train, sample_weight= sample_weight_train, max_depth=args.depth, num_trees=args.num_trees)
        # calculate basic statistics for this iteration
        if is_multi:
            TPRS, TNRS, ACC = predict_random_forest_multi(forest, x_test, y_test, np.unique(y_test))
        else:
            TPR, TNR, ACC = predict_random_forest(forest, x_test, y_test)
        if TPR < args.min_TPR_for_dump:
            continue
        if args.verbose == 2:
            tqdm.write("TPR %f, TNR %f, ACC %f"%(TPR, TNR, ACC))
        stats[count] = [TPR, TNR, ACC]
        importances = forest.feature_importances_
        most_important_ind = np.argmax(importances)
        # add best feature to np array
        important_features.append(features_names[most_important_ind].astype(np.str))
        # remove the feature before next iteration
        x_train = np.delete(x_train, most_important_ind, axis=1)
        x_test = np.delete(x_test, most_important_ind, axis=1)
        features_names = np.delete(features_names, most_important_ind)
        count += 1
        pbar.update(1)
    pbar.close()
    return important_features, stats

def merge_indices_and_dump(outpath, important_features, stats, df_illumina_sorted, df_bed):
    df_illumina_sites = pd.DataFrame(np.array(important_features), columns=['illumina_id'])

    # join sites with illumina_id
    final_df = df_illumina_sites.join(df_illumina_sorted, on='illumina_id', how='left')
    final_df = final_df.join(df_bed, on=['locus', 'chr'], how='left')
    final_df['TPR'] = pd.Series(stats[:,0], index=final_df.index)
    final_df['TNR'] = pd.Series(stats[:, 1], index=final_df.index)
    final_df['ACC'] = pd.Series(stats[:, 2], index=final_df.index)
    dump_sites_to_tsv(outpath, final_df)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath', type=str,
                        help='Outpath of our data and sites')
    parser.add_argument('--create_bins', type=int, default=0,
                        help='Whether to create binary files of pairs. 0=False,1=SickHealthy,2=HealthyHealthy,3=Both')
    parser.add_argument('--dump_sites', type=int, default=0,
                        help='Whether to dump sites of pairs. 0=False,1=SickHealthy,2=HealthyHealthy,3=Both')
    parser.add_argument('--files_exist', type=bool, default=True,
                        help="Whether binary files exist, if they don't need to build them in RAM")
    parser.add_argument('--n', type=int, default=200,
                        help='Number of sites to dump per pair')
    parser.add_argument('--num_trees', type=int, default=10,
                        help='Number of trees for random forest')
    parser.add_argument('--depth', type=int, default=3,
                        help='Depth of tree')
    parser.add_argument('--min_TPR_for_dump', type=float, default=0.75,
                        help='Minimal TPR for dumping a site')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity of script')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.create_bins == 1 or args.create_bins == 3:
        create_sick_healthy_pairs_bin(args.outpath, PAIRED_SICK_HEALTHY, verbose=args.verbose)
    elif args.create_bins == 2 or args.create_bins == 3:
        # create array of paired paths for all pairs of healthy
        a = np.array(PATHS_NORMAL_SUBTYPE)
        i, j = np.triu_indices(len(a), 1)
        paired_array = np.stack([a[i], a[j]]).T
        prog = re.compile('.+/(.+)_Solid_Tissue_Normal.*')
        paired_types = [[prog.match(pair[0])[1], prog.match(pair[1])[1]] for pair in paired_array]
        # create the binary files of said pairs
        create_healthy_healthy_pairs_bin(args.outpath, paired_array, paired_types, verbose=args.verbose)

    if args.dump_sites > 0:
        # read illumina and cpg.bed files in order to join dataframes later on
        df_illumina_sorted = pd.read_csv('/cs/cbio/dank/project/indices/Illumina_450k_sorted.txt', sep='\t', names=['chr', 'locus', 'illumina_id'])
        df_illumina_sorted.set_index('illumina_id', inplace=True)
        df_illumina_sorted.locus = df_illumina_sorted.locus.astype(np.str)
        df_bed = pd.read_csv('/cs/cbio/dank/project/indices/CpG.bed.gz', sep='\t',names=['chr', 'locus', 'cpg_index'])
        df_bed.locus = df_bed.locus.astype(np.str)
        df_bed.set_index(['locus', 'chr'], inplace=True)

    if args.dump_sites == 1 or args.dump_sites == 3:
        for type in tqdm(PAIRED_SICK_HEALTHY.keys()):
            if args.verbose >= 1:
                tqdm.write("Calculating important sites for data type %s" % type)
            path = "%s/sickHealthy/%s" % (args.outpath, type)
            if args.filesExist:
                # read the binary data
                x, y, sample_weight, features_names = read_data_from_bins(path)
            else:
                # just create the binary data for current pair and keep in RAM
                x, y, sample_weight, features_names = create_one_sick_healthy_pair(args.outpath, PAIRED_SICK_HEALTHY,
                                                                                   type, verbose=args.verbose, dump_to_file=False)

            important_features, stats = find_important_features(x, y, sample_weight, features_names, False, args)

            outpath = "%s/sickHealthy/%s/site_predictions_%s.tsv" % (args.outpath, type, type)

            merge_indices_and_dump(outpath, important_features, stats, df_illumina_sorted, df_bed)

    if args.dump_sites == 2 or args.dump_sites == 3:
        # create array of paired paths for all pairs of healthy
        a = np.array(PATHS_NORMAL_SUBTYPE)
        i, j = np.triu_indices(len(a), 1)
        paired_array = np.stack([a[i], a[j]]).T
        prog = re.compile('.+/(.+)_Solid_Tissue_Normal.*')
        paired_types = [[prog.match(pair[0])[1], prog.match(pair[1])[1]] for pair in paired_array]
        for i in tqdm(np.arange(paired_array.shape[0])):
            type1 = paired_types[i][0]
            type2 = paired_types[i][1]
            if args.verbose >= 1:
                tqdm.write("Calculating important sites for data types %s and %s" % (type1, type2))
            path = "%s/HealthyHealthy/%s_%s" % (args.outpath, type1, type2)
            if args.filesExist:
                # read the binary data
                x, y, sample_weight, features_names = read_data_from_bins(path)
            else:
                # just create the binary data for current pair and keep in RAM
                x, y, sample_weight, features_names = create_one_healthy_healthy_pair(args.outpath, paired_array[i], type1, type2, verbose=args.verbose, dump_to_file=False)

            important_features, stats = find_important_features(x, y, sample_weight, features_names, False, args)

            outpath = "%s/healthyHealthy/%s/site_predictions_%s.tsv" % (args.outpath, type1, type2)

            merge_indices_and_dump(outpath, important_features, stats, df_illumina_sorted, df_bed)

