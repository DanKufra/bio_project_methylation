from utils import *
import numpy as np
import pandas as pd
from consts import *
from tqdm import tqdm
import argparse as argparse

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
    parser.add_argument('--min_TPR_for_dump', type=float, default=0.75,
                        help='Minimal TPR for dumping a site')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity of script')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.createBins == 1:
        createSickHealthyPairsBin(args.outpath, PAIRED_SICK_HEALTHY, verbose=args.verbose)
    elif args.createBins == 2:
        # createSickHealthyPairsBin(args.outpath, PAIRED_SICK_HEALTHY, verbose=args.verbose)
        pass
    elif args.createBins == 3:
        createSickHealthyPairsBin(args.outpath, PAIRED_SICK_HEALTHY, verbose=args.verbose)
        pass
    if args.dumpSites == 1:
        # read illumina and cpg.bed files in order to join dataframes later on
        df_illumina_sorted = pd.read_csv('/cs/cbio/dank/project/indices/Illumina_450k_sorted.txt', sep='\t', names=['chr', 'locus', 'illumina_id'])
        df_illumina_sorted.set_index('illumina_id', inplace=True)
        df_illumina_sorted.locus = df_illumina_sorted.locus.astype(np.str)
        df_bed = pd.read_csv('/cs/cbio/dank/project/indices/CpG.bed.gz', sep='\t',names=['chr', 'locus', 'cpg_index'])
        df_bed.locus = df_bed.locus.astype(np.str)
        df_bed.set_index(['locus', 'chr'], inplace=True)

        for type in tqdm(PAIRED_SICK_HEALTHY.keys()):
            if args.verbose >= 1:
                tqdm.write("Calculating important sites for data type %s" % type)
            path = "%s/sickHealthy/%s" % (args.outpath, type)
            # read the binary date
            X, Y, sample_weight, features_names = readDataFromBins(path)
            # shuffle this data
            X_train, Y_train, sample_weight_train, X_test, Y_test, sample_weight_test = shuffle_data(X, Y, sample_weight)
            important_features = []
            stats = np.zeros((args.n, 3))
            pbar = tqdm(total=args.n)
            count = 0
            # we want to dump n sites, so train the random forest, get the most important feature and dump it
            while count <  args.n:
                forest = trainRandomForest(X_train, Y_train, sample_weight= sample_weight_train, max_depth=args.depth, num_trees=args.num_trees)
                # calculate basic statistics for this iteration
                TPR, TNR, ACC = predictRandomForest(forest, X_test, Y_test)
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
                X_train = np.delete(X_train, most_important_ind, axis=1)
                X_test = np.delete(X_test, most_important_ind, axis=1)
                features_names = np.delete(features_names, most_important_ind)
                count += 1
                pbar.update(1)
            pbar.close()
            df_illumina_sites = pd.DataFrame(np.array(important_features), columns=['illumina_id'])

            # join sites with illumina_id
            final_df = df_illumina_sites.join(df_illumina_sorted, on='illumina_id', how='left')
            final_df = final_df.join(df_bed, on=['locus', 'chr'], how='left')
            final_df['TPR'] = pd.Series(stats[:,0], index=final_df.index)
            final_df['TNR'] = pd.Series(stats[:, 1], index=final_df.index)
            final_df['ACC'] = pd.Series(stats[:, 2], index=final_df.index)
            dumpSitesToTSV("%s/sickHealthy/%s/site_predictions_%s.tsv" % (args.outpath, type, type), final_df)