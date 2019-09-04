from utils import*
from sklearn.svm import SVC

DIAG_MAPPING = {'Positive' : 1, 'Negative': -1,
                'Indeterminate': 0, 'Equivocal': 0, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}

LEVEL_SCORE_MAPPING = {'0' : -1, '1+': -1, '2+': 0, '3+': 1, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}

REL_COLS = ['er_ihc', 'pr_ihc', 'her2_ihc', 'her2_fish', 'her2_ihc_level', 'pos', 'her2_ihc_and_fish']

# class encoding er,pr,her2
CLASSES = {'111' : 7, '110' : 6, '101' : 5, '100' : 4,
           '011' : 3, '010' : 2, '001' : 1, '000' : 0}


def print_stats(clsf, tt, receptor, preds, lbls):
    # get some basic stats
    pos_num = np.sum(lbls)
    neg_num = lbls.shape[0] - pos_num
    # get masks
    correct_ind_mask = preds == lbls
    # calc TPR
    tpr = np.sum(correct_ind_mask & (lbls == 1)) / pos_num
    # calc TNR
    tnr = np.sum(correct_ind_mask & (lbls == 0)) / neg_num
    # calc ACC
    acc = np.sum(correct_ind_mask) / float(lbls.shape[0])
    print("%s %s %s: ACC: %f, TPR: %f, TNR: %f, numPos: %f, numNeg: %f" %(clsf, tt, receptor, acc, tpr, tnr, pos_num, neg_num))


def readData():
    print("Reading data")
    # read BRCA patients prognosis
    df_BRCA_diagnosis = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/BRCA.tsv', delimiter='\t')
    df_BRCA_diagnosis.set_index('bcr_patient_barcode', inplace=True)
    df_BRCA_diagnosis.drop(['bcr_patient_barcode', 'CDE_ID:2003301'], inplace=True)

    # read BRCA patients matching
    df_id_matching = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/sample_sheet_BRCA.tsv', delimiter='\t')

    # read methylation data
    # df_healthy = read_data_from_tsv('/cs/cbio/tommy/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz')
    df_sick = read_data_from_tsv('/cs/cbio/dank/BRCA_TCGA_data/BRCA_Primary_Tumor.tsv.gz')

    # match to methylation data
    # df_joined = df_id_matching.join(pd.concat([df_sick.T, df_healthy.T]), on='Array.Data.File', how='inner')
    df_joined = df_id_matching.join(df_sick.T, on='Sample ID', how='inner')
    # df_joined = df_id_matching

    df_joined.drop(['File ID', 'File Name', 'Data Category', 'Data Type', 'Project ID', 'Sample ID', 'Sample Type'],
                   axis=1, inplace=True)
    df_joined.set_index('Case ID', inplace=True)

    final_df = df_BRCA_diagnosis[
        ['er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'her2_fish_status', 'her2_ihc_score']].join(
        df_joined, how='inner')

    # replace string labels with int for ease of parsing
    final_df['er_ihc'] = final_df['er_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['er_status_by_ihc'])
    final_df['pr_ihc'] = final_df['pr_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['pr_status_by_ihc'])
    final_df['her2_ihc'] = final_df['her2_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['her2_status_by_ihc'])
    final_df['her2_fish'] = final_df['her2_fish_status'].map(DIAG_MAPPING).fillna(final_df['her2_fish_status'])

    # create a vector of her2_ihc_level which is the status based on the score we had
    final_df['her2_ihc_level'] = final_df['her2_ihc_score'].map(LEVEL_SCORE_MAPPING).fillna(final_df['her2_ihc_score'])
    return final_df


def getMismatches(df):
    # Crosstab between her2 ihc status and fish status
    print("her2_ihc vs her2_fish")
    print(pd.crosstab(df['her2_ihc'], df['her2_fish']))

    # get list of patients where her2 fish status does not match ihc status
    fish_vs_ihc = df[((df['her2_ihc'] == 1) & (df['her2_fish'] == -1)) | (
                (df['her2_ihc'] == -1) & (df['her2_fish'] == 1))]
    print("%d patients with mismatch between her2_ihc and her2_fish" % len(fish_vs_ihc))

    # Crosstab between her2 ihc status and ihc score
    print("her2_ihc_status vs her2_ihc_score")
    print(pd.crosstab(df['her2_ihc'], df['her2_ihc_level']))

    # get list of patients where her2 ihc level does not match ihc status
    ihc_status_vs_ihc_level = df[((df['her2_ihc'] == 1) & (df['her2_ihc_level'] == -1)) | (
                (df['her2_ihc'] == -1) & (df['her2_ihc_level'] == 1))]
    print("%d patients with mismatch between her2_ihc_status and her2_ihc_score" % len(ihc_status_vs_ihc_level))


def fixMismatches(df):
    # Replacing HER2 (IHC) status by HER2 (FISH) (a more accurate test) whenever available:
    print("Replacing her2_ihc by her2_fish where available")
    df['her2_ihc_and_fish'] = df['her2_ihc']
    df.loc[df['her2_fish'] != -2, 'her2_ihc_and_fish'] = df.loc[
        df['her2_fish'] != -2, 'her2_fish']

    # Crosstab between her2 ihc status and ihc status with fish
    print("her2_ihc vs her2_ihc_fish")
    print(pd.crosstab(df['her2_ihc'], df['her2_ihc_and_fish']))

    er_vs_pr = df[((df['er_ihc'] == 1) & (df['pr_ihc'] == -1)) | (
                (df['er_ihc'] == -1) & (df['pr_ihc'] == 1))]
    print("%d patients with mismatch between er_ihc and pr_ihc" % len(er_vs_pr))

    # create Triple Negative vs not Triple Negative dataset
    print("Creating Triple Negative vs not Triple negative labels")
    df_clinical = pd.DataFrame(df[np.hstack([df.columns[['cg' in col for col in df.columns]],
                                                   ['er_ihc', 'pr_ihc', 'her2_ihc_and_fish', 'her2_ihc', 'her2_fish',
                                                    'her2_ihc_level']])])

    df_clinical['neg'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (
                df_clinical['her2_ihc_and_fish'] == -1)
    df_clinical['pos'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (
                df_clinical['her2_ihc_and_fish'] == 1)
    df_clinical['NA'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (
                df_clinical['her2_ihc_and_fish'] == -2)

    df_clinical['neg_pre_fish'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (
                df_clinical['her2_ihc'] == -1)
    df_clinical['pos_pre_fish'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (
                df_clinical['her2_ihc'] == 1)
    df_clinical['NA_pre_fish'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (
                df_clinical['her2_ihc'] == -2)

    # drop patients who are NA and not pos
    # df_clinical.drop(df_clinical[df_clinical['NA'] & np.logical_not(df_clinical['pos'])].index, axis=0, inplace=True)
    print("Removing patients with NA")
    df_clinical.drop(df_clinical[df_clinical['NA'] == True].index, axis=0, inplace=True)
    return df_clinical


def classify(receptor, X_test, X_train, Y_test, Y_train):
    print("Running SVM on data - predict %s :" % receptor)
    clf = SVC(class_weight='balanced', kernel='linear')
    clf.fit(X_train, Y_train)

    pred_test = clf.predict(X_test)
    pred_train = clf.predict(X_train)

    print_stats('SVM', 'train', 'triple_negative', pred_train, Y_train)
    print_stats('SVM', 'test', 'triple_negative', pred_test, Y_test)

    print("Running random forest  - predict triple negative :")
    clf_rf = RandomForestClassifier(random_state=666, max_depth=3, n_estimators=10,
                                                    class_weight='balanced')
    clf_rf = clf_rf.fit(X_train, Y_train)
    pred_test_rf = clf_rf.predict(X_test)
    pred_train_rf = clf_rf.predict(X_train)

    print_stats('Random Forest', 'train', receptor, pred_train_rf, Y_train)
    print_stats('Random Forest', 'test', receptor, pred_test_rf, Y_test)
    return pred_test, pred_train, pred_train_rf, pred_test_rf


def shuffle_idx(X, Y, train_idx):
    train_idx[np.random.choice(np.arange(0, Y.shape[0])[np.logical_not(train_idx)],
                               np.round(0.75 * Y.shape[0] - train_idx.sum()).astype(np.uint32))] = True

    shuf_test_idx = np.random.permutation(np.where(~train_idx)[0])
    shuf_train_idx = np.random.permutation(np.where(train_idx)[0])
    Y_test = Y[shuf_test_idx]
    Y_train = Y[shuf_train_idx]
    X_test = X[shuf_test_idx]
    X_train = X[shuf_train_idx]
    return X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx


def classifyTripleNegative(df, print_wrong=False):
    # Create labels
    Y = np.zeros(df.shape[0])
    Y[df.pos] = 1
    Y[df.neg] = 0

    X = df[df.columns[['cg' in col for col in df.columns]]].values

    # shuffle and split X and Y into train and test
    """ 
    Keep the following in test:
    1. Observations that switched classification when replacing IHC with FISH
    2. Observations where IHC level and IHC status did not match
    """

    train_idx = (df['neg_pre_fish'] != df['neg']) | \
                (df['pos_pre_fish'] != df['pos'])
    train_idx = train_idx | ((df['her2_ihc'] != df['her2_ihc_level']) &
                             (df['her2_ihc_level'] != -2) &
                             ((df['her2_fish'] == -2) | (df['her2_fish'] == 0)))

    X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx = shuffle_idx(X, Y, train_idx)

    pred_test_her2_svm, pred_train_her2_svm, pred_test_her2_rf, pred_train_her2_rf = classify('triple negative', X_test,
                                                                                              X_train,
                                                                                              Y_test,
                                                                                              Y_train)

    if print_wrong:
        patients_changed_by_fish = df.iloc[np.where((df['neg_pre_fish'] != df['neg']) |
                                                    (df['pos_pre_fish'] != df['pos']))][REL_COLS]
        print("Patients whose label changed by fish update:")
        print(patients_changed_by_fish)

        print("Patients whose er_ihc mismatches pr_ihc and changes label:")
        patients_changed_by_mismatch = df.iloc[np.where((df['pos'] == 1) & (((df['er_ihc'] == 1) &
                                                                             (df['pr_ihc'] == -1)) |
                                                                            ((df['er_ihc'] == -1) &
                                                                             (df['pr_ihc'] == 1))) &
                                                        (df['her2_ihc_and_fish'] == -1))][REL_COLS]

        patients_with_ihc_level_diff = df.iloc[np.where(((df['her2_ihc'] != df['her2_ihc_level']) &
                                                         (df['her2_ihc_level'] != -2) &
                                                         ((df['her2_fish'] == -2) | (df['her2_fish'] == 0))))][REL_COLS]

        print("Patients whose IHC level mismatches status (and no fish used):")
        print(patients_with_ihc_level_diff)



        patients_wrong_test_svm = df.iloc[shuf_test_idx[np.where(pred_test_her2_svm != Y_test)]][REL_COLS]
        patients_wrong_train_svm = df.iloc[shuf_train_idx[np.where(pred_train_her2_svm != Y_train)]][REL_COLS]

        patients_wrong_test_rf = df.iloc[shuf_test_idx[np.where(pred_test_her2_rf != Y_test)]][REL_COLS]
        patients_wrong_train_rf = df.iloc[shuf_train_idx[np.where(pred_train_her2_rf != Y_train)]][REL_COLS]

        patients_wrong_test_rf.index.name = 'patient_name'
        patients_wrong_train_rf.index.name = 'patient_name'
        patients_wrong_test_svm.index.name = 'patient_name'
        patients_wrong_train_svm.index.name = 'patient_name'
        patients_changed_by_fish.index.name = 'patient_name'

        print("Patients changed by fish that we misclassified - svm")
        print(patients_wrong_train_svm.join(patients_changed_by_fish, lsuffix='_new', how='inner'))
        print(patients_wrong_test_svm.join(patients_changed_by_fish, lsuffix='_new', how='inner'))

        print("Patients changed by fish that we misclassified - random forest")
        print(patients_wrong_train_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))
        print(patients_wrong_test_rf.join(patients_changed_by_fish, lsuffix='_new', how='inner'))


def classifyReceptor(df, receptor, print_wrong=False):
    X = df[df.columns[['cg' in col for col in df.columns]]].values

    train_idx = np.zeros(df.shape[0], dtype=np.bool)
    train_idx[np.random.choice(np.arange(df.shape[0]), int(df.shape[0] * 0.8))] = True

    # ER
    Y = np.zeros(df.shape[0])
    Y[df[receptor] == 1] = 1

    X_train, Y_train, X_test, Y_test, shuf_test_idx, shuf_train_idx = shuffle_idx(X, Y, train_idx)

    pred_test_svm, pred_train_svm, pred_test_rf, pred_train_rf = classify(receptor, X_test, X_train, Y_test, Y_train)

    patients_wrong_test_svm = df.iloc[shuf_test_idx[np.where(pred_test_svm != Y_test)]][REL_COLS]
    patients_wrong_train_svm = df.iloc[shuf_train_idx[np.where(pred_train_svm != Y_train)]][REL_COLS]
    patients_wrong_test_svm.index.name = 'patient_name'
    patients_wrong_train_svm.index.name = 'patient_name'

    patients_wrong_test_rf = df.iloc[shuf_test_idx[np.where(pred_test_rf != Y_test)]][REL_COLS]
    patients_wrong_train_rf = df.iloc[shuf_train_idx[np.where(pred_train_rf != Y_train)]][REL_COLS]
    patients_wrong_test_rf.index.name = 'patient_name'
    patients_wrong_train_rf.index.name = 'patient_name'

    if receptor in ['er_ihc', 'pr_ihc'] and print_wrong:
        er_pr_mismatch = df.iloc[np.where(((df['er_ihc'] == 1) & (df['pr_ihc'] == -1)) |
                                          ((df['er_ihc'] == -1) & (df['pr_ihc'] == 1)))][REL_COLS]
        er_pr_mismatch.index.name = 'patient_name'
        other_receptor = 'er_ihc' if receptor == 'pr_ihc' else 'pr_ihc'
        print("%s wrong pred in mismatch with %s in svm" % (receptor, other_receptor))
        print(patients_wrong_train_svm.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print(patients_wrong_test_svm.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print("%s wrong pred in mismatch with %s in rf" % (receptor, other_receptor))
        print(patients_wrong_train_rf.join(er_pr_mismatch, lsuffix='_new', how='inner'))
        print(patients_wrong_test_rf.join(er_pr_mismatch, lsuffix='_new', how='inner'))


if __name__ == '__main__':
    final_df = readData()
    getMismatches(final_df)
    df_clinical = fixMismatches(final_df)
    classifyTripleNegative(df_clinical)
    classifyReceptor(df_clinical, 'er_ihc')
    classifyReceptor(df_clinical, 'pr_ihc')
    classifyReceptor(df_clinical, 'her2')




