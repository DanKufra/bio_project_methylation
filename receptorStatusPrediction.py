from utils import*
from sklearn.svm import SVC

DIAG_MAPPING = {'Positive' : 1, 'Negative': -1,
                'Indeterminate': 0, 'Equivocal': 0, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}

LEVEL_SCORE_MAPPING = {'0' : -1, '1+': -1, '2+': 0, '3+': 1, '[Not Available]': -2, 'Not Performed': -2, '[Not Evaluated]' : -2}
# read BRCA patients prognosis
df_BRCA_diagnosis = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/BRCA.tsv', delimiter='\t')
df_BRCA_diagnosis.set_index('bcr_patient_barcode', inplace=True)
df_BRCA_diagnosis.drop(['bcr_patient_barcode','CDE_ID:2003301'], inplace=True)

# read BRCA patients matching
df_id_matching = pd.read_csv('/cs/cbio/dank/BRCA_TCGA_data/sample_sheet_BRCA.tsv', delimiter='\t')

# read methylation data
# df_healthy = read_data_from_tsv('/cs/cbio/tommy/TCGA/BRCA_Solid_Tissue_Normal.tsv.gz')
df_sick = read_data_from_tsv('/cs/cbio/dank/BRCA_TCGA_data/BRCA_Primary_Tumor.tsv.gz')


# match to methylation data
# df_joined = df_id_matching.join(pd.concat([df_sick.T, df_healthy.T]), on='Array.Data.File', how='inner')
df_joined = df_id_matching.join(df_sick.T, on='Sample ID', how='inner')
# df_joined = df_id_matching

df_joined.drop(['File ID', 'File Name', 'Data Category', 'Data Type', 'Project ID','Sample ID', 'Sample Type'], axis=1, inplace=True)
df_joined.set_index('Case ID', inplace=True)

final_df = df_BRCA_diagnosis[['er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'her2_fish_status', 'her2_ihc_score']].join(df_joined, how='inner')

# replace string labels with int for ease of parsing
final_df['er_ihc'] = final_df['er_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['er_status_by_ihc'])
final_df['pr_ihc'] = final_df['pr_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['pr_status_by_ihc'])
final_df['her2_ihc'] = final_df['her2_status_by_ihc'].map(DIAG_MAPPING).fillna(final_df['her2_status_by_ihc'])
final_df['her2_fish'] = final_df['her2_fish_status'].map(DIAG_MAPPING).fillna(final_df['her2_fish_status'])

# create a vector of her2_ihc_level which is the status based on the score we had
final_df['her2_ihc_level'] = final_df['her2_ihc_score'].map(LEVEL_SCORE_MAPPING).fillna(final_df['her2_ihc_score'])

# Give some summary statistics

# How many of each measurement type we have

# Crosstab between her2 ihc status and fish status
pd.crosstab(final_df['her2_ihc'], final_df['her2_fish'])

# get list of patients where her2 fish status does not match ihc status
fish_vs_ihc = final_df[((final_df['her2_ihc'] == 1) & (final_df['her2_fish'] == -1)) | ((final_df['her2_ihc'] == -1) & (final_df['her2_fish'] == 1))]

# Crosstab between her2 ihc status and ihc score
pd.crosstab(final_df['her2_ihc'], final_df['her2_ihc_level'])

# get list of patients where her2 ihc level does not match ihc status
ihc_status_vs_ihc_level = final_df[((final_df['her2_ihc'] == 1) & (final_df['her2_ihc_level'] == -1)) | ((final_df['her2_ihc'] == -1) & (final_df['her2_ihc_level'] == 1))]

# Replacing HER2 (IHC) status by HER2 (FISH) (a more accurate test) whenever available:
final_df['her2_ihc_and_fish'] = final_df['her2_ihc']
final_df.loc[final_df['her2_fish'] != -2,'her2_ihc_and_fish'] = final_df.loc[final_df['her2_fish'] != -2, 'her2_fish']

# Crosstab between her2 ihc status and ihc status with fish
pd.crosstab(final_df['her2_ihc'], final_df['her2_ihc_and_fish'])


# create Triple Negative vs not Triple Negative dataset
df_clinical = pd.DataFrame(final_df[np.hstack([final_df.columns[['cg' in col for col in final_df.columns]],['er_ihc', 'pr_ihc', 'her2_ihc_and_fish', 'her2_ihc', 'her2_fish', 'her2_ihc_level']])])

df_clinical['neg'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (df_clinical['her2_ihc_and_fish'] == -1)
df_clinical['pos'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (df_clinical['her2_ihc_and_fish'] == 1)
df_clinical['NA'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (df_clinical['her2_ihc_and_fish'] == -2)

df_clinical['neg_pre_fish'] = (df_clinical['er_ihc'] == -1) & (df_clinical['pr_ihc'] == -1) & (df_clinical['her2_ihc'] == -1)
df_clinical['pos_pre_fish'] = (df_clinical['er_ihc'] == 1) | (df_clinical['pr_ihc'] == 1) | (df_clinical['her2_ihc'] == 1)
df_clinical['NA_pre_fish'] = (df_clinical['er_ihc'] == -2) | (df_clinical['pr_ihc'] == -2) | (df_clinical['her2_ihc'] == -2)

# drop patients who are NA and not pos
# df_clinical.drop(df_clinical[df_clinical['NA'] & np.logical_not(df_clinical['pos'])].index, axis=0, inplace=True)
df_clinical.drop(df_clinical[df_clinical['NA'] == True].index, axis=0, inplace=True)
# Create labels
Y = np.zeros(df_clinical.shape[0])
Y[df_clinical.pos] = 1
Y[df_clinical.neg] = 0

X = df_clinical[df_clinical.columns[['cg' in col for col in df_clinical.columns]]].as_matrix()

# shuffle and split X and Y into train and test

""" 
Keep the following in test:
1. Observations that switched classification when replacing IHC with FISH
2. Observations where IHC level and IHC status did not match
"""

train_idx =  (df_clinical['neg_pre_fish'] != df_clinical['neg']) |  \
             (df_clinical['pos_pre_fish'] != df_clinical['pos'])
train_idx =  train_idx | ((df_clinical['her2_ihc'] != df_clinical['her2_ihc_level']) &
                          (df_clinical['her2_ihc_level'] != -2) &
                          ((df_clinical['her2_fish'] == -2) | (df_clinical['her2_fish'] == 0)))

# out of remaning indices pick 0.25*Y.shape[0] - test_idx.sum() examples for test
train_idx[np.random.choice(np.arange(0, Y.shape[0])[np.logical_not(train_idx)], np.round(0.75*Y.shape[0] - train_idx.sum()).astype(np.uint32))] = True


shuf_test_idx = np.random.permutation(np.where(~train_idx)[0])
shuf_train_idx = np.random.permutation(np.where(train_idx)[0])
Y_test = Y[shuf_test_idx]
Y_train = Y[shuf_train_idx]
X_test = X[shuf_test_idx]
X_train = X[shuf_train_idx]
clf = SVC(class_weight='balanced', kernel='linear')
clf.fit(X_train, Y_train)

pred_test = clf.predict(X_test)
pred_train = clf.predict(X_train)

rel_cols = ['er_ihc', 'pr_ihc', 'her2_ihc', 'her2_fish', 'her2_ihc_level', 'pos', 'her2_ihc_and_fish']
patients_changed_by_fish = df_clinical.iloc[np.where((df_clinical['neg_pre_fish'] != df_clinical['neg']) |
                                                     (df_clinical['pos_pre_fish'] != df_clinical['pos']))][rel_cols]

patients_with_ihc_level_diff = df_clinical.iloc[np.where(((df_clinical['her2_ihc'] != df_clinical['her2_ihc_level']) &
                                                          (df_clinical['her2_ihc_level'] != -2) &
                                                          ((df_clinical['her2_fish'] == -2) | (df_clinical['her2_fish'] == 0))))][rel_cols]

patients_wrong_test = df_clinical.iloc[shuf_test_idx[np.where(pred_test != Y_test)]][rel_cols]
patients_wrong_train = df_clinical.iloc[shuf_train_idx[np.where(pred_train != Y_train)]][rel_cols]

clf = RandomForestClassifier(random_state=666, max_depth=3, n_estimators=10)
clf = clf.fit(X_train, Y_train)
pred_rf_test = clf.predict(X_test)
pred_rf_train = clf.predict(X_train)

patients_wrong_test_rf = df_clinical.iloc[shuf_test_idx[np.where(pred_rf_test != Y_test)]][rel_cols]
patients_wrong_train_rf = df_clinical.iloc[shuf_train_idx[np.where(pred_rf_train != Y_train)]][rel_cols]

pred_rf_train.index.name = 'patient_name'
pred_rf_test.index.name = 'patient_name'
patients_wrong_test_rf.index.name = 'patient_name'
patients_wrong_train_rf.index.name = 'patient_name'
patients_changed_by_fish.index.name = 'patient_name'
patients_wrong_train_rf.join(patients_changed_by_fish, lsuffix='new', how='inner')

import pdb
pdb.set_trace()




#['er_ihc', 'pr_ihc', 'her2_ihc', 'her2_fish', 'her2_ihc_level']

# weird_combinations = [[ 1,   -1,   1],
#                       [-1,    1,   1],
#                       [ 1,   -1,  -1],
#                       [-1,    1,  -1],
#                       [ 1,    0,   1],
#                       [ 1,    0,  -1],
#                       [ 0,    1,   1],
#                       [ 0,    1,  -1],
#                       [ 1,    1,   0],
#                       [-1,   -1,   0],
#                       [ 1,   -1,   0],
#                       [-1,    1,   0],
#                       [ 0,    0,   0],
#                       [ 0,    0,   1],
#                       [ 0,    0,  -1]]
#
# # find weird stuff (i.e er differs from pr and her2_ihc differs from her2_fish)
# for comb in weird_combinations:
#     num_samples = final_df[(final_df['er_status_by_ihc'] == comb[0]) & (final_df['pr_status_by_ihc'] == comb[1]) & (final_df['her2_status_by_ihc'] == comb[2])].shape[0]
#     print("%f samples in weird combination: er_status=%f,pr_status=%f,her2_status=%f"% (num_samples, comb[0], comb[1], comb[2]))
# print("##############################################")
#
# combinations = [[ 1,   1,   1],
#                 [ 1,   1,  -1],
#                 [ 1,  -1,   1],
#                 [ 1,  -1,  -1],
#                 [-1,   1,   1],
#                 [-1,   1,  -1],
#                 [-1,  -1,   1],
#                 [-1,  -1,  -1]]
# # get some statistics on different amounts of cancer types we have
# for comb in combinations:
#     num_samples = final_df[(final_df['er_status_by_ihc'] == comb[0]) &
#                            (final_df['pr_status_by_ihc'] == comb[1]) &
#                            ((final_df['her2_status_by_ihc'] == comb[2]) | ((final_df['her2_status_by_ihc'] == 0) & (final_df['her2_fish_status'] == comb[2])))].shape[0]
#     print("%f samples in combination: er_status=%f,pr_status=%f,her2_status=%f"% (num_samples, comb[0], comb[1], comb[2]))

# 9.000000 samples in weird combination: er_status=1.000000,pr_status=-1.000000,her2_status=1.000000
# 3.000000 samples in weird combination: er_status=-1.000000,pr_status=1.000000,her2_status=1.000000
# 50.000000 samples in weird combination: er_status=1.000000,pr_status=-1.000000,her2_status=-1.000000
# 8.000000 samples in weird combination: er_status=-1.000000,pr_status=1.000000,her2_status=-1.000000
# 0.000000 samples in weird combination: er_status=1.000000,pr_status=0.000000,her2_status=1.000000
# 1.000000 samples in weird combination: er_status=1.000000,pr_status=0.000000,her2_status=-1.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=1.000000,her2_status=1.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=1.000000,her2_status=-1.000000
# 160.000000 samples in weird combination: er_status=1.000000,pr_status=1.000000,her2_status=0.000000
# 49.000000 samples in weird combination: er_status=-1.000000,pr_status=-1.000000,her2_status=0.000000
# 30.000000 samples in weird combination: er_status=1.000000,pr_status=-1.000000,her2_status=0.000000
# 6.000000 samples in weird combination: er_status=-1.000000,pr_status=1.000000,her2_status=0.000000
# 51.000000 samples in weird combination: er_status=0.000000,pr_status=0.000000,her2_status=0.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=0.000000,her2_status=1.000000
# 0.000000 samples in weird combination: er_status=0.000000,pr_status=0.000000,her2_status=-1.000000
# ##############################################
# 78.000000 samples in combination: er_status=1.000000,pr_status=1.000000,her2_status=1.000000  # Luminal B
# 383.000000 samples in combination: er_status=1.000000,pr_status=1.000000,her2_status=-1.000000 # Luminal A
# 17.000000 samples in combination: er_status=1.000000,pr_status=-1.000000,her2_status=1.000000
# 64.000000 samples in combination: er_status=1.000000,pr_status=-1.000000,her2_status=-1.000000
# 4.000000 samples in combination: er_status=-1.000000,pr_status=1.000000,her2_status=1.000000
# 11.000000 samples in combination: er_status=-1.000000,pr_status=1.000000,her2_status=-1.000000
# 19.000000 samples in combination: er_status=-1.000000,pr_status=-1.000000,her2_status=1.000000 # Her2/neu+
# 117.000000 samples in combination: er_status=-1.000000,pr_status=-1.000000,her2_status=-1.000000 # Triple Negative/Basal cell like



# try and fix some weird combinations



